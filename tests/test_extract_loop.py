import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import cache_io
import extract_features_multi as ex
from vlm_datasets import DatasetBundle, select_pope, split_of
from vlm_registry import score_logits

V, L, H = 50, 4, 8
YES, NO = [1, 2], [3, 4]


class FakeBackend:
    """Deterministic stand-in for HFBackend: no GPU, no model."""
    yes_ids, no_ids = YES, NO
    layers, n_entries = [0, 1, 2, 3], 4

    def __init__(self, mass_ok=True):
        self.calls, self.mass_ok = 0, mass_ok

    def decode(self, tid):
        return f"<{tid}>"

    def forward(self, question, image):
        self.calls += 1
        k = int(question.split()[-1])
        rng = np.random.default_rng(k)
        logits = rng.normal(size=V)
        logits[YES[0]] = 6.0 if k % 3 else -2.0               # yes-ish for 2/3 of the questions
        logits[NO[0]] = -2.0 if k % 3 else 6.0
        if not self.mass_ok:
            logits[0] = 30.0                                    # model starts with some other token
        hidden = rng.normal(size=(L, H)).astype(np.float32)
        return hidden, logits.astype(np.float32), 17


def make_bundle(n=8):
    meta = pd.DataFrame(dict(row_idx=np.arange(n), id=[f"id{i}" for i in range(n)], question_id=np.arange(n),
                             image_id=[f"img{i % 4}" for i in range(n)], setting=["random", "popular"] * (n // 2),
                             question=[f"is there item {i}" for i in range(n)],
                             gt_answer=["yes", "no", "no", "yes"] * (n // 4)))
    b = DatasetBundle("pope", meta, lambda r: Image.new("RGB", (4, 4)), [], {})
    return select_pope(b, None, 0)


def test_full_run_merge_and_record_semantics(tmp_path):
    b, be = make_bundle(), FakeBackend()
    stats = ex.run_extraction(be, b, tmp_path / "sh", "fake", checkpoint_every=3, log=lambda *_: None)
    assert stats["n_new"] == 8 and be.calls == 8 and len(ex.list_shards(tmp_path / "sh")) == 3   # 3 + 3 + 2
    row_idx, feats, meta = ex.merge_shards(tmp_path / "sh", b.selected)
    assert list(row_idx) == list(range(8)) and feats.shape == (8, L, H) and feats.dtype == np.float16
    assert (meta.row_idx.to_numpy() == row_idx).all()
    # record <-> independent recomputation for one row
    r = 5
    logits = FakeBackend().forward(f"is there item {r}", None)[1]
    s = score_logits(logits, YES, NO)
    row = meta.iloc[r]
    assert row.pred_answer == ("yes" if s["p_yes"] > s["p_no"] else "no")
    assert row.hallucination == int(row.pred_answer == "yes" and row.gt_answer == "no")
    assert row.p_yes_norm == pytest.approx(s["p_yes_norm"]) and row.answer_mass == pytest.approx(s["p_yes"] + s["p_no"])
    assert row.model_key == "fake" and row.dataset == "pope" and row.n_input_tokens == 17
    assert row.split == split_of(row.image_id)
    # hidden state stored for the right row
    expected = FakeBackend().forward(f"is there item {r}", None)[0].astype(np.float16)
    assert np.array_equal(feats[r], expected)


def test_resume_after_interruption_does_not_redo_or_duplicate(tmp_path):
    b = make_bundle()
    be1 = FakeBackend()
    ex.run_extraction(be1, b, tmp_path / "sh", "fake", checkpoint_every=3, limit=5, log=lambda *_: None)
    assert be1.calls == 5
    be2 = FakeBackend()
    ex.run_extraction(be2, b, tmp_path / "sh", "fake", checkpoint_every=3, log=lambda *_: None)
    assert be2.calls == 3                                     # only the 3 rows that were missing
    row_idx, feats, meta = ex.merge_shards(tmp_path / "sh", b.selected)
    assert list(row_idx) == list(range(8)) and meta.row_idx.is_unique


def test_corrupt_and_stale_shards_are_redone(tmp_path):
    b, sh = make_bundle(), tmp_path / "sh"
    ex.run_extraction(FakeBackend(), b, sh, "fake", checkpoint_every=4, log=lambda *_: None)
    first = ex.list_shards(sh)[0]
    first.write_bytes(first.read_bytes()[:50])                # truncate
    (sh / "shard_00099.tmp.npz").write_bytes(b"junk")        # stale half-written file
    be = FakeBackend()
    ex.run_extraction(be, b, sh, "fake", checkpoint_every=4, log=lambda *_: None)
    assert be.calls == 4                                      # exactly the corrupt shard's rows
    assert not list(sh.glob("*.tmp.npz"))
    assert list(ex.merge_shards(sh, b.selected)[0]) == list(range(8))


def test_float16_overflow_is_refused(tmp_path):
    class Big(FakeBackend):
        def forward(self, q, i):
            h, l, n = super().forward(q, i)
            return h * 1e6, l, n
    with pytest.raises(RuntimeError, match="overflow"):
        ex.run_extraction(Big(), make_bundle(), tmp_path / "sh", "fake", log=lambda *_: None)


def test_verify_flags_low_answer_mass_and_passes_clean_run(tmp_path, capsys):
    for ok, expect_flag in [(True, False), (False, True)]:
        sh = tmp_path / f"sh{ok}"
        b = make_bundle()
        ex.run_extraction(FakeBackend(mass_ok=ok), b, sh, "fake", log=lambda *_: None)
        row_idx, feats, meta = ex.merge_shards(sh, b.selected)
        flags = ex.verify(meta, feats, row_idx, b.selected)
        assert any("answer_mass" in f for f in flags) == expect_flag
    capsys.readouterr()


def test_meta_roundtrip_via_csv_fallback(tmp_path):
    meta = pd.DataFrame(dict(row_idx=[0, 1], a=[1.5, 2.5]))
    p = ex.write_meta(meta, tmp_path / "x_metadata.parquet")     # pyarrow may be absent -> csv
    assert p.exists() and cache_io.read_meta(p).equals(meta)


# ---- the .pt contract between write_pt (writer) and cache_io.load_cache (reader), with a minimal fake torch -----------
class _T:
    def __init__(self, a):
        self.a = a

    def numpy(self):
        return self.a


def _fake_torch():
    m = types.ModuleType("torch")
    m.from_numpy = _T
    m.save = lambda obj, path: open(path, "wb").write(pickle.dumps(obj))
    m.load = lambda path, **kw: pickle.load(open(path, "rb"))
    return m


def test_pt_contract_writer_reader(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    b = make_bundle()
    ex.run_extraction(FakeBackend(), b, tmp_path / "sh", "fake", checkpoint_every=3, log=lambda *_: None)
    row_idx, feats, meta = ex.merge_shards(tmp_path / "sh", b.selected)
    perm = np.array([3, 0, 7, 1, 2, 6, 5, 4])                  # reader must re-align even if meta is shuffled
    f_name, m_name, _ = cache_io.file_names("pope", "fake")
    ex.write_pt(tmp_path / f_name, row_idx, feats, [0, 1, 2, 3], 4)
    ex.write_meta(meta.iloc[perm], tmp_path / m_name)
    c = cache_io.load_cache("pope", "fake", tmp_path)
    assert c["features"].shape == (8, L, H) and c["layers"] == [0, 1, 2, 3] and c["n_entries"] == 4
    assert (c["meta"].row_idx.to_numpy() == row_idx).all()
    for col in ["gt_yes", "pred_yes", "correct", "error", "conf", "raw_conf", "split"]:
        assert col in c["meta"].columns
    assert (c["meta"].conf.between(0.5, 1.0)).all()


def test_legacy_pope_qwen_cache_is_found(tmp_path):
    for n in ("pope_full_features.pt", "pope_metadata.csv"):
        (tmp_path / n).write_bytes(b"x")
    f, m = cache_io.find_cache("pope", "qwen2vl-2b", tmp_path)
    assert f.name == "pope_full_features.pt" and m.name == "pope_metadata.csv"
    with pytest.raises(FileNotFoundError):
        cache_io.find_cache("pope", "smolvlm-500m", tmp_path)
