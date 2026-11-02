import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest

import cache_io
import extract_features_multi as ex
import probe_transfer_eval as pe
from vlm_datasets import split_of

class _T:
    def __init__(self, a):
        self.a = a

    def numpy(self):
        return self.a


L, H, PLANTED = 5, 24, 3
W = np.random.default_rng(123).normal(size=H)
W /= np.linalg.norm(W)


def synth_cache(kind="pope", n_img=240, per=10, seed=0, shift=0.0, shuffle_labels=False, existence_all_no=False,
                settings=("random", "popular", "adversarial")):
    """Synthetic cache. The 'error' label is planted along W in hidden-state entry PLANTED only."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_img):
        for j in range(per):
            rows.append(dict(image_id=f"{kind}_img{i}", setting=settings[(i + j) % len(settings)]))
    m = pd.DataFrame(rows)
    n = len(m)
    m.insert(0, "row_idx", np.arange(n))
    m["gt_answer"] = np.where(rng.random(n) < 0.5, "yes", "no")
    if existence_all_no:
        m.loc[m.setting == settings[0], "gt_answer"] = "no"
    flip = rng.random(n) < 0.25                                   # 25% of answers are wrong
    m["pred_answer"] = np.where(flip, np.where(m.gt_answer == "yes", "no", "yes"), m.gt_answer)
    m["hallucination"] = ((m.pred_answer == "yes") & (m.gt_answer == "no")).astype(int)
    err = (m.pred_answer != m.gt_answer).astype(int).to_numpy()
    conf = np.clip(0.75 + 0.05 * (1 - err) + rng.normal(0, 0.12, n), 0.5, 0.999)   # weakly informative baseline
    pyes = np.where(m.pred_answer == "yes", conf, 1 - conf)
    m["p_yes"], m["p_no"] = pyes * 0.98, (1 - pyes) * 0.98
    m["p_yes_norm"] = pyes
    m["margin"] = np.log(pyes / (1 - pyes))
    m["binary_entropy"] = -(pyes * np.log2(pyes) + (1 - pyes) * np.log2(1 - pyes))
    m["vocab_entropy"] = rng.random(n)
    m["answer_mass"] = 0.98
    m["split"] = [split_of(i) for i in m.image_id]
    lab = rng.permutation(err) if shuffle_labels else err
    F = rng.normal(size=(n, L, H)).astype(np.float32)
    F[:, PLANTED, :] += 1.6 * lab[:, None] * W[None, :] + shift
    return dict(features=F.astype(np.float16), meta=cache_io.derive_columns(m), layers=list(range(L)), n_entries=L,
                dataset=kind, model_key="synth")


def amber_like(seed=1, shift=0.7):
    c = synth_cache("amber", n_img=120, per=15, seed=seed, shift=shift, existence_all_no=True,
                    settings=("existence", "attribute", "relation"))
    return c


def run_eval(pope, amber=None, tasks=("A", "B"), n_boot=60, grid=None):
    return pe.evaluate_model("synth", pe.View(pope), None if amber is None else pe.View(amber), tasks, "logreg",
                             grid or pe.GRIDS["logreg"], 1, n_boot, 0, log=lambda *_: None)


@pytest.fixture(scope="module")
def result():
    return run_eval(synth_cache(), amber_like())


def test_selected_layer_is_the_planted_layer_chosen_on_dev(result):
    layer_df, pope_df, _, _ = result
    for task in ("A", "B"):
        d = layer_df[layer_df.task == task]
        sel = pope_df[(pope_df.task == task) & (pope_df.method == "probe (logreg)")].iloc[0]
        assert sel.layer == PLANTED == d.loc[d.dev_auroc.idxmax(), "layer"]
        assert sel.auroc > 0.8
        best_base = pope_df[(pope_df.task == task) & pope_df.method.str.startswith("baseline")].auroc.max()
        assert sel.auroc > best_base + 0.1
        others = d[d.layer != PLANTED].test_auroc
        assert others.between(0.35, 0.65).all()                    # no signal elsewhere -> chance
        assert sel.ci_lo <= sel.auroc <= sel.ci_hi


def test_shuffled_labels_give_chance_level():
    layer_df, pope_df, _, _ = run_eval(synth_cache(shuffle_labels=True), None, tasks=("B",))
    sel = pope_df[pope_df.method == "probe (logreg)"].iloc[0]
    assert 0.38 < sel.auroc < 0.62 and layer_df.test_auroc.between(0.38, 0.62).all()


def test_transfer_to_shifted_domain_and_degenerate_existence(result):
    _, _, amber_df, _ = result
    g = lambda task, sub, ev, meth: amber_df[(amber_df.task == task) & (amber_df.subset == sub) & (amber_df.eval_set == ev)
                                             & (amber_df.method == meth)]
    for sub in ("attribute", "relation"):
        t = g("B", sub, "all", "probe POPE->AMBER").iloc[0]
        assert t.auroc > 0.75 and t.ci_lo <= t.auroc <= t.ci_hi      # signal survives a mean shift (scaler fit on POPE-train)
        assert len(g("B", sub, "test", "probe AMBER->AMBER")) == 1   # in-domain ceiling is computed
    # In this synthetic fixture the 'existence' subset has only 'no' truths (real AMBER-existence is 50/50 yes/no; this is the
    # degenerate edge case) -> Task A has a single class -> undefined, not a crash
    assert np.isnan(g("A", "existence", "all", "probe POPE->AMBER").auroc.iloc[0])
    assert g("A", "existence", "test", "probe AMBER->AMBER").empty
    # Task B on existence is defined (an error is exactly a 'yes' answer)
    assert not np.isnan(g("B", "existence", "all", "probe POPE->AMBER").auroc.iloc[0])


def test_layerwise_transfer_columns_present(result):
    layer_df = result[0]
    assert {"transfer_existence", "transfer_attribute", "transfer_relation"} <= set(layer_df.columns)
    d = layer_df[layer_df.task == "B"].set_index("layer")
    assert d.loc[PLANTED, "transfer_attribute"] > 0.75 and d.loc[0, "transfer_attribute"] < 0.65


def test_auc_ci_clusters_and_edge_cases():
    rng = np.random.default_rng(0)
    y = np.repeat(rng.integers(0, 2, 40), 5)                          # 40 images x 5 identical-label questions
    s = y + rng.normal(0, 1.0, len(y))
    groups = np.repeat(np.arange(40), 5)
    r = pe.auc_ci(y, s, groups, n_boot=200)
    assert r["ci_lo"] < r["auroc"] < r["ci_hi"] and r["n"] == 200
    assert np.isnan(pe.auc_ci(np.ones(10), rng.random(10), np.arange(10))["auroc"])      # single class
    assert np.isnan(pe.auc_ci([], [], [])["auroc"])


def test_acc_coverage():
    correct = np.array([1, 1, 1, 0, 0])
    cov, acc = pe.acc_coverage(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), correct)      # perfect error ranking
    assert acc[:3].tolist() == [1, 1, 1] and cov[-1] == 1 and acc[-1] == pytest.approx(0.6)


def test_row_restriction_view_only_uses_kept_rows():
    c = synth_cache(n_img=20, per=5)
    v = pe.View(c, keep_rows={0, 1, 2, 50})
    assert list(v.meta.row_idx) == [0, 1, 2, 50] and v.X([3], PLANTED).shape == (1, H)
    assert np.allclose(v.X([3], 2), c["features"][50, 2, :].astype(np.float32))


def test_end_to_end_run_with_files(tmp_path, monkeypatch):
    fake = types.ModuleType("torch")           # minimal fake torch (see test_extract_loop)
    fake.from_numpy = _T
    fake.save = lambda o, p: open(p, "wb").write(pickle.dumps(o))
    fake.load = lambda p, **kw: pickle.load(open(p, "rb"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    for model, seed in [("synth", 0), ("synth2", 5)]:
        for kind, cache in [("pope", synth_cache(seed=seed)), ("amber", amber_like(seed=seed + 1))]:
            f, m, _ = cache_io.file_names(kind, model)
            ex.write_pt(tmp_path / f, cache["meta"].row_idx.to_numpy(), cache["features"], cache["layers"], L)
            ex.write_meta(cache["meta"], tmp_path / m)
    out = tmp_path / "out"
    layer_df, pope_df, amber_df = pe.run(["synth", "synth2"], tmp_path, out, "logreg", ("B",), 2, 30, True, 0, log=lambda *_: None)
    for f in ["layerwise_pope.csv", "pope_test_summary.csv", "transfer_amber.csv", "layerwise_auroc_taskB.png",
              "transfer_taskB.png", "accuracy_vs_coverage.png"]:
        assert (out / f).exists(), f
    assert set(layer_df.model) == {"synth", "synth2"}
    assert (layer_df.layer.isin([0, 2, 4, 3]) | (layer_df.layer == L - 1)).all()      # stride 2 keeps the last layer too
