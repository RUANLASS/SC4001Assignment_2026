import hashlib

import numpy as np
import pandas as pd
import pytest

from vlm_datasets import (AMBER_EXPECTED, AMBER_RANGES, DatasetBundle, amber_report, amber_subset_of, build_amber_meta,
                          seeded_order, select_amber, select_pope, split_of)


def test_split_rule_is_notebook01_rule():
    # literal copy of the formula in notebook 01 / 02
    def nb(image_id, seed=0, fr=(0.70, 0.15, 0.15)):
        u = int(hashlib.md5(f"{seed}-{image_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "train" if u < fr[0] else ("dev" if u < fr[0] + fr[1] else "test")
    ids = [f"COCO_val2014_{i:012d}.jpg" for i in range(2000)]
    assert all(split_of(i) == nb(i) for i in ids)
    frac = pd.Series([split_of(i) for i in ids]).value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.04 and abs(frac["dev"] - 0.15) < 0.04 and abs(frac["test"] - 0.15) < 0.04


def test_seeded_order_matches_notebook01():
    order = np.arange(9000)
    np.random.default_rng(0).shuffle(order)
    assert seeded_order(np.arange(9000), 0) == order.tolist()


def _pope_like(n=60):
    return pd.DataFrame(dict(row_idx=np.arange(n), id=np.arange(n), question_id=np.arange(n),
                             image_id=[f"img{i % 10}" for i in range(n)], setting=["random"] * n,
                             question=["q"] * n, gt_answer=["yes", "no"] * (n // 2)))


def test_select_pope_subsets_are_nested_and_deterministic():
    full = select_pope(DatasetBundle("pope", _pope_like(), None, [], {}), None).selected
    small = select_pope(DatasetBundle("pope", _pope_like(), None, [], {}), 20).selected
    assert small == full[:20] and sorted(full) == list(range(60))


def test_amber_ranges_cover_the_paper_counts():
    for k, (lo, hi) in AMBER_RANGES.items():
        assert hi - lo + 1 == AMBER_EXPECTED[k]
    assert AMBER_RANGES["existence"][0] == 1005 and AMBER_RANGES["relation"][1] == 15220
    assert amber_subset_of(13557) == "relation" and amber_subset_of(5928) == "existence" and amber_subset_of(5929) == "attribute"
    assert amber_subset_of(10) is None                                  # generative query


def _amber_files(per=30, as_dict=False, truth_key="truth"):
    queries, ann = [], []
    for sub, (lo, _) in AMBER_RANGES.items():
        for j in range(per):
            qid = lo + j
            queries.append(dict(id=qid, image=f"AMBER_{(qid % 7) + 1}.jpg", query=f"query {qid}?"))
            ann.append(dict(id=qid, type="discriminative", **{truth_key: "no" if sub == "existence" or j % 2 else "Yes"}))
    queries.append(dict(id=3, image="AMBER_1.jpg", query="describe"))     # generative: must be ignored
    ann.append(dict(id=3, type="generative"))
    return queries, ({str(a["id"]): a for a in ann} if as_dict else ann)


@pytest.mark.parametrize("as_dict", [False, True])
def test_build_amber_meta(as_dict):
    q, a = _amber_files(as_dict=as_dict)
    meta = build_amber_meta(q, a)
    assert len(meta) == 90 and list(meta.row_idx) == list(range(90))
    assert set(meta.setting) == {"existence", "attribute", "relation"}
    assert set(meta.gt_answer) == {"yes", "no"}                          # "Yes" normalised
    assert (meta[meta.setting == "existence"].gt_answer == "no").all()
    assert "existence" in amber_report(meta)


def test_build_amber_meta_real_annotation_format():
    """Entries copied in shape from the real AMBER annotations.json: generative rows carry LIST truths ('truth': [...],
    'hallu': [...]), discriminative rows carry the string 'yes' / 'no'. Existence is balanced in the real data."""
    ann = [dict(id=1, type="generative", truth=["sky", "forest"], hallu=["cloud", "sun"]),
           dict(id=1005, type="discriminative", truth="yes"), dict(id=1006, type="discriminative", truth="no"),
           dict(id=5929, type="discriminative-attribute", truth="no"),
           dict(id=13557, type="discriminative-relation", truth="yes")]
    qs = [dict(id=1005, image="AMBER_1.jpg", query="Is the sky sunny in this image?"),
          dict(id=1006, image="AMBER_1.jpg", query="Is there a cow in this image?"),
          dict(id=5929, image="AMBER_2.jpg", query="Is the dog brown?"),
          dict(id=13557, image="AMBER_3.jpg", query="Is there direct contact between a and b?")]
    meta = build_amber_meta(qs, ann)
    assert meta.setting.tolist() == ["existence", "existence", "attribute", "relation"]
    assert meta.gt_answer.tolist() == ["yes", "no", "no", "yes"]


def test_real_amber_repo_counts_if_available():
    """Optional: set AMBER_REPO=/path/to/clone of github.com/junyangwang0410/AMBER to check the full data (skipped otherwise)."""
    import json
    import os
    from pathlib import Path
    repo = os.environ.get("AMBER_REPO")
    if not repo or not (Path(repo) / "data" / "annotations.json").exists():
        pytest.skip("AMBER_REPO not set")
    d = Path(repo) / "data"
    meta = build_amber_meta(json.loads((d / "query" / "query_discriminative.json").read_text()),
                            json.loads((d / "annotations.json").read_text()))
    assert meta.setting.value_counts().to_dict() == AMBER_EXPECTED
    assert meta[meta.setting == "existence"].gt_answer.value_counts().to_dict() == {"yes": 2462, "no": 2462}
    assert meta.image_id.nunique() == 1004


def test_build_amber_meta_fails_loudly_on_unknown_schema():
    q, a = _amber_files(truth_key="answer")
    with pytest.raises(ValueError, match="truth"):
        build_amber_meta(q, a)


def test_build_amber_meta_fails_on_missing_truth_for_a_query():
    q, a = _amber_files()
    a = [x for x in a if x["id"] != 1010]
    with pytest.raises(KeyError):
        build_amber_meta(q, a)


def test_select_amber_is_stratified_deterministic_and_seeded():
    q, a = _amber_files()
    meta = build_amber_meta(q, a)
    mk = lambda: DatasetBundle("amber", meta, None, [], {})
    b1 = select_amber(mk(), ["existence", "attribute", "relation"], 10, seed=0)
    b2 = select_amber(mk(), ["existence", "attribute", "relation"], 10, seed=0)
    b3 = select_amber(mk(), ["existence", "attribute", "relation"], 10, seed=1)
    assert b1.selected == b2.selected and b1.selected != b3.selected
    sel = meta.set_index("row_idx").loc[b1.selected]
    assert sel.setting.value_counts().to_dict() == {"existence": 10, "attribute": 10, "relation": 10}
    assert len(set(b1.selected)) == 30
    only = select_amber(mk(), ["relation"], None, seed=0)
    assert len(only.selected) == 30
    with pytest.raises(ValueError):
        select_amber(mk(), ["nonexistent"], 5)
