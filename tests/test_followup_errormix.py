"""Tests for src/followup_errormix.py (error-mix follow-up): weights, routing, E0 split AUROC, leakage guards, reproduction."""
import json

import numpy as np
import pandas as pd
import pytest

import followup_errormix as fu
import probe_transfer_eval as pe
from test_probe_eval import PLANTED, synth_cache, write_caches


# ------------------------------------------------------------------------------------------------ error_type_weights
def _toy(n_neg=60, n_h=6, n_m=18):
    y = np.r_[np.zeros(n_neg), np.ones(n_h + n_m)].astype(int)
    pyes = np.r_[np.zeros(n_neg), np.ones(n_h), np.zeros(n_m)].astype(int)          # negatives are all "no" here: irrelevant for weights
    return y, pyes


def test_weights_balance_classes_and_set_the_error_mix():
    y, py = _toy()
    for share in (0.5, 0.1, 0.9, 0.0, 1.0):
        w = fu.error_type_weights(y, py, share)
        pos_h, pos_m = (y == 1) & (py == 1), (y == 1) & (py == 0)
        assert w[y == 0].tolist() == [1.0] * (y == 0).sum()                         # correct answers keep weight 1
        assert w[y == 1].sum() == pytest.approx((y == 0).sum())                      # positive total == negative total (balanced)
        assert w[pos_h].sum() == pytest.approx(share * (y == 0).sum())               # hallucinated-yes share of the positive weight
        assert w[pos_m].sum() == pytest.approx((1 - share) * (y == 0).sum())


def test_natural_share_reproduces_sklearn_balanced_class_weights():
    y, py = _toy()
    w = fu.error_type_weights(y, py, share := ((y == 1) & (py == 1)).sum() / y.sum())
    bal = len(y) / (2 * np.bincount(y))                                              # sklearn's class_weight="balanced"
    assert np.allclose(w / w[y == 0][0], np.where(y == 1, bal[1] / bal[0], 1.0))
    assert share == pytest.approx(0.25)


def test_weights_reject_impossible_requests():
    y, py = _toy(n_h=0)
    with pytest.raises(ValueError):
        fu.error_type_weights(y, py, 0.5)                                            # wants hallucinated-yes weight but there are none
    assert fu.error_type_weights(y, py, 0.0)[y == 1].sum() == pytest.approx((y == 0).sum())   # fine when that type has zero target share
    with pytest.raises(ValueError):
        fu.error_type_weights(*_toy(), 1.5)


# ------------------------------------------------------------------------------------------------ AMBER-train share
def test_amber_train_share_uses_the_train_split_only():
    m = pd.DataFrame(dict(setting="attribute", split=["train"] * 4 + ["dev"] * 3 + ["test"] * 3,
                          error=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], pred_yes=[1, 1, 0, 0, 1, 1, 1, 1, 1, 1]))   # train: 2 halluc, 2 missed
    assert fu.amber_train_halluc_share(m, "attribute") == pytest.approx(0.5)         # dev/test rows (all hallucinated) are ignored
    m["error"] = 0
    with pytest.raises(ValueError):
        fu.amber_train_halluc_share(m, "attribute")


# ------------------------------------------------------------------------------------------------ routing / E0
def test_route_scores_picks_the_probe_that_matches_the_answer():
    r = fu.route_scores([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4])
    assert r.tolist() == [0.9, 0.2, 0.7, 0.4]


def test_answer_split_auc_separates_the_two_answer_groups_and_the_flip():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    py = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([.9, .1, .8, .2, .1, .9, .2, .8])        # right way round for answered-yes rows, reversed for answered-no rows
    g = np.arange(8)
    got = {r["rows"]: r["auroc"] for r in fu.answer_split_auc(y, score, py, g, n_boot=0)}
    assert got["answered yes"] == 1.0 and got["answered no"] == 0.0 and got["all rows"] == pytest.approx(0.5)
    flipped = {r["rows"]: r["auroc"] for r in fu.answer_split_auc(y, 1 - score, py, g, n_boot=0)}
    assert flipped["answered yes"] == 0.0 and flipped["answered no"] == 1.0
    n = {r["rows"]: (r["n"], r["n_pos"]) for r in fu.answer_split_auc(y, score, py, g, n_boot=0)}
    assert n == {"all rows": (8, 4), "answered yes": (4, 2), "answered no": (4, 2)}


def test_trivial_answered_no_score_exploits_the_error_mix():
    """If errors are mostly missed objects, 'answered no' alone ranks errors well; if mostly hallucinated yes, it ranks them badly."""
    rng = np.random.default_rng(0)
    py = rng.integers(0, 2, 4000)
    y_missed = ((py == 0) & (rng.random(4000) < 0.4)).astype(int) | ((py == 1) & (rng.random(4000) < 0.03)).astype(int)
    y_halluc = ((py == 1) & (rng.random(4000) < 0.4)).astype(int) | ((py == 0) & (rng.random(4000) < 0.03)).astype(int)
    a = lambda y: fu.answer_split_auc(y, (py == 0).astype(float), py, np.arange(4000), 0)[0]["auroc"]
    assert a(y_missed) > 0.7 and a(y_halluc) < 0.3


# ------------------------------------------------------------------------------------------------ weights reach the probe
def test_sample_weights_reach_the_fit_and_the_dev_metric(monkeypatch):
    c = synth_cache()
    y = c["meta"]["error"].to_numpy()
    tr, dv = (np.flatnonzero((c["meta"].split == s).to_numpy()) for s in ("train", "dev"))
    X = c["features"][:, PLANTED, :].astype(np.float32)
    seen = []                                                                         # sample_weight handed to every fit_probe call
    real = pe.fit_probe

    def spy(*a, **k):
        seen.append(a[7] if len(a) > 7 else k.get("sample_weight"))
        return real(*a, **k)
    monkeypatch.setattr(pe, "fit_probe", spy)
    w = np.zeros(len(y))
    w[tr] = np.where(y[tr] == 1, 3.0, 1.0)
    w[dv] = np.where(y[dv] == 1, 3.0, 1.0)
    f_w = pe.select_fit(X, y, np.arange(len(y)), tr, dv, "logreg", pe.make_configs([0.01]), 0, False, 0, w)
    f_u = pe.select_fit(X, y, np.arange(len(y)), tr, dv, "logreg", pe.make_configs([0.01]), 0, False, 0, None)
    assert np.array_equal(seen[0], w[tr]) and seen[1] is None                         # weighted fit got w[tr]; unweighted fit got None
    assert not np.allclose(f_w["clf"].coef_, f_u["clf"].coef_)                       # weights change the fitted probe
    assert f_w["dev_auroc"] == pytest.approx(pe.safe_auc(y[dv], pe.probe_scores(f_w, X[dv]), w[dv]))
    with pytest.raises(ValueError):
        pe.fit_probe(X[tr], y[tr], "mlp", 0.1, sample_weight=w[tr])
    with pytest.raises(NotImplementedError):
        pe.select_fit(X, y, np.arange(len(y)), tr, dv, "logreg", pe.make_configs([0.01]), 0, False, 3, w)


# ------------------------------------------------------------------------------------------------ end to end on synthetic caches
@pytest.fixture(scope="module")
def followup(tmp_path_factory):
    import sys, types
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("fu")
    write_caches(root, mp, models=(("synth", 0),))
    ref = root / "ref"
    pe.run(["synth"], root, ref, "logreg", ("B",), 1, 20, True, 0, log=lambda *_: None, opts=pe.Options(n_boot_paired=10, combined=False))
    out = root / "out"
    fu.main(["--models", "synth", "--cache-dir", str(root), "--out-dir", str(out), "--n-boot", "20", "--n-boot-paired", "20",
             "--reference-dirs", str(ref)])
    yield root, out / "synth"
    mp.undo()


def test_outputs_exist_and_prediction_is_logged_first(followup):
    root, out = followup
    for f in ["e0_sign_diagnostic.csv", "e1_reweight.csv", "e2_answer_conditioned.csv", "followup_summary.csv", "layerwise_e1a_e2.csv",
              "followup_transfer_taskB.png", "command.txt", "run_config.json", "run.log"]:
        assert (out / f).exists(), f
    log = (out / "run.log").read_text()
    assert log.startswith("PREDICTION (stated before any result)") and log.index("PREDICTION") < log.index("selected")
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["prediction"] == fu.PREDICTION and set(cfg["selected"]) >= {"orig", "e1a", "e2_yes", "e2_no", "e1b_existence"}


def test_original_probe_reproduces_the_main_pipeline_exactly(followup):
    _, out = followup
    cfg = json.loads((out / "run_config.json").read_text())
    assert len(cfg["reproduction_check"]) == 3
    for v in cfg["reproduction_check"].values():
        assert v["this_run_auroc"] == pytest.approx(v["reference_auroc"], abs=1e-12)


def test_summary_table_has_every_variant_with_n_npos_and_paired_columns(followup):
    _, out = followup
    d = pd.read_csv(out / "followup_summary.csv")
    assert set(d.experiment) == {"baseline", "orig", "E1a", "E1b", "E2", "E2 sub-probe", "ceiling"}
    assert set(d.eval_set) == {"POPE test", "AMBER existence test", "AMBER attribute test", "AMBER relation test"}
    assert d[["n", "n_pos"]].notna().all().all()
    r = d[(d.experiment == "E1a") & (d.eval_set == "AMBER attribute test")].iloc[0]
    assert {"diff_vs_baseline_lo", "diff_vs_baseline_hi", "diff_vs_baseline_p", "diff_vs_orig", "diff_vs_orig_p", "best_baseline"} <= set(d.columns)
    assert not r.uses_target_domain_stats
    assert d[d.experiment == "E1b"].uses_target_domain_stats.all() and d[d.experiment == "ceiling"].uses_target_domain_stats.all()
    assert (d[d.experiment == "E1b"].variant.str.contains("target-domain")).all()    # E1b is always labelled as such
    e1b_pope = d[(d.experiment == "E1b") & (d.eval_set == "POPE test")]
    assert len(e1b_pope) == 3                                                         # POPE-test cost of each per-subset probe
    e2s = d[d.experiment == "E2 sub-probe"]
    assert set(e2s.variant.str.extract(r"(yes|no)-probe")[0]) == {"yes", "no"}


def test_e0_has_the_split_by_answer_rows_for_probe_flip_and_trivial_score(followup):
    _, out = followup
    e0 = pd.read_csv(out / "e0_sign_diagnostic.csv")
    assert set(e0.score) == {"original POPE probe", "1 - original probe", "trivial: model answered no"}
    assert set(e0.rows) == {"all rows", "answered yes", "answered no"} and set(e0.eval_set) >= {"POPE test", "AMBER attribute test"}
    a = e0[(e0.eval_set == "POPE test") & (e0.rows == "all rows")].set_index("score").auroc
    assert a["original POPE probe"] + a["1 - original probe"] == pytest.approx(1.0)  # flipping the score flips the AUROC


def test_selection_never_uses_pope_test_or_amber_dev_test_labels(tmp_path, monkeypatch):
    """Corrupt the labels of POPE test rows and of AMBER dev+test rows: layers / C / shares chosen must not change."""
    def flip_rows(cache, splits):
        m = cache["meta"].copy()
        sel = m["split"].isin(splits).to_numpy()
        m.loc[sel, "pred_answer"] = np.where(m.loc[sel, "pred_answer"] == "yes", "no", "yes")
        m["hallucination"] = ((m.pred_answer == "yes") & (m.gt_answer == "no")).astype(int)
        return dict(cache, meta=m)
    runs = {}
    for tag, kw in [("clean", {}), ("corrupt", dict(pope_fn=lambda c: flip_rows(c, ["test"]), amber_fn=lambda c: flip_rows(c, ["dev", "test"])))]:
        d = tmp_path / tag
        d.mkdir()
        write_caches(d, monkeypatch, **kw)
        fu.main(["--models", "synth", "--cache-dir", str(d), "--out-dir", str(d / "o"), "--n-boot", "10", "--n-boot-paired", "10", "--reference-dirs"])
        runs[tag] = json.loads((d / "o" / "synth" / "run_config.json").read_text())
    assert runs["clean"]["selected"] == runs["corrupt"]["selected"]
    assert runs["clean"]["e1b_amber_train_halluc_share"] == runs["corrupt"]["e1b_amber_train_halluc_share"]


def test_missing_cache_stops_instead_of_falling_back_to_fake_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        fu.main(["--models", "qwen2vl-2b", "--cache-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--reference-dirs"])
