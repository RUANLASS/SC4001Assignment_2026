import json
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


def run_eval(pope, amber=None, tasks=("A", "B"), n_boot=60, grid=None, kind="logreg", **kw):
    return pe.evaluate_model("synth", pe.View(pope), None if amber is None else pe.View(amber), tasks, kind,
                             grid or pe.GRIDS[kind], 1, n_boot, 0, log=lambda *_: None, **kw)


def write_caches(tmp_path, monkeypatch, models=(("synth", 0),), amber=True, pope_fn=None, amber_fn=None):
    """Write synthetic caches to disk through the real writer (torch replaced by a tiny pickle-based stand-in)."""
    fake = types.ModuleType("torch")
    fake.from_numpy = _T
    fake.save = lambda o, p: open(p, "wb").write(pickle.dumps(o))
    fake.load = lambda p, **kw: pickle.load(open(p, "rb"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    for model, seed in models:
        kinds = [("pope", synth_cache(seed=seed))] + ([("amber", amber_like(seed=seed + 1))] if amber else [])
        for kind, cache in kinds:
            fn = pope_fn if kind == "pope" else amber_fn
            cache = fn(cache) if fn else cache                              # optional hook, e.g. to corrupt held-out labels in a test
            f, m, _ = cache_io.file_names(kind, model)
            ex.write_pt(tmp_path / f, cache["meta"].row_idx.to_numpy(), cache["features"], cache["layers"], L)
            ex.write_meta(cache["meta"], tmp_path / m)


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
    write_caches(tmp_path, monkeypatch, models=(("synth", 0), ("synth2", 5)))     # minimal fake torch inside (see test_extract_loop)
    out = tmp_path / "out"
    layer_df, pope_df, amber_df = pe.run(["synth", "synth2"], tmp_path, out, "logreg", ("B",), 2, 30, True, 0, log=lambda *_: None,
                                             opts=pe.Options(n_boot_paired=30))
    for f in ["layerwise_pope.csv", "pope_test_summary.csv", "transfer_amber.csv", "layerwise_auroc_taskB.png",
              "transfer_taskB.png", "accuracy_vs_coverage.png"]:
        assert (out / f).exists(), f
    assert set(layer_df.model) == {"synth", "synth2"}
    assert (layer_df.layer.isin([0, 2, 4, 3]) | (layer_df.layer == L - 1)).all()      # stride 2 keeps the last layer too


# ====================================================================================================================
# probe-training options (class weights, PCA, CV selection, seeds), leakage guards, combined probe, new tables
# ====================================================================================================================
def _split_rows(meta, task="B"):
    mask, y = pe.task_arrays(meta, task)
    return y, [pe.split_idx(meta, mask, s) for s in ("train", "dev", "test")]


def test_class_weight_option_reaches_the_probe_and_changes_the_scores():
    assert pe.make_probe("logreg", 0.1, class_weight=True).class_weight == "balanced"
    assert pe.make_probe("logreg", 0.1, class_weight=False).class_weight is None
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    y = (X[:, 0] + rng.normal(0, 1, 2000) > 1.6).astype(int)            # rare positive class (~7%)
    s_plain = pe.probe_scores(pe.fit_probe(X, y, "logreg", 1.0, class_weight=False), X)
    s_bal = pe.probe_scores(pe.fit_probe(X, y, "logreg", 1.0, class_weight=True), X)
    assert s_bal.mean() > 2 * s_plain.mean()                            # balanced weights lift the positive-class probability
    assert pe.GRIDS["logreg"] == [1e-4, 1e-3, 1e-2, 1e-1, 1.0] and pe.GRIDS["mlp"] == [1e-3, 1e-2, 1e-1, 1.0]


def test_pca_is_fitted_on_train_rows_only_and_its_size_is_chosen_on_dev():
    c = synth_cache()
    y, (tr, dv, te) = _split_rows(c["meta"])
    X = c["features"][:, PLANTED, :].astype(np.float32)
    f = pe.select_fit(X, y, c["meta"].image_id.to_numpy(), tr, dv, "logreg", pe.make_configs([0.01, 0.1], pca=[2, 8, 16]))
    assert f["pca_k"] in (2, 8, 16) and f["pca"].n_components_ == f["pca_k"]
    assert f["pca"].n_samples_ == len(tr)                               # train rows only (dev/test never seen by the PCA)
    assert np.allclose(f["pca"].mean_, 0, atol=1e-4)                    # centred on the standardised TRAIN data
    assert pe.probe_scores(f, X[te]).shape == (len(te),)
    assert pe.make_configs([1, 2]) == [dict(param=1, pca=None), dict(param=2, pca=None)]
    assert len(pe.make_configs([1, 2], pca=[4, 8])) == 4


def test_selection_never_reads_test_rows():
    """Poisoning the test rows of X with NaN must not change the selected config or raise (selection uses train/dev only)."""
    c = synth_cache()
    y, (tr, dv, te) = _split_rows(c["meta"])
    X = c["features"][:, PLANTED, :].astype(np.float32)
    g = c["meta"].image_id.to_numpy()
    cfgs = pe.make_configs(pe.GRIDS["logreg"])
    clean = pe.select_fit(X, y, g, tr, dv, "logreg", cfgs)
    Xp = X.copy()
    Xp[te] = np.nan
    poisoned = pe.select_fit(Xp, y, g, tr, dv, "logreg", cfgs)
    assert poisoned["param"] == clean["param"] and poisoned["sel_auroc"] == clean["sel_auroc"]
    cv = pe.select_fit(Xp, y, g, tr, dv, "logreg", cfgs, cv_k=3)         # CV mode only uses train+dev rows too
    assert np.isfinite(cv["sel_auroc"])


def test_layer_and_regularisation_choice_is_independent_of_test_labels():
    c = synth_cache()
    a = run_eval(c, tasks=("B",))
    c2 = synth_cache()
    m = c2["meta"]
    te = (m.split == "test").to_numpy()
    m.loc[te, "error"] = 1 - m.loc[te, "error"]                         # destroy the test labels
    b = run_eval(c2, tasks=("B",))
    la, lb = a[0].set_index("layer"), b[0].set_index("layer")
    assert (la.dev_auroc == lb.dev_auroc).all() and (la.param == lb.param).all()    # selection metric unchanged
    assert not np.allclose(la.test_auroc, lb.test_auroc)                             # only the test score reacts
    sel = lambda r: r[1][(r[1].method == "probe (logreg)")].iloc[0].layer
    assert sel(a) == sel(b) == PLANTED


def test_cv_selection_uses_image_disjoint_folds_and_refits_on_train_plus_dev(monkeypatch):
    c = synth_cache()
    y, (tr, dv, te) = _split_rows(c["meta"])
    g = c["meta"].image_id.to_numpy()
    X = c["features"][:, PLANTED, :].astype(np.float32)
    seen = []
    real = pe.fit_probe

    def spy(Xtr, ytr, *a, **k):
        seen.append(len(ytr))
        return real(Xtr, ytr, *a, **k)

    monkeypatch.setattr(pe, "fit_probe", spy)
    f = pe.select_fit(X, y, g, tr, dv, "logreg", pe.make_configs([0.01, 0.1]), cv_k=4)
    tv = np.concatenate([tr, dv])
    assert seen[-1] == len(tv)                                          # final refit uses ALL train+dev rows
    assert len(f["cv_folds"]) == 4 and 0.5 < f["cv_auroc_mean"] <= 1 and f["cv_auroc_std"] >= 0
    idx, oof = f["oof"]
    assert (idx == tv).all() and not np.isnan(oof).any()                # every train+dev row got an out-of-fold score
    from sklearn.model_selection import GroupKFold
    for a, b in GroupKFold(4).split(tv, groups=g[tv]):                  # the folds select_fit builds never split an image
        assert not set(g[tv][a]) & set(g[tv][b])


def test_cv_mode_picks_the_planted_layer_and_reports_fold_std():
    layer_df, pope_df, _, _ = run_eval(synth_cache(), tasks=("B",), opts=pe.Options(cv_k=3))
    assert layer_df.cv_auroc_mean.notna().all() and layer_df.cv_auroc_std.notna().all() and layer_df.dev_auroc.isna().all()
    assert layer_df.loc[layer_df.sel_auroc.idxmax(), "layer"] == PLANTED
    sel = pope_df[pope_df.method == "probe (logreg)"].iloc[0]
    assert sel.layer == PLANTED and sel.auroc > 0.8


def test_combined_probe_beats_logit_only_when_the_signal_is_in_the_hidden_state_and_paired_test_agrees():
    store = {}
    _, pope_df, _, _ = run_eval(synth_cache(), None, tasks=("B",), store=store)
    r = lambda m: pope_df[pope_df.method == m].iloc[0].auroc
    hid, logit, comb = r("probe (logreg)"), r("probe (logit-only logreg)"), r("probe (hidden+logits, logreg)")
    assert hid > logit + 0.1 and comb > logit + 0.1                     # hidden state carries the planted signal, logits do not
    paired, calib, by_set, aurc, reliab = pe.analyse(store, 0, 40, 150)
    row = paired[(paired.probe == "probe (logreg)") & (paired.reference == "probe (logit-only logreg)")].iloc[0]
    assert row.reference_type == "probe" and row["diff"] > 0.1 and row.ci_lo > 0 and row.p_value < 0.05
    base = paired[(paired.reference_type == "output baseline") & (paired.probe == "probe (logreg)")]
    assert len(base) == 4 and (base["diff"] > 0.1).all()                # vs each of the four output baselines


def test_logit_only_probe_wins_when_the_error_is_in_the_output_distribution():
    """Mirror case: remove the hidden-state signal (shuffled labels in the features) but keep an informative confidence."""
    c = synth_cache(shuffle_labels=True)                                # hidden state is uninformative
    m = c["meta"]
    err = (m.pred_answer != m.gt_answer).to_numpy()
    conf = np.where(err, 0.55, 0.95)
    pyes = np.where(m.pred_answer == "yes", conf, 1 - conf)
    m["p_yes"], m["p_no"], m["p_yes_norm"] = pyes * 0.98, (1 - pyes) * 0.98, pyes
    m["margin"] = np.log(pyes / (1 - pyes))
    c["meta"] = cache_io.derive_columns(m.drop(columns=["split"]).assign(split=m.split))
    _, pope_df, _, _ = run_eval(c, None, tasks=("B",))
    r = lambda mm: pope_df[pope_df.method == mm].iloc[0].auroc
    assert r("probe (logit-only logreg)") > 0.95 and r("probe (logreg)") < 0.6
    assert r("probe (hidden+logits, logreg)") > 0.9                      # combined still finds the output signal


def test_platt_recalibration_is_reported_and_improves_ece_of_the_balanced_probe():
    store = {}
    run_eval(synth_cache(), None, tasks=("B",), store=store, opts=pe.Options(recalibrate=True, combined=False))
    d = store[("synth", 0, "B", "POPE test")]
    assert "probe_platt" in d
    _, calib, _, _, reliab = pe.analyse(store, 0, 10, 10)
    g = calib.set_index("method")
    assert g.loc["probe (hidden) + Platt", "ece"] < g.loc["probe (logreg)", "ece"]       # balanced weights bias the raw probabilities
    assert {"model confidence (P(wrong) = 1 - conf)", "probe (logreg)"} <= set(reliab[("synth", "B")])


def test_analyse_tables_have_expected_structure():
    store = {}
    run_eval(synth_cache(), amber_like(), tasks=("A", "B"), store=store)
    paired, calib, by_set, aurc, _ = pe.analyse(store, 0, 30, 60)
    assert {("synth", 0, t, "POPE test") for t in "AB"} <= set(store)
    assert {f"AMBER {s} test" for s in pe.AMBER_SUBSETS if s != "existence"} <= {k[3] for k in store}      # AMBER test entries
    assert set(paired.eval_set) >= {"POPE test", "AMBER attribute test"}
    assert {"diff", "ci_lo", "ci_hi", "p_value", "auc_probe", "auc_baseline"} <= set(paired.columns)
    assert set(by_set.setting) == {"overall", "random", "popular", "adversarial"}
    assert calib.ece.between(0, 1).all() and calib.brier.between(0, 1).all()
    b = aurc[(aurc.eval_set == "POPE test") & (aurc.method == "probe (logreg)")].iloc[0]
    assert b.oracle_aurc <= b.aurc < b.random_aurc and b.auacc == pytest.approx(1 - b.aurc)


def test_mlp_seeds_give_mean_and_std_and_per_seed_values(tmp_path, monkeypatch):
    write_caches(tmp_path, monkeypatch, amber=False)
    out = tmp_path / "mlp"
    _, pope_df, _ = pe.run(["synth"], tmp_path, out, "mlp", ("B",), 4, 20, True, 0, log=lambda *_: None,
                           opts=pe.Options(seeds=(0, 1, 2), n_boot_paired=20, combined=False))
    ss = pd.read_csv(out / "pope_test_seed_summary.csv")
    r = ss[ss.method == "probe (mlp)"].iloc[0]
    assert r.n_seeds == 3 and r.auroc_std >= 0 and len(r.per_seed_auroc.split(";")) == 3
    per_seed = pope_df[pope_df.method == "probe (mlp)"].auroc
    assert r.auroc_mean == pytest.approx(per_seed.mean()) and r.auroc_std == pytest.approx(per_seed.std(ddof=1))
    assert (pope_df.method.str.startswith("baseline")).sum() == 4                     # baselines are not repeated per seed


def test_logreg_ignores_extra_seeds(tmp_path, monkeypatch):
    write_caches(tmp_path, monkeypatch, amber=False)
    msgs = []
    pe.run(["synth"], tmp_path, tmp_path / "o", "logreg", ("B",), 4, 20, True, 0, log=msgs.append,
           opts=pe.Options(seeds=(0, 1, 2), n_boot_paired=20, combined=False))
    assert any("deterministic" in m for m in msgs)
    assert len(pd.read_csv(tmp_path / "o" / "pope_test_seed_summary.csv").query("method == 'probe (logreg)'")) == 1


def test_full_run_writes_all_tables_figures_and_run_config(tmp_path, monkeypatch):
    write_caches(tmp_path, monkeypatch)
    out = tmp_path / "out"
    pe.run(["synth"], tmp_path, out, "logreg", ("B", "A"), 2, 30, True, 0, log=lambda *_: None,
           opts=pe.Options(recalibrate=True, n_boot_paired=30, cv_k=3))
    for f in ["layerwise_pope.csv", "pope_test_summary.csv", "pope_test_seed_summary.csv", "transfer_amber.csv", "paired_tests.csv",
              "calibration.csv", "pope_by_setting.csv", "aurc.csv", "run_config.json", "layerwise_auroc_taskA.png",
              "layerwise_auroc_taskB.png", "transfer_taskB.png", "accuracy_vs_coverage.png", "probe_reliability_taskA.png",
              "probe_reliability_taskB.png"]:
        assert (out / f).exists(), f
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["options"]["cv_k"] == 3 and cfg["probe"] == "logreg" and "pope/synth" in cfg["cache_files"]
    amb = pd.read_csv(out / "transfer_amber.csv")
    assert {"probe POPE->AMBER", "probe (logit-only logreg) POPE->AMBER", "probe (hidden+logits, logreg) POPE->AMBER"} <= set(amb.method)


def test_missing_cache_stops_instead_of_falling_back_to_fake_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        pe.run(["qwen2vl-2b"], tmp_path, tmp_path / "o", "logreg", ("B",), 1, 10, True, 0, log=lambda *_: None)


def test_no_mock_or_random_label_code_remains_in_the_results_pipeline():
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    bad = re.compile(r"generate_mock|mock_features|mock_labels|Mock probe|np\.random\.(seed|uniform|randint)\(")
    for f in src.glob("*.py"):                                           # src/old/ is excluded on purpose (legacy reference only)
        assert not bad.search(f.read_text()), f"mock/random-label code found in {f.name}"
    assert not (src / "train_probe_regression.py").exists() and not (src / "evaluation_baseline.py").exists()
    assert not (src / "abstention_curves.py").exists()
