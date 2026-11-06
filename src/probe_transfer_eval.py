#!/usr/bin/env python
"""Cross-model and cross-dataset evaluation of hallucination probes (CPU only; reads cached hidden states).

For every model with a POPE cache (and optionally an AMBER cache) this script
  1. computes training-free output-level baselines (token prob, |margin|, binary / vocab entropy),
  2. trains one probe per layer on POPE-train, picks layer AND regularisation on POPE-dev only, reports POPE-test,
  3. applies the POPE-trained probes, unchanged, to AMBER (transfer) and compares with the baselines on the same rows and
     with a probe trained in-domain on AMBER-train (the "ceiling"),
  4. plots layer-wise AUROC against relative depth (so models with different depths can share one axis).

Two tasks, defined exactly as in notebook 02:
  A  hallucinated-yes vs correct-yes, among questions the model answered "yes"   (label = `hallucination`)
  B  wrong vs right answer, over all questions                                    (label = `error`)
Uncertainty scores: higher = more likely bad. AUROC CIs are percentile bootstraps that resample IMAGES (questions on one
image are not independent).

    python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m qwen25vl-3b --out-dir results/final/run

Probe options (all selection happens on train/dev only; the test split is used once, for the reported numbers):
  --class-weight / --no-class-weight  balanced class weights for the logistic-regression probe (default on)
  --pca 64 128 256                    fit PCA on train only and pick the size on dev (default: no PCA)
  --seeds 0 1 2                       MLP probe: repeat the whole selection per seed and report mean +- std of test AUROC
  --layer-cv K                        select layer/regularisation by image-grouped K-fold CV over train+dev (default: dev split)
  --recalibrate                       also report Platt-scaled probe probabilities (fit on dev / out-of-fold scores)
Extra tables written next to the layer-wise ones: paired_tests.csv (probe vs baselines, paired image bootstrap), calibration.csv,
pope_by_setting.csv, aurc.csv, plus the logit-only and hidden+logits ("combined") probes in every table.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from cache_io import find_cache, load_cache
from probe_stats import (aurc_metrics, brier_score, ece_score, paired_auc_diff, platt_apply, platt_fit,  # noqa: F401
                         reliability_bins)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Task B (wrong vs right) is the primary result: it has ~8x more positives than Task A, so it is listed (and printed) first.
TASKS = {"B": dict(label="error", mask=lambda m: np.ones(len(m), bool), desc="wrong vs right (all questions)"),
         "A": dict(label="hallucination", mask=lambda m: (m["pred_yes"] == 1).to_numpy(),
                   desc="hallucinated-yes vs correct-yes (answered-yes only)")}
AMBER_SUBSETS = ["existence", "attribute", "relation"]
GRIDS = {"logreg": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],               # C for logreg
         "mlp": [1e-3, 1e-2, 1e-1, 1.0]}                         # L2 alpha for mlp (both picked on dev / CV only, never on test)
SIGNAL_NAMES = ["raw_conf", "|margin|", "binary_entropy", "vocab_entropy"]


@dataclass
class Options:
    """Probe-training options (see module docstring). The defaults give the standard dev-split protocol."""
    class_weight: bool = True        # balanced class weights (logistic regression only; sklearn's MLP has no class_weight)
    pca: tuple = ()                  # PCA sizes to choose from on dev; empty = no PCA
    cv_k: int = 0                    # >0: image-grouped K-fold CV over train+dev for selection, then refit on train+dev
    seeds: tuple = (0,)              # MLP init seeds (logreg is deterministic, so only the first seed is used for it)
    recalibrate: bool = False        # Platt scaling of the hidden probe (fit on dev, or out-of-fold scores in CV mode)
    combined: bool = True            # also train logit-only and hidden+logits probes
    n_boot_paired: int = 1000        # bootstrap draws for the paired tests


# --------------------------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------------------------
def unc_signals(m: pd.DataFrame) -> dict:
    """Training-free uncertainty scores, higher = more likely a bad answer (same definitions as notebook 02)."""
    return {"token prob (raw)": (1 - m["raw_conf"]).to_numpy(), "margin (|log-odds|)": (-m["margin"].abs()).to_numpy(),
            "binary entropy": m["binary_entropy"].to_numpy(), "vocab entropy": m["vocab_entropy"].to_numpy()}


def safe_auc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(y) and len(np.unique(y)) == 2 else float("nan")


def auc_ci(y, s, groups, n_boot: int = 500, seed: int = 0) -> dict:
    """AUROC with a 95% percentile CI from resampling whole groups (images)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    out = dict(auroc=safe_auc(y, s), ci_lo=np.nan, ci_hi=np.nan, n=int(len(y)), n_pos=int(y.sum()))
    if np.isnan(out["auroc"]) or n_boot <= 0:
        return out
    codes, uniq = pd.factorize(np.asarray(groups))
    by_group = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[g] for g in rng.integers(0, len(uniq), len(uniq))])
        v = safe_auc(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) >= 20:
        out["ci_lo"], out["ci_hi"] = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    return out


def acc_coverage(err_score, correct):
    """Accuracy of the answered part when abstaining on the highest error-scores first. Returns (coverage, accuracy)."""
    o = np.argsort(np.asarray(err_score), kind="stable")           # most confident first
    c = np.asarray(correct)[o]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k


# --------------------------------------------------------------------------------------------------------------------
# data views (no copy of the big feature array)
# --------------------------------------------------------------------------------------------------------------------
class View:
    """A (possibly row-restricted) window onto a cache. X(idx, layer) copies only [len(idx), H]."""

    def __init__(self, cache: dict, keep_rows=None):
        self.F, self.layers, self.n_entries = cache["features"], cache["layers"], cache["n_entries"]
        meta = cache["meta"]
        self.rows = (np.arange(len(meta)) if keep_rows is None
                     else np.flatnonzero(meta["row_idx"].isin(keep_rows).to_numpy()))
        self.meta = meta.iloc[self.rows].reset_index(drop=True)
        self.pos = {hs: i for i, hs in enumerate(self.layers)}

    def X(self, idx, hs_layer) -> np.ndarray:
        return self.F[self.rows[np.asarray(idx)], self.pos[hs_layer], :].astype(np.float32)


def task_arrays(meta: pd.DataFrame, task: str):
    t = TASKS[task]
    return t["mask"](meta), meta[t["label"]].to_numpy()


def split_idx(meta, mask, split):
    return np.flatnonzero(mask & (meta["split"].to_numpy() == split))


# --------------------------------------------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------------------------------------------
def make_probe(kind: str, param: float, seed: int = 0, class_weight: bool = True):
    if kind == "logreg":
        # class_weight="balanced" reweights the rare positive class (hallucinations / errors); the MLP has no such option
        return LogisticRegression(C=param, max_iter=300, class_weight="balanced" if class_weight else None)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), alpha=param, early_stopping=True, max_iter=200, random_state=seed)
    raise ValueError(kind)


def fit_probe(Xtr, ytr, kind, param, seed=0, class_weight=True, pca=None):
    """Scaler (+ optional PCA) + classifier, every statistic fitted on the training rows only."""
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    pc = None
    if pca:
        pc = PCA(n_components=int(min(pca, Z.shape[1], Z.shape[0] - 1)), random_state=seed).fit(Z)
        Z = pc.transform(Z)
    clf = make_probe(kind, param, seed, class_weight).fit(Z, ytr)
    return dict(scaler=sc, pca=pc, clf=clf, param=param, pca_k=pca)


def probe_scores(fit, X) -> np.ndarray:
    Z = fit["scaler"].transform(X)
    if fit.get("pca") is not None:
        Z = fit["pca"].transform(Z)
    return fit["clf"].predict_proba(Z)[:, 1]


def make_configs(grid, pca=()):
    """Hyper-parameter candidates: regularisation grid x PCA sizes (a single 'no PCA' entry when pca is empty)."""
    return [dict(param=p, pca=k) for k in (list(pca) or [None]) for p in grid]


def signal_matrix(meta: pd.DataFrame) -> np.ndarray:
    """The four output-level signals as raw features [n, 4] (order = SIGNAL_NAMES); scaled inside fit_probe."""
    return np.column_stack([meta["raw_conf"].to_numpy(), meta["margin"].abs().to_numpy(),
                            meta["binary_entropy"].to_numpy(), meta["vocab_entropy"].to_numpy()]).astype(np.float32)


def select_fit(X, y, groups, tr, dv, kind, configs, seed=0, class_weight=True, cv_k=0):
    """Choose the best config and return its fitted probe, using train/dev rows only (test rows are never touched).

    cv_k == 0 : fit every config on `tr`, pick the best dev AUROC (the standard protocol); the returned probe is trained on tr.
    cv_k  > 0 : image-grouped K-fold CV over tr+dev (GroupKFold on `groups`), pick the best mean CV AUROC, then REFIT on tr+dev.
                The out-of-fold scores of the chosen config are returned in fit["oof"] = (row indices, scores).
    Returns None when the needed classes are missing. Keys added to the fit: sel_auroc (the selection metric), dev_auroc,
    cv_auroc_mean, cv_auroc_std, cv_folds.
    """
    nan = float("nan")
    tv = np.concatenate([tr, dv])
    if len(np.unique(y[tv])) < 2:
        return None
    best = None
    if cv_k:
        gk = GroupKFold(n_splits=int(min(cv_k, len(np.unique(groups[tv])))))
        folds = list(gk.split(tv, groups=groups[tv]))
        for cfg in configs:
            aucs, oof = [], np.full(len(tv), np.nan)
            for a, b in folds:
                ia, ib = tv[a], tv[b]
                if len(np.unique(y[ia])) < 2:
                    aucs.append(nan)
                    continue
                f = fit_probe(X[ia], y[ia], kind, cfg["param"], seed, class_weight, cfg["pca"])
                oof[b] = probe_scores(f, X[ib])
                aucs.append(safe_auc(y[ib], oof[b]))
            ok = [v for v in aucs if not np.isnan(v)]
            m = float(np.mean(ok)) if ok else nan
            if not np.isnan(m) and (best is None or m > best["m"]):
                best = dict(m=m, sd=float(np.std(ok)), folds=aucs, cfg=cfg, oof=oof)
        if best is None:
            return None
        f = fit_probe(X[tv], y[tv], kind, best["cfg"]["param"], seed, class_weight, best["cfg"]["pca"])
        return dict(f, sel_auroc=best["m"], dev_auroc=nan, cv_auroc_mean=best["m"], cv_auroc_std=best["sd"],
                    cv_folds=best["folds"], oof=(tv, best["oof"]))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[dv])) < 2:
        return None
    for cfg in configs:
        f = fit_probe(X[tr], y[tr], kind, cfg["param"], seed, class_weight, cfg["pca"])
        a = safe_auc(y[dv], probe_scores(f, X[dv]))
        if not np.isnan(a) and (best is None or a > best["sel_auroc"]):
            best = dict(f, sel_auroc=a, dev_auroc=a, cv_auroc_mean=nan, cv_auroc_std=nan, cv_folds=[])
    return best


def fit_best_on_dev(Xtr, ytr, Xdev, ydev, kind, grid, seed=0):
    """Backwards-compatible helper (dev-split selection over a regularisation grid, no PCA, balanced weights)."""
    X, y = np.vstack([Xtr, Xdev]), np.concatenate([ytr, ydev])
    tr, dv = np.arange(len(ytr)), np.arange(len(ytr), len(y))
    return select_fit(X, y, np.arange(len(y)), tr, dv, kind, make_configs(grid), seed)


# --------------------------------------------------------------------------------------------------------------------
# per-model evaluation
# --------------------------------------------------------------------------------------------------------------------
def evaluate_model(model_key, pope: View, amber: View | None, tasks, kind, grid, stride, n_boot, seed=0, log=print,
                   opts: Options | None = None, probe_seed: int | None = None, store: dict | None = None,
                   include_baselines: bool = True):
    """Layer sweep + final probes for one model. Returns (layer_df, pope_df, amber_df, curves).

    `seed` drives the bootstrap CIs; `probe_seed` (default: seed) drives the MLP initialisation. If `store` is a dict it is
    filled with the raw test-set scores, ``store[(model, probe_seed, task, eval_set)]``, which `analyse()` turns into the
    paired tests, calibration, by-setting and AURC tables.
    """
    opts = opts or Options()
    pseed = seed if probe_seed is None else probe_seed
    cw, cv_k = opts.class_weight, opts.cv_k
    layer_rows, pope_rows, amber_rows, curves = [], [], [], {}
    pm = pope.meta
    groups_all = pm["image_id"].to_numpy()
    sig_all = unc_signals(pm)
    sigm = signal_matrix(pm)
    hs_layers = [hs for i, hs in enumerate(pope.layers) if i % stride == 0 or i == len(pope.layers) - 1]
    if amber is not None:
        hs_layers = [hs for hs in hs_layers if hs in amber.pos]
    cfgs = make_configs(grid, opts.pca)
    M_HID, M_LOG, M_COMB = f"probe ({kind})", "probe (logit-only logreg)", f"probe (hidden+logits, {kind})"
    all_rows = np.arange(len(pm))
    tname = lambda name: "probe POPE->AMBER" if name == M_HID else f"{name} POPE->AMBER"      # method label on AMBER

    for task in tasks:
        mask, y = task_arrays(pm, task)
        tr, dv, te = (split_idx(pm, mask, s) for s in ("train", "dev", "test"))
        log(f"[{model_key}] task {task}: train {len(tr)} (pos {int(y[tr].sum())}) | dev {len(dv)} (pos {int(y[dv].sum())}) | "
            f"test {len(te)} (pos {int(y[te].sum())})" + (f" | selection: {cv_k}-fold image-grouped CV on train+dev" if cv_k else ""))

        # --- baselines on POPE-test
        sig_te = {k: v[te] for k, v in sig_all.items()}
        if include_baselines:
            for name, s in sig_te.items():
                pope_rows.append(dict(model=model_key, task=task, method=f"baseline: {name}", layer=np.nan,
                                      **auc_ci(y[te], s, groups_all[te], n_boot, seed)))

        # --- amber targets (rows, labels) per subset and eval set
        targets, am, a_mask, a_y, a_sig, a_sigm = {}, None, None, None, None, None
        if amber is not None:
            am = amber.meta
            a_mask, a_y = task_arrays(am, task)
            a_sig, a_sigm = unc_signals(am), signal_matrix(am)
            for sub in AMBER_SUBSETS:
                for ev in ("all", "test"):
                    sel = a_mask & (am["setting"].to_numpy() == sub)
                    if ev == "test":
                        sel &= am["split"].to_numpy() == "test"
                    targets[(sub, ev)] = np.flatnonzero(sel)

        # --- layer sweep: choose layer + regularisation (+ PCA size) on dev / CV, never on test
        fits = {}
        for hs in hs_layers:
            Xl = pope.X(all_rows, hs)
            f = select_fit(Xl, y, groups_all, tr, dv, kind, cfgs, pseed, cw, cv_k)
            if f is None:
                continue
            f["test_auroc"] = safe_auc(y[te], probe_scores(f, Xl[te]))
            fits[hs] = f
            row = dict(model=model_key, task=task, layer=hs, depth=hs / max(pope.n_entries - 1, 1), seed=pseed,
                       sel_auroc=f["sel_auroc"], dev_auroc=f["dev_auroc"], cv_auroc_mean=f["cv_auroc_mean"],
                       cv_auroc_std=f["cv_auroc_std"], test_auroc=f["test_auroc"], param=f["param"], pca_k=f["pca_k"])
            for (sub, ev), idx in targets.items():
                if ev == "all":
                    row[f"transfer_{sub}"] = safe_auc(a_y[idx], probe_scores(f, amber.X(idx, hs))) if len(idx) else np.nan
            layer_rows.append(row)
        if not fits:
            log(f"[{model_key}] task {task}: no usable layer (a class is missing in train or dev)")
            continue
        best_layer = max(fits, key=lambda h: fits[h]["sel_auroc"])                      # dev / CV only
        bf = fits[best_layer]
        log(f"[{model_key}] task {task} seed {pseed}: best layer {best_layer} ({'CV' if cv_k else 'dev'} AUROC {bf['sel_auroc']:.3f}, "
            f"test {bf['test_auroc']:.3f}, param {bf['param']}, pca {bf['pca_k']})")
        Xb = pope.X(all_rows, best_layer)
        s_te = probe_scores(bf, Xb[te])
        gte = groups_all[te]
        pope_rows.append(dict(model=model_key, task=task, method=M_HID, layer=best_layer, param=bf["param"], pca_k=bf["pca_k"],
                              sel_auroc=bf["sel_auroc"], **auc_ci(y[te], s_te, gte, n_boot, seed)))

        # --- logit-only and hidden+logits probes (same dev/CV protocol; hidden layer = the one picked above)
        fl = fc = None
        if opts.combined:
            fl = select_fit(sigm, y, groups_all, tr, dv, "logreg", make_configs(GRIDS["logreg"]), pseed, cw, cv_k)
            fc = select_fit(np.hstack([Xb, sigm]), y, groups_all, tr, dv, kind, make_configs(grid), pseed, cw, cv_k)
            for name, f, Xf in [(M_LOG, fl, sigm), (M_COMB, fc, np.hstack([Xb, sigm]))]:
                if f is not None:
                    pope_rows.append(dict(model=model_key, task=task, method=name, layer=best_layer, param=f["param"],
                                          sel_auroc=f["sel_auroc"], **auc_ci(y[te], probe_scores(f, Xf[te]), gte, n_boot, seed)))

        # --- raw scores for the paired tests / calibration / by-setting / AURC
        if store is not None:
            scores = {f"baseline: {k}": v for k, v in sig_te.items()}
            scores[M_HID] = s_te
            if fl is not None:
                scores[M_LOG] = probe_scores(fl, sigm[te])
            if fc is not None:
                scores[M_COMB] = probe_scores(fc, np.hstack([Xb, sigm])[te])
            ent = dict(y=y[te], groups=gte, setting=pm["setting"].to_numpy()[te], scores=scores, p_model_err=1 - pm["conf"].to_numpy()[te])
            if opts.recalibrate:                                                          # Platt scaling fitted on dev / OOF only
                if cv_k:
                    idx_oof, s_oof = bf["oof"]
                    ok = ~np.isnan(s_oof)
                    pl = platt_fit(s_oof[ok], y[idx_oof][ok])
                else:
                    pl = platt_fit(probe_scores(bf, Xb[dv]), y[dv])
                if pl is not None:
                    ent["probe_platt"] = platt_apply(pl, s_te)
            store[(model_key, pseed, task, "POPE test")] = ent
        if task == "B":                                                                 # for accuracy-coverage curves
            c = dict(correct=1 - y[te], probe=s_te, **{k: v for k, v in sig_te.items()})
            if fc is not None:
                c["probe combined"] = probe_scores(fc, np.hstack([Xb, sigm])[te])
            curves[(model_key, "POPE test")] = c

        # --- AMBER: baselines, transferred probes, in-domain probe
        if amber is None:
            continue
        Xa_all = amber.X(np.arange(len(am)), best_layer)
        for (sub, ev), idx in targets.items():
            if len(idx) == 0:
                continue
            groups = am["image_id"].to_numpy()[idx]
            if include_baselines:
                for name, s in a_sig.items():
                    amber_rows.append(dict(model=model_key, task=task, subset=sub, eval_set=ev, method=f"baseline: {name}",
                                           layer=np.nan, **auc_ci(a_y[idx], s[idx], groups, n_boot, seed)))
            s_tr = probe_scores(bf, Xa_all[idx])
            amber_rows.append(dict(model=model_key, task=task, subset=sub, eval_set=ev, method=tname(M_HID),
                                   layer=best_layer, **auc_ci(a_y[idx], s_tr, groups, n_boot, seed)))
            tr_scores = {M_HID: s_tr}
            for name, f, Xf in [(M_LOG, fl, a_sigm), (M_COMB, fc, np.hstack([Xa_all, a_sigm]))]:
                if f is not None:
                    tr_scores[name] = probe_scores(f, Xf[idx])
                    amber_rows.append(dict(model=model_key, task=task, subset=sub, eval_set=ev, method=tname(name),
                                           layer=best_layer, **auc_ci(a_y[idx], tr_scores[name], groups, n_boot, seed)))
            if ev == "test":
                itr = np.flatnonzero(a_mask & (am["setting"].to_numpy() == sub) & (am["split"].to_numpy() == "train"))
                if len(itr) >= 50 and len(np.unique(a_y[itr])) == 2:
                    f_in = fit_probe(Xa_all[itr], a_y[itr], kind, bf["param"], pseed, cw, bf["pca_k"])
                    s_in = probe_scores(f_in, Xa_all[idx])
                    amber_rows.append(dict(model=model_key, task=task, subset=sub, eval_set=ev, method="probe AMBER->AMBER",
                                           layer=best_layer, **auc_ci(a_y[idx], s_in, groups, n_boot, seed)))
                if store is not None:
                    sc = {f"baseline: {k}": v[idx] for k, v in a_sig.items()}
                    sc.update({tname(k): v for k, v in tr_scores.items()})
                    store[(model_key, pseed, task, f"AMBER {sub} test")] = dict(
                        y=a_y[idx], groups=groups, setting=am["setting"].to_numpy()[idx], scores=sc,
                        p_model_err=1 - am["conf"].to_numpy()[idx])
            if task == "B" and ev == "all":
                c = dict(correct=1 - a_y[idx], probe=s_tr, **{k: v[idx] for k, v in a_sig.items()})
                if M_COMB in tr_scores:
                    c["probe combined"] = tr_scores[M_COMB]
                curves[(model_key, f"AMBER {sub}")] = c
    frames = [pd.DataFrame(r) for r in (layer_rows, pope_rows, amber_rows)]
    for d in frames[1:]:
        if len(d):
            d["seed"] = pseed
    return frames[0], frames[1], frames[2], curves


# --------------------------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------------------------
def plot_layerwise(layer_df, pope_df, task, out_dir):
    """POPE test AUROC per layer (mean over seeds). Shaded band = std of the per-fold CV AUROC when --layer-cv is used
    (the curve the layer is chosen from), otherwise the std of test AUROC across MLP seeds when there are several."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = layer_df[layer_df.task == task]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (model, g) in enumerate(d.groupby("model")):
        c = f"C{i}"
        agg = g.groupby(["layer", "depth"]).agg(test=("test_auroc", "mean"), test_sd=("test_auroc", "std"),
                                                cv=("cv_auroc_mean", "mean"), cv_sd=("cv_auroc_std", "mean")).reset_index().sort_values("depth")
        ax.plot(agg["depth"], agg["test"], marker="o", ms=3, color=c, label=f"{model} probe (test AUROC)")
        if agg["cv"].notna().any():
            ax.plot(agg["depth"], agg["cv"], color=c, ls="-.", lw=1, label=f"{model} CV AUROC (selection curve)")
            ax.fill_between(agg["depth"], agg["cv"] - agg["cv_sd"], agg["cv"] + agg["cv_sd"], color=c, alpha=0.2)
        elif agg["test_sd"].notna().any():
            ax.fill_between(agg["depth"], agg["test"] - agg["test_sd"], agg["test"] + agg["test_sd"], color=c, alpha=0.2)
        base = pope_df[(pope_df.model == model) & (pope_df.task == task) & pope_df.method.str.startswith("baseline")]
        if len(base):
            ax.axhline(base["auroc"].max(), color=c, ls="--", lw=1, alpha=0.7)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("relative depth (hidden-state entry / last entry)")
    ax.set_ylabel("POPE test AUROC")
    ax.set_title(f"Task {task}: {TASKS[task]['desc']}\n(dashed = best output-level baseline per model; layer picked on dev/CV, not test)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"layerwise_auroc_task{task}.png", dpi=150)
    plt.close(fig)


def plot_transfer(amber_df, task, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = amber_df[(amber_df.task == task) & (amber_df.eval_set == "test")] if len(amber_df) else amber_df
    if d is None or d.empty:
        return
    models = sorted(d.model.unique())
    fig, axes = plt.subplots(1, len(AMBER_SUBSETS), figsize=(4.2 * len(AMBER_SUBSETS), 3.8), sharey=True)
    for ax, sub in zip(np.atleast_1d(axes), AMBER_SUBSETS):
        g = d[d.subset == sub]
        w = 0.26
        for j, (lab, col) in enumerate([("best baseline", "#9aa5b1"), ("probe POPE->AMBER", "#d1495b"),
                                        ("probe AMBER->AMBER", "#2e86ab")]):
            vals, lo, hi = [], [], []
            for m in models:
                r = g[g.model == m]
                r = r[r.method.str.startswith("baseline")].sort_values("auroc").tail(1) if j == 0 else r[r.method == lab]
                r = r.drop_duplicates(subset=["method"], keep="first")
                if len(r) and not np.isnan(r["auroc"].iloc[0]):
                    v = r["auroc"].iloc[0]
                    vals.append(v)
                    lo.append(max(v - r["ci_lo"].iloc[0], 0) if not np.isnan(r["ci_lo"].iloc[0]) else 0)
                    hi.append(max(r["ci_hi"].iloc[0] - v, 0) if not np.isnan(r["ci_hi"].iloc[0]) else 0)
                else:
                    vals.append(np.nan); lo.append(0); hi.append(0)
            ax.bar(np.arange(len(models)) + (j - 1) * w, vals, w, yerr=[lo, hi], capsize=2, color=col, label=lab)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=8, rotation=15)
        ax.set_title(f"AMBER {sub}", fontsize=10)
        if g["auroc"].isna().all():
            ax.text(0.5, 0.5, "undefined\n(single class)", transform=ax.transAxes, ha="center")
    np.atleast_1d(axes)[0].set_ylabel("AUROC on AMBER test split")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.suptitle(f"Task {task} transfer: POPE-trained probe on AMBER (95% image-bootstrap CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"transfer_task{task}.png", dpi=150)
    plt.close(fig)


def plot_coverage(curves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not curves:
        return
    models = sorted({k[0] for k in curves})
    cols = ["POPE test"] + [f"AMBER {s}" for s in AMBER_SUBSETS]
    fig, axes = plt.subplots(len(models), len(cols), figsize=(3.6 * len(cols), 2.9 * len(models)), squeeze=False, sharey=False)
    for i, m in enumerate(models):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            d = curves.get((m, c))
            if d is None:
                ax.axis("off")
                continue
            correct = d["correct"]
            for name in ("probe", "probe combined", "token prob (raw)", "binary entropy"):
                if name not in d:
                    continue
                cov, acc = acc_coverage(d[name], correct)
                keep = np.arange(len(cov)) >= min(30, len(cov) - 1)          # drop the noisy first few answers
                cov, acc = cov[keep], acc[keep]
                ax.plot(cov, acc, label=("hidden probe" if name == "probe" else name), lw=1.6 if name.startswith("probe") else 1)
            ax.hlines(correct.mean(), 0, 1, color="gray", ls="--", label="no abstention")
            ax.set_title(f"{m} | {c}", fontsize=8)
            ax.set_xlabel("coverage", fontsize=8)
            ax.set_ylabel("accuracy", fontsize=8)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Accuracy vs coverage (abstain on the least certain answers first; Task B scores)", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "accuracy_vs_coverage.png", dpi=150)
    plt.close(fig)


def plot_reliability(reliab, out_dir):
    """Reliability diagrams on POPE test: the probe's predicted P(positive) next to the model's own answer confidence
    (as P(wrong) = 1 - confidence). One figure per task, one panel per model; point labels are bin counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in sorted({k[1] for k in reliab}):
        models = sorted({k[0] for k in reliab if k[1] == task})
        fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), squeeze=False)
        for ax, m in zip(axes[0], models):
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            for name, t in reliab[(m, task)].items():
                t = t[t.n >= 3]
                ax.plot(t.mean_p, t.frac_pos, marker="o", ms=4, label=name)
                for _, r in t.iterrows():
                    ax.annotate(str(int(r.n)), (r.mean_p, r.frac_pos), fontsize=5, xytext=(2, -8), textcoords="offset points")
            ax.set_xlabel("predicted probability of the bad outcome")
            ax.set_ylabel("observed fraction")
            ax.set_title(f"{m} | Task {task} | POPE test", fontsize=9)
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"probe_reliability_task{task}.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------
# analysis of the stored test-set scores
# --------------------------------------------------------------------------------------------------------------------
POPE_SETTINGS = ["random", "popular", "adversarial"]


def _role(name: str) -> str:
    """'logit' (4 output signals only), 'comb' (hidden + signals), 'hid' (hidden only) or 'baseline'."""
    if name.startswith("baseline"):
        return "baseline"
    return "logit" if "logit-only" in name else ("comb" if "hidden+logits" in name else "hid")


def analyse(store, primary_seed, n_boot, n_boot_paired, log=print):
    """Turn the stored test scores into the paired-test, calibration, by-setting and AURC tables.

    Paired tests are computed for every seed; the other tables use `primary_seed` only (the first seed).
    Returns (paired_df, calib_df, setting_df, aurc_df, reliab) where reliab[(model, task)][name] is a bins DataFrame.
    """
    paired, calib, by_set, aurc, reliab = [], [], [], [], {}
    for (model, seed, task, ev), d in sorted(store.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        y, g, sc = d["y"], d["groups"], d["scores"]
        names = list(sc)
        by_role = {r: [n for n in names if _role(n) == r] for r in ("baseline", "hid", "logit", "comb")}
        pairs = [(p, b, "output baseline") for p in by_role["hid"] + by_role["logit"] + by_role["comb"] for b in by_role["baseline"]]
        pairs += [(a, b, "probe") for a, b in [(by_role["hid"], by_role["logit"]), (by_role["comb"], by_role["logit"]),
                                               (by_role["comb"], by_role["hid"])] for a in a[:1] for b in b[:1]]
        for p, b, kind in pairs:
            r = paired_auc_diff(y, sc[p], sc[b], g, n_boot_paired, 0)
            paired.append(dict(model=model, seed=seed, task=task, eval_set=ev, probe=p, reference=b, reference_type=kind, **r))
        if seed != primary_seed:
            continue
        # --- by POPE setting
        if ev == "POPE test":
            for st in ["overall"] + POPE_SETTINGS:
                sel = np.ones(len(y), bool) if st == "overall" else d["setting"] == st
                for n in names:
                    by_set.append(dict(model=model, task=task, setting=st, method=n,
                                       **auc_ci(y[sel], sc[n][sel], g[sel], n_boot, 0)))
            # --- calibration of probabilities of the bad outcome
            probs = {n: sc[n] for n in names if _role(n) != "baseline"}
            probs["model confidence (P(wrong) = 1 - conf)"] = d["p_model_err"]
            if "probe_platt" in d:
                probs["probe (hidden) + Platt"] = d["probe_platt"]
            base = float(y.mean())
            for n, p in probs.items():
                calib.append(dict(model=model, task=task, method=n, ece=ece_score(p, y), brier=brier_score(p, y), n=len(y),
                                  n_pos=int(y.sum()), base_rate=base, brier_constant_base_rate=base * (1 - base)))
            reliab[(model, task)] = {n: reliability_bins(probs[n], y) for n in
                                     [n for n in probs if _role(n) == "hid" or n.startswith("model conf") or n.endswith("Platt")]}
        # --- selective prediction (Task B: abstain on the answers judged most likely wrong)
        if task == "B":
            for n in names:
                aurc.append(dict(model=model, eval_set=ev, method=n, **aurc_metrics(sc[n], 1 - y)))
    cols = lambda rows: pd.DataFrame(rows)
    return cols(paired), cols(calib), cols(by_set), cols(aurc), reliab


# --------------------------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------------------------
def seed_summary(pope_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std of POPE test AUROC across probe seeds, with the per-seed values (only probe rows)."""
    d = pope_df[pope_df.method.str.startswith("probe")]
    rows = []
    for (model, task, method), g in d.groupby(["model", "task", "method"]):
        g = g.sort_values("seed")
        rows.append(dict(model=model, task=task, method=method, n_seeds=len(g), auroc_mean=g.auroc.mean(),
                         auroc_std=g.auroc.std(ddof=1) if len(g) > 1 else np.nan,
                         per_seed_auroc=";".join(f"{s}:{a:.4f}" for s, a in zip(g.seed, g.auroc)),
                         per_seed_layer=";".join(f"{s}:{int(l)}" for s, l in zip(g.seed, g.layer))))
    return pd.DataFrame(rows)


def run(models, cache_dir, out_dir, kind="logreg", tasks=("B", "A"), stride=1, n_boot=500, restrict_common=True, seed=0,
        log=print, opts: Options | None = None):
    opts = opts or Options()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # logistic regression is deterministic, so extra seeds would just repeat it; only the MLP uses several seeds
    seeds = tuple(opts.seeds) if kind == "mlp" else (opts.seeds[0] if opts.seeds else 0,)
    if kind == "logreg" and len(opts.seeds) > 1:
        log(f"logreg probe is deterministic -> using seed {seeds[0]} only (--seeds applies to the MLP probe)")
    popes, ambers = {}, {}
    for m in models:
        popes[m] = load_cache("pope", m, cache_dir)
        try:
            ambers[m] = load_cache("amber", m, cache_dir)
        except FileNotFoundError:
            log(f"[{m}] no AMBER cache found -> transfer skipped for this model")
    common_p = common_a = None
    if restrict_common and len(popes) > 1:
        common_p = set.intersection(*(set(c["meta"]["row_idx"]) for c in popes.values()))
        log(f"restricting POPE to the {len(common_p)} questions common to all models (train/dev/test rows included)")
    if restrict_common and len(ambers) > 1:
        common_a = set.intersection(*(set(c["meta"]["row_idx"]) for c in ambers.values()))
        log(f"restricting AMBER to the {len(common_a)} queries common to all models")

    L, P, A, C, store, used = [], [], [], {}, {}, {}
    for m in models:
        pv = View(popes[m], common_p)
        av = View(ambers[m], common_a) if m in ambers else None
        used[m] = dict(pope_rows=len(pv.meta), amber_rows=None if av is None else len(av.meta))
        for i, sd in enumerate(seeds):
            l, p, a, c = evaluate_model(m, pv, av, tasks, kind, GRIDS[kind], stride, n_boot, seed, log, opts=opts,
                                        probe_seed=sd, store=store, include_baselines=(i == 0))
            L.append(l); P.append(p); A.append(a)
            if i == 0:
                C.update(c)
    layer_df, pope_df = pd.concat(L, ignore_index=True), pd.concat(P, ignore_index=True)
    amber_df = pd.concat([a for a in A if len(a)], ignore_index=True) if any(len(a) for a in A) else pd.DataFrame()
    layer_df.to_csv(out_dir / "layerwise_pope.csv", index=False)
    pope_df.to_csv(out_dir / "pope_test_summary.csv", index=False)
    seed_summary(pope_df).to_csv(out_dir / "pope_test_seed_summary.csv", index=False)
    if len(amber_df):
        amber_df.to_csv(out_dir / "transfer_amber.csv", index=False)
    paired_df, calib_df, set_df, aurc_df, reliab = analyse(store, seeds[0], n_boot, opts.n_boot_paired, log)
    for name, df in [("paired_tests", paired_df), ("calibration", calib_df), ("pope_by_setting", set_df), ("aurc", aurc_df)]:
        if len(df):
            df.to_csv(out_dir / f"{name}.csv", index=False)
    for t in tasks:
        plot_layerwise(layer_df, pope_df, t, out_dir)
        if len(amber_df):
            plot_transfer(amber_df, t, out_dir)
    plot_coverage(C, out_dir)
    plot_reliability(reliab, out_dir)
    files = {}
    for ds in ("pope", "amber"):
        for m in models:
            try:
                f, mp = find_cache(ds, m, cache_dir)
                files[f"{ds}/{m}"] = dict(features=f.name, features_bytes=f.stat().st_size, metadata=mp.name)
            except FileNotFoundError:
                pass
    import sklearn
    (out_dir / "run_config.json").write_text(json.dumps(dict(
        models=list(models), cache_dir=str(cache_dir), probe=kind, tasks=list(tasks), layer_stride=stride, n_boot=n_boot,
        restrict_common=restrict_common, bootstrap_seed=seed, probe_seeds=list(seeds), options=asdict(opts), grid=GRIDS[kind],
        rows_used=used, cache_files=files, versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        date=str(pd.Timestamp.now().date())), indent=2, default=str))
    return layer_df, pope_df, amber_df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="model keys, e.g. qwen2vl-2b smolvlm-500m")
    p.add_argument("--cache-dir", default="results/cached_features")
    p.add_argument("--out-dir", default="results/final/run")
    p.add_argument("--probe", choices=list(GRIDS), default="logreg")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--layer-stride", type=int, default=1, help="evaluate every k-th layer (plus the last) to save time")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--no-restrict-common", action="store_true", help="use each model's full data instead of the shared questions")
    p.add_argument("--class-weight", action=argparse.BooleanOptionalAction, default=True,
                   help="balanced class weights for the logreg probe (default: on)")
    p.add_argument("--pca", type=int, nargs="*", default=[], help="PCA sizes to choose from on dev, e.g. 64 128 256 (default: no PCA)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="MLP probe seeds (logreg is deterministic: first seed only)")
    p.add_argument("--layer-cv", type=int, default=0, metavar="K", help="select layer/regularisation by image-grouped K-fold CV over train+dev")
    p.add_argument("--recalibrate", action="store_true", help="also report Platt-scaled probe probabilities")
    p.add_argument("--n-boot-paired", type=int, default=1000)
    a = p.parse_args(argv)
    opts = Options(class_weight=a.class_weight, pca=tuple(a.pca), cv_k=a.layer_cv, seeds=tuple(a.seeds),
                   recalibrate=a.recalibrate, n_boot_paired=a.n_boot_paired)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text("python src/probe_transfer_eval.py " + " ".join(argv if argv is not None else sys.argv[1:]) + "\n")
    layer_df, pope_df, amber_df = run(a.models, a.cache_dir, a.out_dir, a.probe, a.tasks, a.layer_stride, a.n_boot,
                                      not a.no_restrict_common, 0, print, opts)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 300)
    show = [c for c in ["model", "task", "seed", "method", "layer", "auroc", "ci_lo", "ci_hi", "n", "n_pos"] if c in pope_df]
    print("\n=== POPE test (Task B = primary; probe layer + regularisation chosen on dev/CV, never on test) ===")
    print(pope_df[show].round(3).to_string(index=False))
    if len(amber_df):
        print("\n=== AMBER (test split rows) ===")
        print(amber_df[amber_df.eval_set == "test"].round(3).to_string(index=False))
        undefined = amber_df[(amber_df.eval_set == "all") & amber_df.method.eq("probe POPE->AMBER") & amber_df.auroc.isna()]
        for _, r in undefined.iterrows():
            print(f"\nNOTE: [{r.model}] task {r.task} on AMBER-{r.subset} is undefined (only one class present in the selection).")
    print(f"\nTables and figures written to {a.out_dir}")


if __name__ == "__main__":
    main()
