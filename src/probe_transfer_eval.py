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

    python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m qwen25vl-3b --out-dir results/extra_models
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from cache_io import load_cache

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TASKS = {"A": dict(label="hallucination", mask=lambda m: (m["pred_yes"] == 1).to_numpy(),
                   desc="hallucinated-yes vs correct-yes (answered-yes only)"),
         "B": dict(label="error", mask=lambda m: np.ones(len(m), bool), desc="wrong vs right (all questions)")}
AMBER_SUBSETS = ["existence", "attribute", "relation"]
GRIDS = {"logreg": [1e-3, 1e-2, 1e-1], "mlp": [1e-2, 1.0]}       # C for logreg, L2 alpha for mlp (picked on dev only)


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
def make_probe(kind: str, param: float, seed: int = 0):
    if kind == "logreg":
        return LogisticRegression(C=param, max_iter=300)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), alpha=param, early_stopping=True, max_iter=200, random_state=seed)
    raise ValueError(kind)


def fit_probe(Xtr, ytr, kind, param, seed=0):
    sc = StandardScaler().fit(Xtr)                                    # statistics from POPE-train only
    clf = make_probe(kind, param, seed).fit(sc.transform(Xtr), ytr)
    return dict(scaler=sc, clf=clf, param=param)


def probe_scores(fit, X) -> np.ndarray:
    return fit["clf"].predict_proba(fit["scaler"].transform(X))[:, 1]


def fit_best_on_dev(Xtr, ytr, Xdev, ydev, kind, grid, seed=0):
    """Fit one probe per grid value, keep the one with the best dev AUROC. None if a class is missing."""
    if len(np.unique(ytr)) < 2 or len(np.unique(ydev)) < 2:
        return None
    best = None
    for p in grid:
        f = fit_probe(Xtr, ytr, kind, p, seed)
        a = safe_auc(ydev, probe_scores(f, Xdev))
        if best is None or a > best["dev_auroc"]:
            best = dict(f, dev_auroc=a)
    return best


# --------------------------------------------------------------------------------------------------------------------
# per-model evaluation
# --------------------------------------------------------------------------------------------------------------------
def evaluate_model(model_key, pope: View, amber: View | None, tasks, kind, grid, stride, n_boot, seed=0, log=print):
    layer_rows, pope_rows, amber_rows, curves = [], [], [], {}
    pm = pope.meta
    hs_layers = [hs for i, hs in enumerate(pope.layers) if i % stride == 0 or i == len(pope.layers) - 1]
    if amber is not None:
        hs_layers = [hs for hs in hs_layers if hs in amber.pos]

    for task in tasks:
        mask, y = task_arrays(pm, task)
        tr, dv, te = (split_idx(pm, mask, s) for s in ("train", "dev", "test"))
        log(f"[{model_key}] task {task}: train {len(tr)} (pos {int(y[tr].sum())}) | dev {len(dv)} | test {len(te)} (pos {int(y[te].sum())})")

        # --- baselines on POPE-test
        sig_te = {k: v[te] for k, v in unc_signals(pm).items()}
        for name, s in sig_te.items():
            pope_rows.append(dict(model=model_key, task=task, method=f"baseline: {name}", layer=np.nan,
                                  **auc_ci(y[te], s, pm["image_id"].to_numpy()[te], n_boot, seed)))

        # --- amber targets (rows, labels) per subset and eval set
        targets = {}
        if amber is not None:
            am = amber.meta
            a_mask, a_y = task_arrays(am, task)
            for sub in AMBER_SUBSETS:
                for ev in ("all", "test"):
                    sel = a_mask & (am["setting"].to_numpy() == sub)
                    if ev == "test":
                        sel &= am["split"].to_numpy() == "test"
                    targets[(sub, ev)] = np.flatnonzero(sel)

        # --- layer sweep: choose layer + regularisation on dev, never on test
        fits, best_layer = {}, None
        for hs in hs_layers:
            f = fit_best_on_dev(pope.X(tr, hs), y[tr], pope.X(dv, hs), y[dv], kind, grid, seed)
            if f is None:
                continue
            f["test_auroc"] = safe_auc(y[te], probe_scores(f, pope.X(te, hs)))
            fits[hs] = f
            row = dict(model=model_key, task=task, layer=hs, depth=hs / max(pope.n_entries - 1, 1),
                       dev_auroc=f["dev_auroc"], test_auroc=f["test_auroc"], param=f["param"])
            for (sub, ev), idx in targets.items():
                if ev == "all":
                    row[f"transfer_{sub}"] = safe_auc(a_y[idx], probe_scores(f, amber.X(idx, hs))) if len(idx) else np.nan
            layer_rows.append(row)
        if not fits:
            log(f"[{model_key}] task {task}: no usable layer (a class is missing in train or dev)")
            continue
        best_layer = max(fits, key=lambda h: fits[h]["dev_auroc"])                     # dev only
        bf = fits[best_layer]
        log(f"[{model_key}] task {task}: best dev layer {best_layer} (dev AUROC {bf['dev_auroc']:.3f}, test {bf['test_auroc']:.3f}, "
            f"param {bf['param']})")
        s_te = probe_scores(bf, pope.X(te, best_layer))
        pope_rows.append(dict(model=model_key, task=task, method=f"probe ({kind})", layer=best_layer,
                              **auc_ci(y[te], s_te, pm["image_id"].to_numpy()[te], n_boot, seed)))
        if task == "B":                                                                 # for accuracy-coverage curves
            curves[(model_key, "POPE test")] = dict(correct=1 - y[te], probe=s_te, **{k: v for k, v in sig_te.items()})

        # --- AMBER: baselines, transferred probe, in-domain probe
        if amber is None:
            continue
        for (sub, ev), idx in targets.items():
            if len(idx) == 0:
                continue
            groups = am["image_id"].to_numpy()[idx]
            for name, s in unc_signals(am).items():
                amber_rows.append(dict(model=model_key, task=task, subset=sub, eval_set=ev, method=f"baseline: {name}",
                                       layer=np.nan, **auc_ci(a_y[idx], s[idx], groups, n_boot, seed)))
            s_tr = probe_scores(bf, amber.X(idx, best_layer))
            amber_rows.append(dict(model=model_key, task=task, subset=sub, eval_set=ev, method="probe POPE->AMBER",
                                   layer=best_layer, **auc_ci(a_y[idx], s_tr, groups, n_boot, seed)))
            if ev == "test":
                itr = np.flatnonzero(a_mask & (am["setting"].to_numpy() == sub) & (am["split"].to_numpy() == "train"))
                if len(itr) >= 50 and len(np.unique(a_y[itr])) == 2:
                    f_in = fit_probe(amber.X(itr, best_layer), a_y[itr], kind, bf["param"], seed)
                    s_in = probe_scores(f_in, amber.X(idx, best_layer))
                    amber_rows.append(dict(model=model_key, task=task, subset=sub, eval_set=ev, method="probe AMBER->AMBER",
                                           layer=best_layer, **auc_ci(a_y[idx], s_in, groups, n_boot, seed)))
            if task == "B" and ev == "all":
                curves[(model_key, f"AMBER {sub}")] = dict(correct=1 - a_y[idx], probe=s_tr,
                                                           **{k: v[idx] for k, v in unc_signals(am).items()})
    return pd.DataFrame(layer_rows), pd.DataFrame(pope_rows), pd.DataFrame(amber_rows), curves


# --------------------------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------------------------
def plot_layerwise(layer_df, pope_df, task, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = layer_df[layer_df.task == task]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (model, g) in enumerate(d.groupby("model")):
        c = f"C{i}"
        g = g.sort_values("depth")
        ax.plot(g["depth"], g["test_auroc"], marker="o", ms=3, color=c, label=f"{model} probe")
        base = pope_df[(pope_df.model == model) & (pope_df.task == task) & pope_df.method.str.startswith("baseline")]
        if len(base):
            ax.axhline(base["auroc"].max(), color=c, ls="--", lw=1, alpha=0.7)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("relative depth (hidden-state entry / last entry)")
    ax.set_ylabel("POPE test AUROC")
    ax.set_title(f"Task {task}: {TASKS[task]['desc']}\n(dashed = best output-level baseline per model; layer picked on dev)", fontsize=9)
    ax.legend(fontsize=8)
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
            for name in ("probe", "token prob (raw)", "binary entropy"):
                cov, acc = acc_coverage(d[name], correct)
                keep = np.arange(len(cov)) >= min(30, len(cov) - 1)          # drop the noisy first few answers
                cov, acc = cov[keep], acc[keep]
                ax.plot(cov, acc, label=("probe" if name == "probe" else name), lw=1.6 if name == "probe" else 1)
            ax.hlines(correct.mean(), 0, 1, color="gray", ls="--", label="no abstention")
            ax.set_title(f"{m} | {c}", fontsize=8)
            ax.set_xlabel("coverage", fontsize=8)
            ax.set_ylabel("accuracy", fontsize=8)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Accuracy vs coverage (abstain on the least certain answers first; Task B scores)", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "accuracy_vs_coverage.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------------------------
def run(models, cache_dir, out_dir, kind="logreg", tasks=("A", "B"), stride=1, n_boot=500, restrict_common=True, seed=0,
        log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    L, P, A, C = [], [], [], {}
    for m in models:
        pv = View(popes[m], common_p)
        av = View(ambers[m], common_a) if m in ambers else None
        l, p, a, c = evaluate_model(m, pv, av, tasks, kind, GRIDS[kind], stride, n_boot, seed, log)
        L.append(l); P.append(p); A.append(a); C.update(c)
    layer_df, pope_df = pd.concat(L, ignore_index=True), pd.concat(P, ignore_index=True)
    amber_df = pd.concat([a for a in A if len(a)], ignore_index=True) if any(len(a) for a in A) else pd.DataFrame()
    layer_df.to_csv(out_dir / "layerwise_pope.csv", index=False)
    pope_df.to_csv(out_dir / "pope_test_summary.csv", index=False)
    if len(amber_df):
        amber_df.to_csv(out_dir / "transfer_amber.csv", index=False)
    for t in tasks:
        plot_layerwise(layer_df, pope_df, t, out_dir)
        if len(amber_df):
            plot_transfer(amber_df, t, out_dir)
    plot_coverage(C, out_dir)
    return layer_df, pope_df, amber_df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="model keys, e.g. qwen2vl-2b smolvlm-500m")
    p.add_argument("--cache-dir", default="results/cached_features")
    p.add_argument("--out-dir", default="results/extra_models")
    p.add_argument("--probe", choices=list(GRIDS), default="logreg")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=["A", "B"])
    p.add_argument("--layer-stride", type=int, default=1, help="evaluate every k-th layer (plus the last) to save time")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--no-restrict-common", action="store_true", help="use each model's full data instead of the shared questions")
    a = p.parse_args(argv)
    layer_df, pope_df, amber_df = run(a.models, a.cache_dir, a.out_dir, a.probe, a.tasks, a.layer_stride, a.n_boot,
                                      not a.no_restrict_common)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 200)
    print("\n=== POPE test (probe layer + regularisation chosen on dev) ===")
    print(pope_df.round(3).to_string(index=False))
    if len(amber_df):
        print("\n=== AMBER (test split rows) ===")
        print(amber_df[amber_df.eval_set == "test"].round(3).to_string(index=False))
        undefined = amber_df[(amber_df.eval_set == "all") & amber_df.method.eq("probe POPE->AMBER") & amber_df.auroc.isna()]
        for _, r in undefined.iterrows():
            print(f"\nNOTE: [{r.model}] task {r.task} on AMBER-{r.subset} is undefined (only one class present in the selection).")
    print(f"\nTables and figures written to {a.out_dir}")


if __name__ == "__main__":
    main()
