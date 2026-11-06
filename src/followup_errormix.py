#!/usr/bin/env python
"""Follow-up experiment: is the POPE -> AMBER transfer failure caused by the error mix?  (hypothesis H5, Task B only; CPU only)

PREDICTION (written before any result is computed; also printed first in run.log and stored in run_config.json)
    If H5 is right, removing the answer-direction shortcut should lift the AMBER existence and attribute Task B AUROC clearly
    above 0.5 (confidence interval entirely above 0.5), ideally close to the output baselines, at a modest cost on POPE test.
    If the variants still score at or below about 0.5 on those subsets, H5 is not supported.

Background. 85% of Qwen2-VL-2B's POPE errors are missed objects (answered "no", object present) but on AMBER the errors are mostly
hallucinated "yes" (58% of existence, 71% of attribute, 18% of relation). A Task B probe trained on POPE may mostly have learned
"the model said no, so it is probably wrong", which points the wrong way on AMBER.

Experiments (all Task B = wrong vs right answer; logistic-regression probe exactly as in the main run: last prompt token, scaler on
train, layer and C chosen on POPE DEV ONLY, test used for reporting only):
  E0  sign diagnostic (no training): the original POPE probe split by the model's answer, the flipped score 1 - s, and the trivial
      score "model answered no".
  E1a error-type re-weighting, balanced (headline, no target information): hallucinated-yes and missed errors weigh equally in the
      positive class. Layer / C re-selected on POPE dev with dev rows weighted the same way.
  E1b the same, but the error-type proportions are those of each AMBER subset's TRAIN split. USES TARGET-DOMAIN STATISTICS (labelled
      as such, never the headline); one probe per AMBER subset. AMBER dev/test labels are never used for any choice.
  E2  answer-conditioned probes: one probe on rows answered "yes", one on rows answered "no" (each selected on the matching POPE dev
      rows, each Platt-calibrated on those dev rows), routed by the model's answer at test time.

    python src/followup_errormix.py --models qwen2vl-2b smolvlm-500m --out-dir results/final/followup_errormix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import probe_transfer_eval as pe
from cache_io import find_cache, load_cache
from probe_stats import paired_auc_diff, platt_apply, platt_fit

PREDICTION = ("If H5 is right, removing the answer-direction shortcut should lift AMBER existence and attribute Task B AUROC clearly above 0.5 "
              "(95% CI entirely above 0.5), ideally close to the output baselines, at a modest cost on POPE test. If the variants still score "
              "at or below about 0.5 on those subsets, H5 is not supported.")
SUBSETS = pe.AMBER_SUBSETS
LAYER_TABLE_ENTRIES = [8, 12, 16, 20, 24, 28]


class Tee:
    """print() to the console and append the same line to run.log."""

    def __init__(self, path: Path):
        self.f = open(path, "w")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


# --------------------------------------------------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_followup_errormix.py)
# --------------------------------------------------------------------------------------------------------------------
def error_type_weights(y, pred_yes, halluc_share: float) -> np.ndarray:
    """Per-row weights that fix the error mix of the positive class (E1).

    y: 1 = wrong answer. pred_yes: 1 = the model answered "yes". A wrong "yes" is a hallucinated yes, a wrong "no" a missed object.
    Correct answers keep weight 1. The weights of the wrong answers are set so that
      * the total positive weight equals the total negative weight (this replaces sklearn's class_weight="balanced"), and
      * hallucinated-yes errors hold `halluc_share` of the positive weight and missed objects the remaining 1 - halluc_share.
    With halluc_share equal to the natural share this reproduces class_weight="balanced" exactly (up to nothing: same weights).
    A type with share 0 gets weight 0; a type with share > 0 but no rows raises ValueError.
    """
    y, py = np.asarray(y).astype(int), np.asarray(pred_yes).astype(int)
    if not 0.0 <= halluc_share <= 1.0:
        raise ValueError("halluc_share must be in [0, 1]")
    neg, hall, miss = (y == 0), (y == 1) & (py == 1), (y == 1) & (py == 0)
    w = np.ones(len(y), dtype=float)
    for mask, share, name in [(hall, halluc_share, "hallucinated-yes"), (miss, 1.0 - halluc_share, "missed-object")]:
        if share > 0 and mask.sum() == 0:
            raise ValueError(f"no {name} errors in these rows but the target share is {share:.3f}")
        w[mask] = share * neg.sum() / mask.sum() if mask.sum() else 0.0
    return w


def amber_train_halluc_share(meta: pd.DataFrame, subset: str) -> float:
    """Share of hallucinated-yes among the errors of one AMBER subset, computed on the TRAIN split only (never dev/test labels)."""
    d = meta[(meta["setting"] == subset) & (meta["split"] == "train")]
    err = d[d["error"] == 1]
    if err.empty:
        raise ValueError(f"no errors in the AMBER-{subset} train split")
    return float((err["pred_yes"] == 1).mean())              # an error with answer yes means the object is absent: hallucinated yes


def route_scores(pred_yes, s_yes, s_no) -> np.ndarray:
    """E2: each row takes the score of the probe that matches the model's answer (s_yes if answered yes, else s_no)."""
    return np.where(np.asarray(pred_yes).astype(int) == 1, np.asarray(s_yes, dtype=float), np.asarray(s_no, dtype=float))


def answer_split_auc(y, score, pred_yes, groups, n_boot: int, seed: int = 0) -> list[dict]:
    """E0: AUROC of one score on all rows and separately on the rows answered yes / answered no (with image-bootstrap CIs)."""
    y, score, py, groups = np.asarray(y), np.asarray(score, dtype=float), np.asarray(pred_yes).astype(int), np.asarray(groups)
    out = []
    for name, sel in [("all rows", np.ones(len(y), bool)), ("answered yes", py == 1), ("answered no", py == 0)]:
        out.append(dict(rows=name, **pe.auc_ci(y[sel], score[sel], groups[sel], n_boot, seed)))
    return out


# --------------------------------------------------------------------------------------------------------------------
# evaluation helper: AUROC + paired differences against the best baseline and the original probe
# --------------------------------------------------------------------------------------------------------------------
def eval_rows(model, experiment, variant, eval_set, y, score, groups, base, orig, n_boot, n_boot_paired, seed, extra=None):
    """One output row: AUROC [CI], n, n_pos and the paired image-bootstrap differences. `base`=(name, scores) is the best
    output-level baseline on these rows, `orig`=scores of the original POPE probe on the same rows (None = not applicable)."""
    row = dict(model=model, experiment=experiment, variant=variant, eval_set=eval_set, **(extra or {}))
    row.update(pe.auc_ci(y, score, groups, n_boot, seed))
    for tag, ref in [("baseline", base[1] if base else None), ("orig", orig)]:
        if ref is None:
            continue
        r = paired_auc_diff(y, score, ref, groups, n_boot_paired, seed)
        row.update({f"diff_vs_{tag}": r["diff"], f"diff_vs_{tag}_lo": r["ci_lo"], f"diff_vs_{tag}_hi": r["ci_hi"], f"diff_vs_{tag}_p": r["p_value"]})
    if base:
        row["best_baseline"] = base[0]
        row["baseline_auroc"] = pe.safe_auc(y, base[1])
    return row


def best_baseline(meta_rows: pd.DataFrame, y) -> tuple[str, np.ndarray]:
    """The output-level baseline (token prob / margin / entropies) with the highest AUROC on these rows (as in transfer_amber.csv)."""
    sig = pe.unc_signals(meta_rows)
    name = max(sig, key=lambda k: (np.nan_to_num(pe.safe_auc(y, sig[k]), nan=-1.0)))
    return f"baseline: {name}", sig[name]


# --------------------------------------------------------------------------------------------------------------------
# one model
# --------------------------------------------------------------------------------------------------------------------
def run_model(model_key, cache_dir, out_dir, seed, n_boot, n_boot_paired, log, reference_dirs=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    pope = pe.View(load_cache("pope", model_key, cache_dir))
    amber = pe.View(load_cache("amber", model_key, cache_dir))
    pm, am = pope.meta, amber.meta
    n_p, n_a = len(pm), len(am)
    all_p, all_a = np.arange(n_p), np.arange(n_a)
    _, y = pe.task_arrays(pm, "B")                                    # Task B: error over all questions
    _, a_y = pe.task_arrays(am, "B")
    tr, dv, te = (pe.split_idx(pm, np.ones(n_p, bool), s) for s in ("train", "dev", "test"))
    pyes = pm["pred_yes"].to_numpy()
    a_pyes = am["pred_yes"].to_numpy()
    groups = pm["image_id"].to_numpy()
    a_groups = am["image_id"].to_numpy()
    cfgs = pe.make_configs(pe.GRIDS["logreg"])
    a_te = {s: np.flatnonzero((am["setting"].to_numpy() == s) & (am["split"].to_numpy() == "test")) for s in SUBSETS}
    a_tr = {s: np.flatnonzero((am["setting"].to_numpy() == s) & (am["split"].to_numpy() == "train")) for s in SUBSETS}
    layers = [hs for hs in pope.layers if hs in amber.pos]
    log(f"[{model_key}] POPE rows {n_p} (train {len(tr)} / dev {len(dv)} / test {len(te)}, errors in test {int(y[te].sum())}); "
        f"AMBER rows {n_a}; layers {len(layers)}")

    # ---- weights for E1 (train and dev rows weighted the same way, dev used for selection)
    def weights_for(share):
        w = np.zeros(n_p)
        w[tr] = error_type_weights(y[tr], pyes[tr], share)
        w[dv] = error_type_weights(y[dv], pyes[dv], share)
        return w
    w_e1a = weights_for(0.5)
    shares = {s: amber_train_halluc_share(am, s) for s in SUBSETS}     # AMBER TRAIN split only (E1b, target-domain statistics)
    w_e1b = {s: weights_for(shares[s]) for s in SUBSETS}
    pope_share = float(((y[tr] == 1) & (pyes[tr] == 1)).sum() / max(int(y[tr].sum()), 1))      # natural hallucinated-yes share of POPE-train errors
    log(f"[{model_key}] hallucinated-yes share of errors: POPE train (natural) {pope_share:.3f}; E1a uses 0.500; E1b uses (AMBER train split, "
        "target-domain statistics) " + ", ".join(f"{s}={shares[s]:.3f}" for s in SUBSETS))
    tr_y, dv_y = tr[pyes[tr] == 1], dv[pyes[dv] == 1]                 # E2 sub-populations (answered yes / answered no)
    tr_n, dv_n = tr[pyes[tr] == 0], dv[pyes[dv] == 0]
    log(f"[{model_key}] E2 rows: answered-yes train {len(tr_y)} (errors {int(y[tr_y].sum())}) dev {len(dv_y)} (errors {int(y[dv_y].sum())}) | "
        f"answered-no train {len(tr_n)} (errors {int(y[tr_n].sum())}) dev {len(dv_n)} (errors {int(y[dv_n].sum())})")

    # ---- layer sweep: every variant is fitted at every layer; selection is by (weighted) POPE dev AUROC only
    fits = {k: {} for k in ["orig", "e1a", "e1b_existence", "e1b_attribute", "e1b_relation", "e2_yes", "e2_no"]}
    layer_rows = []
    for hs in layers:
        Xl = pope.X(all_p, hs)
        Xa = amber.X(all_a, hs)
        cur = {
            "orig": pe.select_fit(Xl, y, groups, tr, dv, "logreg", cfgs, seed, True),
            "e1a": pe.select_fit(Xl, y, groups, tr, dv, "logreg", cfgs, seed, False, 0, w_e1a),
            "e2_yes": pe.select_fit(Xl, y, groups, tr_y, dv_y, "logreg", cfgs, seed, True),
            "e2_no": pe.select_fit(Xl, y, groups, tr_n, dv_n, "logreg", cfgs, seed, True),
        }
        for s in SUBSETS:
            cur[f"e1b_{s}"] = pe.select_fit(Xl, y, groups, tr, dv, "logreg", cfgs, seed, False, 0, w_e1b[s])
        # Platt scaling for the two E2 probes, fitted on their own POPE dev rows (so their probabilities are comparable)
        for k, dvk in (("e2_yes", dv_y), ("e2_no", dv_n)):
            if cur[k] is not None:
                cur[k]["platt"] = platt_fit(pe.probe_scores(cur[k], Xl[dvk]), y[dvk])
        for k, f in cur.items():
            if f is not None:
                f["layer"] = hs
                fits[k][hs] = f
        # descriptive per-layer table (NOT used for selection): POPE test and AMBER test AUROC at this layer
        for name, f_list in [("orig", ["orig"]), ("e1a", ["e1a"]), ("e2", ["e2_yes", "e2_no"])] + [(f"e1b_{s}", [f"e1b_{s}"]) for s in SUBSETS]:
            if any(cur.get(k) is None for k in f_list):
                continue
            row = dict(model=model_key, variant=name, layer=hs)
            if name == "e2":
                if cur["e2_yes"]["platt"] is None or cur["e2_no"]["platt"] is None:
                    continue
                def sc(Xr, pr):
                    return route_scores(pr, platt_apply(cur["e2_yes"]["platt"], pe.probe_scores(cur["e2_yes"], Xr)),
                                        platt_apply(cur["e2_no"]["platt"], pe.probe_scores(cur["e2_no"], Xr)))
            else:
                def sc(Xr, pr, f=cur[f_list[0]]):
                    return pe.probe_scores(f, Xr)
            row["pope_test_auroc"] = pe.safe_auc(y[te], sc(Xl[te], pyes[te]))
            for s in SUBSETS:
                if name.startswith("e1b_") and name != f"e1b_{s}":
                    continue
                row[f"amber_{s}_test_auroc"] = pe.safe_auc(a_y[a_te[s]], sc(Xa[a_te[s]], a_pyes[a_te[s]]))
            layer_rows.append(row)
        log(f"[{model_key}] layer {hs:2d}/{layers[-1]} done")

    def best(k):
        d = {h: f for h, f in fits[k].items() if not np.isnan(f["sel_auroc"])}
        if not d:
            raise RuntimeError(f"no usable layer for variant {k} (a class is missing in train or dev)")
        hs = max(d, key=lambda h: d[h]["sel_auroc"])                    # weighted POPE-dev AUROC only
        return hs, d[hs]
    chosen = {k: best(k) for k in fits}
    for k in ("e2_yes", "e2_no"):
        if chosen[k][1]["platt"] is None:
            raise RuntimeError(f"{k}: a single class in the POPE dev rows, cannot fit the Platt scaling needed to combine the E2 probes")
    for k, (hs, f) in chosen.items():
        log(f"[{model_key}] selected {k}: layer {hs}, C={f['param']}, POPE-dev AUROC (weighted for E1) {f['sel_auroc']:.3f}")

    # ---- final scores on POPE test and on each AMBER test split
    hs0, f0 = chosen["orig"]
    Xp = {hs: pope.X(all_p, hs) for hs in {v[0] for v in chosen.values()}}
    Xq = {hs: amber.X(all_a, hs) for hs in {v[0] for v in chosen.values()}}

    def score(k, which, rows):
        hs, f = chosen[k]
        return pe.probe_scores(f, (Xp if which == "pope" else Xq)[hs][rows])

    def e2_score(which, rows, pr):
        (hy, fy), (hn, fn) = chosen["e2_yes"], chosen["e2_no"]
        X = Xp if which == "pope" else Xq
        return route_scores(pr, platt_apply(fy["platt"], pe.probe_scores(fy, X[hy][rows])),
                            platt_apply(fn["platt"], pe.probe_scores(fn, X[hn][rows])))

    eval_sets = [("POPE test", "pope", te, y, groups, pm, pyes)] + [
        (f"AMBER {s} test", "amber", a_te[s], a_y, a_groups, am, a_pyes) for s in SUBSETS]
    rows, e0_rows = [], []
    reproduce = {}
    for name, which, idx, yy, gg, meta_rows, py_all in eval_sets:
        ylab, g, py = yy[idx], gg[idx], py_all[idx]
        base = best_baseline(meta_rows.iloc[idx], ylab)
        s_orig = score("orig", which, idx)
        sub = name.split()[1] if which == "amber" else None
        common = dict(n_boot=n_boot, n_boot_paired=n_boot_paired, seed=seed)
        rows.append(eval_rows(model_key, "baseline", base[0], name, ylab, base[1], g, None, s_orig, **common))
        rows.append(eval_rows(model_key, "orig", "original POPE probe", name, ylab, s_orig, g, base, None, **common,
                              extra=dict(layer=hs0, C=f0["param"], uses_target_domain_stats=False)))
        rows.append(eval_rows(model_key, "E1a", "E1a balanced error types", name, ylab, score("e1a", which, idx), g, base, s_orig, **common,
                              extra=dict(layer=chosen["e1a"][0], C=chosen["e1a"][1]["param"], uses_target_domain_stats=False)))
        k = f"e1b_{sub}" if sub else None
        for ks in ([k] if k else [f"e1b_{s}" for s in SUBSETS]):          # on POPE test report the cost of each per-subset probe
            rows.append(eval_rows(model_key, "E1b", f"E1b AMBER-matched ({ks[4:]}) [target-domain stats]", name, ylab, score(ks, which, idx), g,
                                  base, s_orig, **common, extra=dict(layer=chosen[ks][0], C=chosen[ks][1]["param"], uses_target_domain_stats=True)))
        s_e2 = e2_score(which, idx, py)
        rows.append(eval_rows(model_key, "E2", "E2 answer-conditioned (routed, Platt)", name, ylab, s_e2, g, base, s_orig, **common,
                              extra=dict(layer=f"yes:{chosen['e2_yes'][0]}/no:{chosen['e2_no'][0]}", uses_target_domain_stats=False)))
        for lab, kk, sel in [("answered yes", "e2_yes", py == 1), ("answered no", "e2_no", py == 0)]:     # each sub-probe on its own rows
            if sel.sum() == 0:
                continue
            sy = ylab[sel]
            b_sub = best_baseline(meta_rows.iloc[idx][sel], sy)
            rows.append(eval_rows(model_key, "E2 sub-probe", f"E2 {kk[3:]}-probe on rows {lab}", name, sy, score(kk, which, idx)[sel], g[sel], b_sub,
                                  s_orig[sel], **common, extra=dict(layer=chosen[kk][0], C=chosen[kk][1]["param"], uses_target_domain_stats=False)))
        # E0: sign diagnostic of the original probe, its flip, and the trivial "answered no" score
        for lab, sc_ in [("original POPE probe", s_orig), ("1 - original probe", 1 - s_orig), ("trivial: model answered no", (py == 0).astype(float))]:
            for r in answer_split_auc(ylab, sc_, py, g, n_boot, seed):
                e0_rows.append(dict(model=model_key, eval_set=name, score=lab, **r))
        # AMBER-trained ceiling (same layer and C as the original probe), only for the figure
        if sub:
            itr = a_tr[sub]
            if len(itr) >= 50 and len(np.unique(a_y[itr])) == 2:
                f_in = pe.fit_probe(Xq[hs0][itr], a_y[itr], "logreg", f0["param"], seed, True, None)
                rows.append(eval_rows(model_key, "ceiling", "AMBER-trained probe (ceiling)", name, ylab, pe.probe_scores(f_in, Xq[hs0][idx]), g, base,
                                      s_orig, **dict(n_boot=n_boot, n_boot_paired=n_boot_paired, seed=seed),
                                      extra=dict(layer=hs0, C=f0["param"], uses_target_domain_stats=True)))
            # reproduction check against the main run (same code path, deterministic)
            for ref in reference_dirs:                     # first main-run folder that has this model (order matters: qwen_logreg before cross_model_*)
                f = Path(ref) / "transfer_amber.csv"
                if not f.exists():
                    continue
                t = pd.read_csv(f)
                t = t[(t.model == model_key) & (t.task == "B") & (t.eval_set == "test") & (t.subset == sub) & (t.method == "probe POPE->AMBER")]
                if len(t):
                    reproduce[name] = dict(reference=str(f), reference_auroc=float(t.auroc.iloc[0]),
                                           this_run_auroc=float([r for r in rows if r["experiment"] == "orig" and r["eval_set"] == name][0]["auroc"]))
                    break
    df = pd.DataFrame(rows)
    if reproduce:
        mx = max(abs(v["reference_auroc"] - v["this_run_auroc"]) for v in reproduce.values())
        log(f"[{model_key}] original probe reproduces the main run's AMBER Task B AUROCs: max |difference| = {mx:.2e}")
    else:
        log(f"[{model_key}] no reference transfer_amber.csv found; reproduction of the main run not checked")

    # ---- write outputs
    df.to_csv(out_dir / "followup_summary.csv", index=False)
    df[df.experiment.isin(["baseline", "orig", "E1a", "E1b"])].to_csv(out_dir / "e1_reweight.csv", index=False)
    df[df.experiment.isin(["baseline", "orig", "E2", "E2 sub-probe"])].to_csv(out_dir / "e2_answer_conditioned.csv", index=False)
    pd.DataFrame(e0_rows).to_csv(out_dir / "e0_sign_diagnostic.csv", index=False)
    lt = pd.DataFrame(layer_rows)
    lt.to_csv(out_dir / "layerwise_e1a_e2.csv", index=False)
    plot_followup(df, out_dir / "followup_transfer_taskB.png", model_key)
    cfg = dict(model=model_key, seed=seed, n_boot=n_boot, n_boot_paired=n_boot_paired, prediction=PREDICTION, probe="logreg",
               selection="POPE dev only (weighted dev AUROC for E1); AMBER train split used only for E1b error shares and the ceiling probe",
               e1b_amber_train_halluc_share=shares, selected={k: dict(layer=int(v[0]), C=v[1]["param"], pope_dev_auroc=float(v[1]["sel_auroc"])) for k, v in chosen.items()},
               reproduction_check=reproduce, rows=dict(pope=n_p, amber=n_a, pope_test=len(te), amber_test={s: len(a_te[s]) for s in SUBSETS}),
               cache_files={d: dict(zip(("features", "metadata"), [p.name for p in find_cache(d, model_key, cache_dir)])) for d in ("pope", "amber")})
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return df


# --------------------------------------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------------------------------------
def plot_followup(df: pd.DataFrame, path: Path, model_key: str):
    """Grouped bars per AMBER subset (Task B, AMBER test split): best baseline, original probe, E1a, E1b, E2, AMBER-trained ceiling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [("baseline", "best output baseline", "#9aa5b1"), ("orig", "original POPE probe", "#d1495b"), ("E1a", "E1a balanced weights", "#edae49"),
             ("E1b", "E1b AMBER-matched (uses target stats)", "#f4d58d"), ("E2", "E2 answer-conditioned", "#66a182"),
             ("ceiling", "AMBER-trained ceiling", "#2e86ab")]
    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(4.6 * len(SUBSETS), 4.2), sharey=True)
    for ax, sub in zip(np.atleast_1d(axes), SUBSETS):
        d = df[df.eval_set == f"AMBER {sub} test"]
        for j, (exp, lab, col) in enumerate(order):
            r = d[d.experiment == exp]
            if exp == "E1b":
                r = r[r.variant.str.contains(f"({sub})", regex=False)]
            if r.empty or np.isnan(r["auroc"].iloc[0]):
                continue
            r = r.iloc[0]
            v = r["auroc"]
            err = [[max(v - r["ci_lo"], 0) if not np.isnan(r["ci_lo"]) else 0], [max(r["ci_hi"] - v, 0) if not np.isnan(r["ci_hi"]) else 0]]
            ax.bar(j, v, 0.8, yerr=err, capsize=3, color=col, label=lab if sub == SUBSETS[0] else None)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        n_pos = int(d[d.experiment == "orig"]["n_pos"].iloc[0]) if len(d[d.experiment == "orig"]) else 0
        ax.set_title(f"AMBER {sub} (test positives = {n_pos})", fontsize=10)
        ax.set_xticks([])
        ax.set_ylim(0, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("Task B AUROC (95% image-bootstrap CI)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{model_key}: does removing the answer-direction shortcut repair POPE -> AMBER transfer? (dashed = chance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["qwen2vl-2b", "smolvlm-500m"])
    p.add_argument("--cache-dir", default="results/cached_features")
    p.add_argument("--out-dir", default="results/final/followup_errormix")
    p.add_argument("--seed", type=int, default=0, help="bootstrap and probe seed (keep 0 to reproduce the main run)")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--n-boot-paired", type=int, default=1000)
    p.add_argument("--reference-dirs", nargs="*", default=["results/final/qwen_logreg", "results/final/cross_model_logreg"],
                   help="main-run folders whose transfer_amber.csv the original probe is checked against")
    a = p.parse_args(argv)
    for m in a.models:
        out = Path(a.out_dir) / m
        out.mkdir(parents=True, exist_ok=True)
        (out / "command.txt").write_text("python src/followup_errormix.py " + " ".join(argv if argv is not None else sys.argv[1:]) + "\n")
        log = Tee(out / "run.log")
        log("PREDICTION (stated before any result):", PREDICTION)
        run_model(m, a.cache_dir, out, a.seed, a.n_boot, a.n_boot_paired, log, a.reference_dirs)
        log(f"[{m}] results written to {out}")


if __name__ == "__main__":
    main()
