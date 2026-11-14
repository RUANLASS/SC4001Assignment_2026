#!/usr/bin/env python
"""Write results/final/SUMMARY.md from the CSVs produced by src/probe_transfer_eval.py.

    python src/make_summary.py --final-dir results/final --cache-dir results/cached_features
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HID, LOG, COMB = "probe (logreg)", "probe (logit-only logreg)", "probe (hidden+logits, logreg)"


# small formatting helpers
def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def with_ci(r) -> str:
    """'0.892 [0.851, 0.925]' from a row with auroc / ci_lo / ci_hi."""
    return f"{f3(r['auroc'])} [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]"


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def read(folder: Path, name: str):
    p = folder / name
    return pd.read_csv(p) if p.exists() else None


def missing_caches(cache_dir: Path) -> list[str]:
    """Metadata files that are committed without the matching features .pt (e.g. not copied from Drive yet)."""
    out = []
    for m in sorted(cache_dir.glob("*_metadata.parquet")):
        stem = m.name[: -len("_metadata.parquet")]
        feats = [f"{stem}_features.pt"] + (["pope_full_features.pt"] if stem == "pope" else [])
        if not any((cache_dir / f).exists() for f in feats):
            out.append(feats[0])
    return out


# sections
def sec_takeaways(main: Path | None, runs: dict[str, Path], cache_dir: Path | None = None) -> str:
    """Key numbers and the caveats that go with them, all read from the CSVs of the main run (and the ablations)."""
    if main is None:
        return ""
    d, p, lw = read(main, "pope_test_summary.csv"), read(main, "paired_tests.csv"), read(main, "layerwise_pope.csv")
    s = "## Key results and caveats (all numbers read from the CSVs below)\n\n"
    for task in ("B", "A"):
        t = d[d.task == task]
        pr, base = t[t.method == HID].iloc[0], t[t.method.str.startswith("baseline")]
        bb = base.loc[base.auroc.idxmax()]
        q = p[(p.task == task) & (p.eval_set == "POPE test") & (p.probe == HID) & (p.reference == bb.method)].iloc[0]
        lg = p[(p.task == task) & (p.eval_set == "POPE test") & (p.probe == HID) & (p.reference == LOG)].iloc[0]
        s += (f"* **Task {task}** ({int(pr.n_pos)} positives in the test split): hidden-state probe (layer {int(pr.layer)}) AUROC {with_ci(pr)} vs best "
              f"output-level baseline ({bb.method.replace('baseline: ', '')}) {with_ci(bb)}. Paired difference {q['diff']:+.3f} "
              f"[{q.ci_lo:+.3f}, {q.ci_hi:+.3f}], p = {f3(q.p_value)}. Versus a probe trained on the four output signals only: {lg['diff']:+.3f} "
              f"[{lg.ci_lo:+.3f}, {lg.ci_hi:+.3f}], p = {f3(lg.p_value)}.\n")
        if task == "B" and lw is not None:
            l = lw[lw.task == "B"].sort_values("layer")
            above = l[l.test_auroc > bb.auroc]
            s += (f"  * Descriptive only (per-layer test AUROC, not used for selection): entry 0 (token embedding, identical for every prompt) gives "
                  f"{f3(l.iloc[0].test_auroc)}; the per-layer probe first exceeds the best baseline at entry {int(above.layer.min()) if len(above) else 'none'} "
                  f"and the best layer by the selection metric is {int(l.loc[l.sel_auroc.idxmax(), 'layer'])} (entries run 0 to {int(l.layer.max())}).\n")
    s += ("* **Task B is the primary result.** Task A has few positives, so its CIs are wide and its per-setting rows (e.g. the `random` setting) "
          "rest on a handful of positives; do not draw conclusions from them.\n")
    if "qwen_logreg" in runs and "qwen_logreg_cv5" in runs:
        a, b = d, read(runs["qwen_logreg_cv5"], "pope_test_summary.csv")
        la = [int(a[(a.task == t) & (a.method == HID)].iloc[0].layer) for t in "BA"]
        lb = [int(b[(b.task == t) & (b.method == HID)].iloc[0].layer) for t in "BA"]
        ca = [with_ci(a[(a.task == t) & (a.method == HID)].iloc[0]) for t in "BA"]
        cb = [with_ci(b[(b.task == t) & (b.method == HID)].iloc[0]) for t in "BA"]
        s += (f"* Layer selection: dev-split picked layer {la[0]} (B) / {la[1]} (A) with test AUROC {ca[0]} / {ca[1]}; image-grouped 5-fold CV picked "
              f"layer {lb[0]} (B) / {lb[1]} (A) with test AUROC {cb[0]} / {cb[1]}. Compare the intervals before reading anything into a layer difference.\n")
    s += ("* The last hidden-state entry is what the output layer reads, so a probe there can partly decode the answer distribution. The logit-only probe "
          "(four output signals) is therefore the fairest 'does the hidden state add information' control; the paired row 'probe vs logit-only' above is that test.\n"
          "* The balanced-class-weight probe outputs are not calibrated probabilities (see Calibration; Platt scaling fitted on dev repairs this). "
          "AUROC is unaffected by calibration.\n"
          "* The p-values are not corrected for the many comparisons in the paired tables.\n")
    ta = read(main, "transfer_amber.csv")
    if ta is not None and len(ta):
        t = ta[(ta.eval_set == "test") & (ta.task == "B") & (ta.model == ta.model.iloc[0])]
        parts = []
        for sub in ("existence", "attribute", "relation"):
            g = t[t.subset == sub]
            tr, ce = g[g.method == "probe POPE->AMBER"], g[g.method == "probe AMBER->AMBER"]
            bl = g[g.method.str.startswith("baseline")]
            if len(tr) and len(ce) and len(bl):
                parts.append(f"{sub}: transferred {with_ci(tr.iloc[0])}, best baseline {f3(bl.auroc.max())}, AMBER-trained ceiling {with_ci(ce.iloc[0])}")
        if parts:
            s += ("* **Transfer (Task B, AMBER test split, n_pos in the subset tables below):** " + "; ".join(parts) + ". Values below 0.5 mean the POPE-trained "
                  "probe ranks AMBER errors the wrong way round. The AMBER-trained ceiling shows the signal is present in-domain, so the failure is the "
                  "transfer of the probe direction, not the absence of signal.\n")
    if cache_dir is not None and (cache_dir / "pope_metadata.parquet").exists():
        pm = error_mix(pd.read_parquet(cache_dir / "pope_metadata.parquet"), "pope")[0]
        am = [r for m in sorted(cache_dir.glob("amber_qwen2vl-2b_metadata.parquet")) for r in error_mix(pd.read_parquet(m), "amber")]
        if am:
            s += (f"* **Plausible (not proven) reason for the failed transfer:** error composition differs. POPE errors are {pm['missed_yes_share']} missed-yes "
                  "and the rest hallucinated-yes, whereas for AMBER " + ", ".join(f"{r['subset']} is {r['hallucinated_yes_share']} hallucinated-yes" for r in am if r["subset"] != "all")
                  + ". A probe trained on POPE errors can therefore partly learn 'answered no although the object is present', which is the wrong signal for AMBER-attribute. "
                  "Task A (answered-yes questions only, so the answer direction is fixed) is the cleaner transfer test, but it has very few positives per AMBER subset.\n")
    s += ("* Scope: Qwen2-VL-2B (main) and SmolVLM-500M (cross-model run, shared 3,000 questions) on POPE, and AMBER with 1,000 queries per subset "
          "(existence / attribute / relation), one image-level split and one extraction run per cache. Qwen2.5-VL-3B was not run.\n\n")
    return s


def sec_runs(final: Path, runs: list[Path]) -> str:
    rows = []
    for r in runs:
        cfg = json.loads((r / "run_config.json").read_text())
        o = cfg["options"]
        cmd = (r / "command.txt").read_text().strip() if (r / "command.txt").exists() else ""
        rows.append(dict(run=r.name, models=" ".join(cfg["models"]), probe=cfg["probe"], class_weight=o["class_weight"],
                         pca=list(o["pca"]) or "none", selection=f"{o['cv_k']}-fold image-grouped CV" if o["cv_k"] else "dev split",
                         seeds=list(cfg["probe_seeds"]), pope_rows=cfg["rows_used"][cfg["models"][0]]["pope_rows"], command=f"`{cmd}`"))
    return "## Runs included\n\nEach folder holds `command.txt` (exact command), `run_config.json` and `run.log`.\n\n" + md_table(pd.DataFrame(rows))


def error_mix(meta: pd.DataFrame, label: str) -> list[dict]:
    """Composition of a cache's errors: hallucinated yes (pred yes, truth no) vs missed yes (pred no, truth yes)."""
    rows = []
    groups = [("all", meta)] + ([(f"{k}", g) for k, g in meta.groupby("setting")] if meta["setting"].nunique() > 1 else [])
    for name, g in groups:
        py, gy = (g.pred_answer == "yes"), (g.gt_answer == "yes")
        err = py != gy
        rows.append(dict(cache=label, subset=name, n=len(g), errors=int(err.sum()),
                         hallucinated_yes_share=f3(float((py & ~gy)[err].mean())) if err.any() else "n/a",
                         missed_yes_share=f3(float((~py & gy)[err].mean())) if err.any() else "n/a",
                         model_yes_rate=f3(float(py.mean())), true_yes_rate=f3(float(gy.mean()))))
    return rows


def sec_error_mix(cache_dir: Path) -> str:
    rows = []
    for m in sorted(cache_dir.glob("*_metadata.parquet")):
        meta = pd.read_parquet(m)
        if {"setting", "gt_answer", "pred_answer"} <= set(meta.columns):
            rows += error_mix(meta, m.name[: -len("_metadata.parquet")])
    if not rows:
        return ""
    return ("## Error composition of each cache\n\nTask B 'error' mixes two different mistakes. The probe is trained on whichever mix its training "
            "data has, which matters for transfer (compare POPE with the AMBER subsets).\n\n" + md_table(pd.DataFrame(rows)) + "\n")


def sec_layer_transfer(run: Path) -> str:
    l = read(run, "layerwise_pope.csv")
    if l is None or "transfer_attribute" not in l.columns or l["transfer_attribute"].isna().all():
        return ""
    keep = sorted(set(list(l.layer.iloc[::4]) + [int(l.layer.max())]))
    s = ("## Per-layer POPE -> AMBER transfer (descriptive)\n\n`transfer_*` is the AUROC of the POPE-trained probe at that layer on all AMBER rows of the "
         "subset (train/dev/test rows alike, since nothing is fitted on AMBER). The reported transfer result uses the layer chosen on POPE dev only; "
         "choosing a layer by looking at this table would be selection on AMBER and is not done.\n\n")
    for task in ("B", "A"):
        t = l[(l.task == task) & l.layer.isin(keep)][["layer", "test_auroc", "transfer_existence", "transfer_attribute", "transfer_relation"]].round(3).astype({"layer": int})
        s += f"**Task {task}**\n\n" + md_table(t) + "\n"
    return s


def sec_headline(run: Path, title: str) -> str:
    """Per model and task: hidden probe vs best output baseline on POPE test, and on AMBER test if transfer was run."""
    d, p, cfg = read(run, "pope_test_summary.csv"), read(run, "paired_tests.csv"), json.loads((run / "run_config.json").read_text())
    if d is None or p is None:
        return ""
    hid = f"probe ({cfg['probe']})"
    rows, trows = [], []
    for model in cfg["models"]:
        for task in ("B", "A"):
            t = d[(d.model == model) & (d.task == task) & (d.seed == d[(d.model == model) & (d.task == task)].seed.min())]
            if t[t.method == hid].empty:
                continue
            pr, base = t[t.method == hid].iloc[0], t[t.method.str.startswith("baseline")]
            bb = base.loc[base.auroc.idxmax()]
            q = p[(p.model == model) & (p.task == task) & (p.eval_set == "POPE test") & (p.probe == hid) & (p.reference == bb.method)]
            q = q[q.seed == q.seed.min()].iloc[0]
            lo = p[(p.model == model) & (p.task == task) & (p.eval_set == "POPE test") & (p.probe == hid) & (p.reference == LOG)]
            lo = lo[lo.seed == lo.seed.min()].iloc[0]
            rows.append(dict(model=model, task=task, test_n=int(pr.n), n_pos=int(pr.n_pos), probe_layer=int(pr.layer), probe_AUROC=with_ci(pr),
                             best_baseline=bb.method.replace("baseline: ", ""), baseline_AUROC=with_ci(bb),
                             diff_vs_baseline=f"{q['diff']:+.3f} [{q.ci_lo:+.3f}, {q.ci_hi:+.3f}] p={f3(q.p_value)}",
                             diff_vs_logit_only=f"{lo['diff']:+.3f} [{lo.ci_lo:+.3f}, {lo.ci_hi:+.3f}] p={f3(lo.p_value)}"))
            for sub in ("existence", "attribute", "relation"):
                tt = p[(p.model == model) & (p.task == task) & (p.eval_set == f"AMBER {sub} test") & (p.probe == "probe POPE->AMBER")
                       & (p.reference_type == "output baseline")]
                if tt.empty:
                    continue
                tt = tt[(tt.seed == tt.seed.min()) & tt.auc_baseline.notna()]
                if tt.empty:                                  # undefined AUROC (single class in this subset/task)
                    continue
                best = tt.loc[tt.auc_baseline.idxmax()]
                trows.append(dict(model=model, task=task, subset=sub, n_pos=int(best.n_pos), transferred_probe_AUROC=f3(best.auc_probe),
                                  best_baseline=best.reference.replace("baseline: ", ""), baseline_AUROC=f3(best.auc_baseline),
                                  diff=f"{best['diff']:+.3f} [{best.ci_lo:+.3f}, {best.ci_hi:+.3f}] p={f3(best.p_value)}"))
    s = f"## {title}\n\nHidden-state probe vs best output-level baseline, POPE test split (paired image bootstrap; probe layer chosen on POPE dev).\n\n" + md_table(pd.DataFrame(rows))
    if trows:
        s += ("\n**POPE-trained probe applied unchanged to AMBER (test split rows) vs the best output-level baseline on the same rows**\n\n"
              + md_table(pd.DataFrame(trows)))
    return s + "\n"


def sec_blocked(cache_dir: Path, runs: list[Path]) -> str:
    miss = missing_caches(cache_dir)
    amber_done = any((r / "transfer_amber.csv").exists() for r in runs)
    s = "## Not run (blocked by missing cache files)\n\n"
    if not miss:
        return s + "All caches with committed metadata have their feature files.\n\n"
    s += ("The following feature files were **not available** when these results were produced (only their metadata `.parquet` / "
          "`run_config.json` are in the repo). Per the project rules nothing was substituted or simulated.\n\n")
    s += "".join(f"* `{m}`\n" for m in miss)
    s += "\nConsequently the following were **not run**: " + ("" if not amber_done else "(AMBER transfer exists for some runs, see below) ")
    s += ("POPE -> AMBER transfer and AMBER -> AMBER ceilings; the cross-model comparison (SmolVLM-500M, Qwen2.5-VL-3B) and the "
          "paired tests on AMBER subsets. The code for all of these is implemented and unit-tested on synthetic caches; rerun the "
          "commands in the README once the files are in `results/cached_features/`.\n\n")
    return s


def sec_pope(run: Path, title: str) -> str:
    d = read(run, "pope_test_summary.csv")
    if d is None:
        return f"## {title}\n\n_not run_\n\n"
    s = f"## {title}\n\nPOPE test split; AUROC with 95% image-bootstrap CI. Layer, regularisation and (if used) PCA size were chosen on dev only.\n\n"
    for task, name in [("B", "Task B (primary): wrong vs right answer, all questions"),
                       ("A", "Task A (secondary): hallucinated-yes vs correct-yes, answered-yes only")]:
        t = d[(d.task == task) & (d.seed == d[d.task == task].seed.min())]
        if t.empty:
            continue
        npos = int(t.n_pos.iloc[0])
        lay = [("" if pd.isna(l) or "logit-only" in m else int(l)) for m, l in zip(t.method, t.layer)]     # layer is meaningless for logit-only
        tab = pd.DataFrame(dict(method=t.method, layer=lay, AUROC_CI=t.apply(with_ci, axis=1)))
        s += f"**{name}** (test n = {int(t.n.iloc[0])}, positives = {npos})\n\n" + md_table(tab) + "\n"
    return s


def sec_paired(run: Path) -> str:
    p = read(run, "paired_tests.csv")
    if p is None:
        return ""
    s = ("## Paired comparison (probe minus reference AUROC, same rows, paired image bootstrap)\n\n"
         "`diff` > 0 means the first method is better. CI = 95% percentile; p = two-sided bootstrap p-value (smallest possible value is about "
         "2/(B+1)). **No multiple-comparison correction** was applied across the rows of this table.\n\n")
    for task in ("B", "A"):
        for ev in sorted(p.eval_set.unique(), key=lambda x: (x != "POPE test", x)):
            d = p[(p.task == task) & (p.eval_set == ev) & (p.seed == p.seed.min())]
            if d.empty:
                continue
            d = d.assign(diff_CI=d.apply(lambda r: f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", axis=1),
                         p=d.p_value.map(f3), n_pos=d.n_pos)
            tab = d[["probe", "reference", "auc_probe", "auc_baseline", "diff_CI", "p", "n_pos"]].round(3)
            s += f"**Task {task}, {ev}**\n\n" + md_table(tab) + "\n"
    return s


def sec_setting(run: Path) -> str:
    d = read(run, "pope_by_setting.csv")
    if d is None:
        return ""
    s = "## By POPE setting (test split; AUROC [95% CI])\n\nPer-setting positives are few for Task A (see n_pos); read those rows as indicative only.\n\n"
    for task in ("B", "A"):
        t = d[d.task == task]
        if t.empty:
            continue
        t = t.assign(cell=t.apply(with_ci, axis=1))
        piv = t.pivot_table(index="method", columns="setting", values="cell", aggfunc="first")[["overall", "random", "popular", "adversarial"]]
        npos = t.drop_duplicates("setting").set_index("setting")["n_pos"]
        s += f"**Task {task}** (n_pos: " + ", ".join(f"{k}={int(npos[k])}" for k in ["overall", "random", "popular", "adversarial"]) + ")\n\n"
        s += md_table(piv.reset_index())
        s += "\n"
    return s


def sec_seeds(run: Path) -> str:
    d = read(run, "pope_test_seed_summary.csv")
    if d is None:
        return ""
    d = d[d.n_seeds > 1]
    if d.empty:
        return ""
    t = d.assign(mean_std=d.apply(lambda r: f"{r.auroc_mean:.3f} ± {f3(r.auroc_std)}", axis=1))
    t.loc[t.method.str.contains("logit-only"), "per_seed_layer"] = ""      # logit-only logreg is deterministic and has no hidden layer
    return ("## MLP probe across seeds (test AUROC; whole layer/regularisation selection repeated per seed)\n\n"
            + md_table(t[["model", "task", "method", "n_seeds", "mean_std", "per_seed_auroc", "per_seed_layer"]]) + "\n")


def sec_ablation(runs: dict[str, Path]) -> str:
    rows = []
    for name, r in runs.items():
        d = read(r, "pope_test_summary.csv")
        if d is None:
            continue
        for task in ("B", "A"):
            t = d[(d.task == task) & (d.method == HID)]
            if t.empty:
                continue
            t = t.iloc[0]
            pk = t.get("pca_k")
            rows.append(dict(run=name, task=task, layer=int(t.layer), param=t.get("param"), pca="none" if pd.isna(pk) else int(pk),
                             test_AUROC_CI=with_ci(t), n_pos=int(t.n_pos)))
    if not rows:
        return ""
    return ("## Ablations (hidden-state logreg probe, POPE test)\n\nSame split, same test rows; only the listed option changes. "
            "Differences between these runs are mostly within the CI width, so do not over-read them.\n\n" + md_table(pd.DataFrame(rows)) + "\n")


def sec_calibration(run: Path) -> str:
    d = read(run, "calibration.csv")
    if d is None:
        return ""
    s = ("## Calibration of the probability of the bad outcome (POPE test; 15 equal-width bins)\n\n"
         "The probe uses balanced class weights, which shifts its raw probabilities upwards, so the raw probe is compared with the model's "
         "own confidence and with a Platt-scaled probe (Platt fitted on dev only, or out-of-fold when CV is used). `brier_constant_base_rate` "
         "is the Brier score of always predicting the test base rate.\n\n")
    for task in ("B", "A"):
        t = d[d.task == task]
        if not t.empty:
            s += f"**Task {task}** (n = {int(t.n.iloc[0])}, positives = {int(t.n_pos.iloc[0])})\n\n"
            s += md_table(t[["method", "ece", "brier", "brier_constant_base_rate"]].round(3)) + "\n"
    return s


def sec_aurc(run: Path) -> str:
    d = read(run, "aurc.csv")
    if d is None:
        return ""
    t = d.round(4)
    return ("## Selective prediction (Task B scores; abstain on the answers judged most likely wrong)\n\n"
            "AURC = mean risk over coverage (lower is better); AUACC = 1 - AURC. `random_aurc` is the error rate, `oracle_aurc` abstains on all errors first.\n\n"
            + md_table(t[["eval_set", "method", "aurc", "auacc", "oracle_aurc", "random_aurc", "n"]]) + "\n")


def sec_transfer(runs: list[Path]) -> str:
    s = "## POPE -> AMBER transfer\n\n"
    found = [(r, read(r, "transfer_amber.csv")) for r in runs]
    found = [(r, d) for r, d in found if d is not None and len(d)]
    if not found:
        return s + "**Not run.** No AMBER feature cache was available (see the section on missing caches).\n\n"
    for r, d in found:
        t = d[d.eval_set == "test"]
        for task in ("B", "A"):
            x = t[t.task == task]
            if x.empty:
                continue
            x = x.assign(AUROC_CI=x.apply(with_ci, axis=1))
            s += f"**{r.name}, Task {task}, AMBER test split**\n\n" + md_table(x[["model", "subset", "method", "AUROC_CI", "n", "n_pos"]]) + "\n"
    return s


def h5_verdict(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Verdict on H5 for one model, from followup_summary.csv:

    * SUPPORTED: some target-free variant (E1a or E2) has, on BOTH AMBER existence and attribute (test split), a 95% CI entirely above 0.5 AND a
      significantly positive paired difference to the original probe (CI lower bound > 0).
    * PARTLY SUPPORTED: not the above, but some variant (E1a, E2 or E1b, which uses target-domain statistics) is significantly better than the
      original probe (paired CI lower bound > 0) on at least one of existence / attribute.
    * NOT SUPPORTED: otherwise.
    """
    why, ok_variants, improved_any = [], [], []
    for exp in ("E1a", "E2", "E1b"):
        flags = {}
        for sub in ("existence", "attribute"):
            r = df[(df.experiment == exp) & (df.eval_set == f"AMBER {sub} test")]
            if exp == "E1b":
                r = r[r.variant.str.contains(f"({sub})", regex=False)]
            if r.empty:
                continue
            r = r.iloc[0]
            above = bool(r["ci_lo"] > 0.5)
            improved = bool(r["diff_vs_orig_lo"] > 0)
            flags[sub] = (above, improved)
            why.append(f"{exp} on AMBER {sub}: AUROC {with_ci(r)}, CI above 0.5: {above}; paired difference to original probe "
                       f"{r['diff_vs_orig']:+.3f} [{r['diff_vs_orig_lo']:+.3f}, {r['diff_vs_orig_hi']:+.3f}], significantly positive: {improved}")
        if exp != "E1b" and len(flags) == 2 and all(a and i for a, i in flags.values()):
            ok_variants.append(exp)
        if any(i for _, i in flags.values()):
            improved_any.append(exp)
    if ok_variants:
        return "SUPPORTED", why
    return ("PARTLY SUPPORTED" if improved_any else "NOT SUPPORTED"), why


def sec_followup(final: Path) -> str:
    """Follow-up error-mix test (src/followup_errormix.py): E0 sign diagnostic, E1a/E1b re-weighting, E2 answer-conditioned probes."""
    root = final / "followup_errormix"
    models = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "followup_summary.csv").exists()) if root.exists() else []
    s = "## Follow-up: error-mix test (H5)\n\n"
    if not models:
        return s + "_not run_\n\n"
    cfg0 = json.loads((root / models[0] / "run_config.json").read_text())
    s += ("**Prediction (written before the results were computed):** " + cfg0["prediction"] + "\n\n"
          "**Verdict rule (fixed in advance):** SUPPORTED if a target-free variant (E1a or E2) has, on both AMBER existence and attribute, a 95% CI entirely "
          "above 0.5 and a significantly positive paired difference to the original probe; PARTLY SUPPORTED if any variant (E1b included) is "
          "significantly better than the original probe on at least one of those two subsets; otherwise NOT SUPPORTED. E1b uses AMBER-train error "
          "statistics (target-domain information) and can never produce 'supported'. Layer / C are selected on POPE dev only; AMBER dev/test labels "
          "are never used for any choice.\n\n")
    for m in models:
        d = pd.read_csv(root / m / "followup_summary.csv")
        e0 = pd.read_csv(root / m / "e0_sign_diagnostic.csv")
        cfg = json.loads((root / m / "run_config.json").read_text())
        lw = pd.read_csv(root / m / "layerwise_e1a_e2.csv")
        verdict, why = h5_verdict(d)
        s += f"### {m}\n\n**Verdict: H5 {verdict}.**\n\n" + "".join(f"* {w}\n" for w in why) + "\n"
        s += ("E1b target-domain error shares (hallucinated-yes share of errors, AMBER train split): "
              + ", ".join(f"{k} {v:.3f}" for k, v in cfg["e1b_amber_train_halluc_share"].items())
              + ". Selected layer / C on POPE dev: " + "; ".join(f"{k}: layer {v['layer']}, C={v['C']}" for k, v in cfg["selected"].items()) + ".\n\n")
        if cfg.get("reproduction_check"):
            diffs = [abs(v["reference_auroc"] - v["this_run_auroc"]) for v in cfg["reproduction_check"].values()]
            s += f"Check: the original probe here reproduces the main run's AMBER Task B AUROCs (max |difference| {max(diffs):.1e}).\n\n"
        # E0
        t = e0.assign(AUROC_CI=e0.apply(with_ci, axis=1))[["eval_set", "score", "rows", "AUROC_CI", "n", "n_pos"]]
        s += "**E0 sign diagnostic** (no training; original POPE probe split by the model's answer, flipped score, trivial 'answered no' score):\n\n" + md_table(t) + "\n"
        # main table, AMBER
        order = ["baseline", "orig", "E1a", "E1b", "E2", "ceiling"]
        for ev in ["AMBER existence test", "AMBER attribute test", "AMBER relation test", "POPE test"]:
            x = d[(d.eval_set == ev) & d.experiment.isin(order)].copy()
            x["o"] = x.experiment.map({k: i for i, k in enumerate(order)})
            x = x.sort_values(["o", "variant"])
            if ev != "POPE test":
                x = x[~((x.experiment == "E1b") & ~x.variant.str.contains(ev.split()[1], regex=False))]
            def pdiff(r, tag):
                if pd.isna(r.get(f"diff_vs_{tag}")):
                    return ""
                return f"{r[f'diff_vs_{tag}']:+.3f} [{r[f'diff_vs_{tag}_lo']:+.3f}, {r[f'diff_vs_{tag}_hi']:+.3f}] p={f3(r[f'diff_vs_{tag}_p'])}"
            tab = pd.DataFrame(dict(variant=x.variant, AUROC_CI=x.apply(with_ci, axis=1), diff_vs_best_baseline=x.apply(lambda r: pdiff(r, "baseline"), axis=1),
                                    diff_vs_original_probe=x.apply(lambda r: pdiff(r, "orig"), axis=1), n=x.n, n_pos=x.n_pos))
            s += (f"**Task B on {ev}**" + (" (cost of each variant on the training domain)" if ev == "POPE test" else "") + "\n\n" + md_table(tab) + "\n")
        # E2 sub probes
        x = d[(d.experiment == "E2 sub-probe") & (d.eval_set != "POPE test")]
        x = pd.concat([d[(d.experiment == "E2 sub-probe") & (d.eval_set == "POPE test")], x])
        s += "**E2 sub-probes on their own rows** (answered-yes rows: label = hallucinated yes; answered-no rows: label = missed object)\n\n"
        s += md_table(pd.DataFrame(dict(variant=x.variant, eval_set=x.eval_set, AUROC_CI=x.apply(with_ci, axis=1), n=x.n, n_pos=x.n_pos))) + "\n"
        # per layer (descriptive)
        keep = lw[lw.layer.isin([8, 12, 16, 20, 24, 28])]
        cols = ["variant", "layer", "pope_test_auroc", "amber_existence_test_auroc", "amber_attribute_test_auroc", "amber_relation_test_auroc"]
        tb = keep[keep.variant.isin(["orig", "e1a", "e2"])][cols].round(3)
        s += ("**Per-layer AUROC (descriptive only, not used for selection; AMBER test split, so small n)**\n\n" + md_table(tb.astype({"layer": int})) + "\n")
        s += f"![follow-up transfer, {m}](followup_errormix/{m}/followup_transfer_taskB.png)\n\n"
    return s


def build_summary(final: Path, cache_dir: Path) -> str:
    runs = sorted(p for p in final.iterdir() if p.is_dir() and (p / "run_config.json").exists()) if final.exists() else []
    byname = {r.name: r for r in runs}
    main = byname.get("qwen_logreg")
    s = ["# Probe vs output-level baselines: results summary\n\n"
         "Generated by `src/make_summary.py` from the CSVs in this folder; every number below is read from a CSV. "
         "Protocol is frozen (prompt, forced yes/no scoring, last prompt token, image-level 70/15/15 split from the `split` column). "
         "Test data was used only for the numbers reported here; layer, regularisation, PCA size and class weighting were chosen on dev "
         "(or by image-grouped CV on train+dev).\n\n"]
    s.append(sec_runs(final, runs))
    s.append(sec_blocked(cache_dir, runs))
    s.append(sec_takeaways(main, byname, cache_dir))
    if main is not None:
        s.append(sec_pope(main, "Main result: Qwen2-VL-2B, logistic-regression probe, POPE test"))
        s.append(sec_paired(main))
        s.append(sec_setting(main))
        s.append(sec_calibration(main))
        s.append(sec_aurc(main))
    if "qwen_mlp" in byname:
        s.append(sec_pope(byname["qwen_mlp"], "Qwen2-VL-2B, MLP probe (seed 0 rows), POPE test"))
        s.append(sec_seeds(byname["qwen_mlp"]))
    abl = {k: v for k, v in byname.items() if k.startswith("qwen_logreg") or k == "qwen_mlp"}
    s.append(sec_ablation(abl))
    cv = byname.get("qwen_logreg_cv5")
    if cv is not None:
        s.append("## Layer selection by image-grouped 5-fold CV (train+dev)\n\nSee `qwen_logreg_cv5/layerwise_auroc_task{A,B}.png` (CV AUROC ± std over folds, and test AUROC per layer).\n\n")
    if main is not None:
        s.append(sec_headline(main, "POPE -> AMBER transfer, Qwen2-VL-2B (logreg probe)") if (main / "transfer_amber.csv").exists() else "")
        s.append(sec_layer_transfer(main))
    s.append(sec_error_mix(cache_dir))
    s.append(sec_followup(final))
    s.append(sec_transfer(runs))
    cross = [r for r in runs if len(json.loads((r / "run_config.json").read_text())["models"]) > 1]
    s.append("## Cross-model comparison\n\n")
    if cross:
        s.append("Both models are restricted to the questions they share (3,000 POPE questions, 3,000 AMBER queries), so these numbers are on a smaller "
                 "subset than the main Qwen run and are not directly comparable with its confidence intervals.\n\n")
        for r in cross:
            s.append(sec_headline(r, f"{r.name}: " + ", ".join(json.loads((r / "run_config.json").read_text())["models"])))
            sd = sec_seeds(r)
            s.append(sd.replace("## MLP probe across seeds", f"### {r.name}: MLP probe across seeds") if sd else "")
    else:
        s.append("**Not run.** It needs the SmolVLM-500M (and optionally Qwen2.5-VL-3B) POPE feature caches, which were not available.\n\n")
    return "".join(s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final-dir", default="results/final")
    p.add_argument("--cache-dir", default="results/cached_features")
    a = p.parse_args(argv)
    text = build_summary(Path(a.final_dir), Path(a.cache_dir))
    out = Path(a.final_dir) / "SUMMARY.md"
    out.write_text(text)
    print("wrote", out, f"({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
