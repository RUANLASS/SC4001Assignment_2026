# NOTES: hallucination-probe experiments (status, results, caveats)

Last updated 2026-11-08. All numbers are copied from the CSVs in `results/final/` (full tables: `results/final/SUMMARY.md`, regenerate with `python src/make_summary.py`).

## 1. Status

All planned CPU work is done on real caches. The only thing left undone is the optional third model (Qwen2.5-VL-3B), which needs a GPU: `notebooks/04_colab_qwen25vl_3b_pope_amber.ipynb` is written but **GPU REQUIRED - RUN IN COLAB - NOT TESTED LOCALLY** (only syntax-checked).

Caches used (all verified: shapes match the run configs, feature row ids match the metadata, `split` equals the md5 rule, no image in two splits):

| Cache | Shape | Rows |
|---|---|---|
| `pope_full_features.pt` (Qwen2-VL-2B) | 9000 x 29 x 1536 | all POPE |
| `amber_qwen2vl-2b_features.pt` | 3000 x 29 x 1536 | 1,000 per AMBER subset |
| `pope_smolvlm-500m_features.pt` | 3000 x 33 x 960 | shared 3,000 POPE questions |
| `amber_smolvlm-500m_features.pt` | 3000 x 33 x 960 | same 3,000 AMBER queries as Qwen |

Re-running the Qwen runs once the AMBER cache was present reproduced the earlier POPE AUROCs exactly (logreg 14 rows, MLP 26 rows, maximum difference 0.0, same selected layers).

## 2. What was done

* **Mock code removed:** deleted `train_probe_regression.py`, `evaluation_baseline.py`, `abstention_curves.py` (random data), all fake figures and `pope_subset_labels.pt`; the legacy 100-sample script is in `src/old/`. A test fails if mock or random-label code reappears in `src/`.
* **Probe options** (`src/probe_transfer_eval.py`): balanced class weights, wider grids, PCA fitted on train only with its size picked on dev, MLP seeds (mean and std), image-grouped CV selection (`--layer-cv`, fold band in the layer plot), Platt recalibration, logit-only and hidden+logits probes.
* **Statistics** (`src/probe_stats.py`): paired image-level bootstrap `paired_auc_diff`, ECE, Brier, Platt scaling, AURC/AUACC.
* **Outputs per run:** layer-wise, POPE summary, seed summary, paired tests, calibration, by-setting, AURC, transfer, figures, `command.txt`, `run_config.json`, `run.log`.
* **Runs** (`results/final/`): `qwen_logreg`, `qwen_mlp` (3 seeds), both now including AMBER transfer; ablations `qwen_logreg_pca`, `qwen_logreg_noclassweight`, `qwen_logreg_cv5` (POPE only, run before the AMBER cache was available); `cross_model_logreg` and `cross_model_mlp` (Qwen2-VL-2B and SmolVLM-500M, restricted to their shared 3,000 POPE questions and 3,000 AMBER queries).
* **`results/final/SUMMARY.md`** is generated from the CSVs by `src/make_summary.py`; it also reports the error composition of every cache and a per-layer transfer table.
* **Docs:** README (tree and "Reproducing the report results"), `docs/EXTRA_MODELS_AMBER.md` (status section).
* **Tests:** 82 passed, 1 skipped (35 passed originally; the skipped one is the existing optional AMBER-repo check). Tests use small synthetic caches only and load no real model.
* **Follow-up error-mix experiment (H5)**, `src/followup_errormix.py`, outputs in `results/final/followup_errormix/<model>/` (section 3.4). CPU only, no GPU or Colab work was needed.
* **No git commits or pushes were made.**

## 3. Results

### 3.1 POPE (Qwen2-VL-2B, logreg probe, test split)

| Task | Test positives | Probe AUROC [95% CI] | Best output-level baseline | Paired difference [95% CI], p |
|---|---|---|---|---|
| B: wrong vs right (primary) | 163 | 0.892 [0.851, 0.925] | 0.801 [0.736, 0.858] (token prob) | +0.090 [+0.046, +0.146], p = 0.002 |
| A: hallucinated yes vs correct yes | 28 | 0.906 [0.842, 0.950] | 0.832 [0.754, 0.895] (token prob) | +0.073 [+0.026, +0.128], p = 0.006 |

* A probe on the four output signals only gets 0.805 on Task B; the hidden-state probe beats it by +0.087 [+0.045, +0.140]. Adding the signals to the hidden probe changes nothing (+0.000).
* The per-layer probe first beats the best baseline at hidden-state entry 13 (of 0 to 28) and plateaus from about entry 18. Entry 0 is identical for every prompt and scores 0.500. (Descriptive; layers were chosen on dev or CV.)
* Selective prediction (Task B): AURC 0.0245 for the probe vs 0.0450 for token probability (error rate 0.124).
* Calibration: raw balanced-weight probe probabilities are poorly calibrated (Task B ECE 0.117 vs 0.045 for the model's own confidence); Platt scaling fitted on dev gives ECE 0.028.
* Variants (Task B test AUROC): MLP over 3 seeds 0.884 +/- 0.004; no class weight 0.893; PCA 256 0.887; CV selection 0.894. All lie within each other's CIs.

### 3.2 POPE -> AMBER transfer (Qwen2-VL-2B): the POPE-trained probe does NOT transfer

Task B, AMBER test split (probe applied unchanged; layer chosen on POPE dev):

| AMBER subset | n_pos | Transferred probe | Best baseline | Paired diff vs baseline [CI], p | AMBER-trained probe (ceiling) |
|---|---|---|---|---|---|
| existence | 22 | 0.453 [0.333, 0.571] | 0.837 | -0.384 [-0.524, -0.240], p = 0.002 | 0.856 [0.764, 0.946] |
| attribute | 16 | 0.352 [0.206, 0.497] | 0.812 | -0.460 [-0.586, -0.335], p = 0.002 | 0.945 [0.887, 0.991] |
| relation | 38 | 0.590 [0.505, 0.681] | 0.719 | -0.129 [-0.259, +0.007], p = 0.062 | 0.821 [0.745, 0.889] |

* Transferred AUROC is near or **below 0.5** on existence and attribute, i.e. the POPE probe ranks AMBER errors the wrong way round, and it is clearly worse than simple output-confidence baselines. The MLP probe shows the same (seed 0: 0.484 / 0.394 / 0.532).
* Probes trained on AMBER itself reach 0.82 to 0.95, so the signal is present in-domain; what fails is the transfer of the POPE-learned direction.
* Task A (answered-yes only, so the answer direction is fixed) transfers somewhat better but is still below the baselines: 0.674 / 0.542 / 0.585 against 0.819 / 0.677 / 0.728, with only 16 / 13 / 7 positives (only existence is significantly worse, p = 0.034).
* In **no** transfer cell (either model, either probe type, either task) was the transferred probe significantly better than the best baseline; in several it was significantly worse.
* **Plausible, not proven, explanation:** the error mix differs. 85% of Qwen's POPE errors are missed-yes (said no, object present); for AMBER, 71% of attribute errors and 58% of existence errors are hallucinated-yes (relation: 18%). A probe trained on POPE errors can learn "answered no although the object is present", which is the wrong signal for AMBER-attribute. This was not tested directly.
* Descriptive only: at some middle layers the same POPE probe transfers a bit better (see the per-layer table in `SUMMARY.md`). The reported result uses the layer chosen on POPE dev; picking a layer from that table would be selection on AMBER and is not done.

### 3.3 Cross-model (Qwen2-VL-2B vs SmolVLM-500M, shared 3,000 POPE questions)

Test split is smaller here (444 questions), so Qwen's CIs are wider than in 3.1 (53 Task B positives, 10 Task A positives).

| Model | Task | n_pos | Probe (logreg) | Best baseline | Paired diff [CI], p |
|---|---|---|---|---|---|
| Qwen2-VL-2B | B | 53 | 0.890 [0.831, 0.934] | 0.806 | +0.084 [+0.029, +0.145], p = 0.002 |
| Qwen2-VL-2B | A | 10 | 0.846 [0.715, 0.934] | 0.790 | +0.056 [-0.080, +0.160], p = 0.318 |
| SmolVLM-500M | B | 96 | 0.862 [0.826, 0.898] | 0.837 (vocab entropy) | +0.025 [-0.008, +0.063], p = 0.150 |
| SmolVLM-500M | A | 16 | 0.749 [0.613, 0.865] | 0.807 | -0.059 [-0.173, +0.067], p = 0.370 |

* With the MLP probe SmolVLM Task B is +0.046 [+0.004, +0.094], p = 0.036 (probe 0.876 +/- 0.007 over 3 seeds); Task A is still not better than baselines.
* So the "hidden states beat output signals" result is clear for Qwen2-VL-2B on Task B, but for SmolVLM-500M it is small and only marginally significant with one probe type and not significant with the other. It does not generalise cleanly to the smaller model at this sample size.
* The POPE-trained probe does not transfer to AMBER for SmolVLM either (Task B logreg: 0.616 / 0.446 / 0.553; attribute significantly worse than the baseline, -0.297 [-0.431, -0.167]).

### 3.4 Follow-up: is the transfer failure caused by the error mix? (H5)

Task B only; logreg probe as in the main run; layer and C chosen on POPE dev only; AMBER dev/test labels never used. Full tables: `SUMMARY.md`, section "Follow-up: error-mix test".

**Prediction, written before the results were computed:** if H5 is right, removing the answer-direction shortcut should lift AMBER existence and attribute Task B AUROC clearly above 0.5 (95% CI entirely above 0.5), ideally close to the output baselines, at a modest cost on POPE test. If the variants still score at or below about 0.5 on those subsets, H5 is not supported.

**Verdict rule fixed in advance:** SUPPORTED needs a target-free variant (E1a or E2) with CI entirely above 0.5 and a significantly positive paired difference to the original probe on both existence and attribute. PARTLY SUPPORTED if any variant (E1b included) is significantly better than the original probe on at least one of them. E1b uses AMBER-train error statistics (target-domain information) and can never give "supported".

Variants: **E0** sign diagnostic (no training); **E1a** error-type weights with hallucinated-yes and missed errors equal in the positive class (target-free, headline); **E1b** weights matched to each AMBER subset's train-split error mix (uses target-domain statistics); **E2** two probes, one for answered-yes rows and one for answered-no rows, each Platt-calibrated on its POPE dev rows and routed by the model's answer. Balanced class weighting is replaced, not stacked: the explicit weights make total positive weight equal total negative weight, and with the natural error mix they reproduce `class_weight="balanced"`. The original probe re-fitted here reproduces the main run's AMBER numbers exactly (difference 0.0, both models).

Qwen2-VL-2B, Task B AUROC [95% CI] on the AMBER test split (n_pos 22 / 16 / 38), with the paired difference to the original probe:

| Variant | existence | attribute | relation | POPE test (cost) |
|---|---|---|---|---|
| best output baseline | 0.837 | 0.812 | 0.719 | 0.801 |
| original POPE probe | 0.453 [0.333, 0.571] | 0.352 [0.206, 0.497] | 0.590 [0.505, 0.681] | 0.892 |
| E1a balanced (target-free) | **0.820 [0.746, 0.887]**, +0.367 [+0.259, +0.482] | 0.512 [0.357, 0.661], +0.160 [+0.063, +0.270] | 0.576 [0.492, 0.662], -0.014 n.s. | 0.888 (-0.004 [-0.021, +0.010]) |
| E1b AMBER-matched (target stats) | 0.827 [0.754, 0.894] | 0.516 [0.338, 0.684] | 0.584 [0.494, 0.660] | 0.886 / 0.872 / 0.891 (attribute probe -0.020 [-0.041, -0.002], p = 0.032) |
| E2 answer-conditioned (target-free) | 0.511 [0.404, 0.615], +0.058 n.s. | 0.347 [0.212, 0.485], -0.005 n.s. | **0.710 [0.635, 0.789]**, +0.120 [+0.051, +0.191] | 0.905 (+0.013 n.s.) |
| AMBER-trained ceiling | 0.856 | 0.945 | 0.821 | |

SmolVLM-500M (n_pos 39 / 17 / 51): original 0.616 / 0.446 / 0.553; E1a 0.681 [0.605, 0.748] (+0.065 n.s.) / **0.703 [0.550, 0.848]** (+0.257 [+0.154, +0.369]) / 0.604 (+0.051 n.s.); E1b 0.663 / 0.715 / 0.554; E2 0.576 / 0.340 / **0.682** (relation +0.129 [+0.026, +0.232]); baselines 0.680 / 0.743 / 0.618; POPE-test cost of E1a -0.017 [-0.049, +0.015], of E2 -0.004.

**E0 (Qwen):** the trivial score "model answered no" gets Task B AUROC 0.641 on POPE test but 0.178 [0.084, 0.294] on AMBER attribute and 0.386 on existence (0.682 on relation), so the answer direction alone is a strong POPE shortcut that points the wrong way on two AMBER subsets. Within answered-no rows the original probe is reversed on AMBER existence (0.217 [0.116, 0.352]) and relation (0.349 [0.250, 0.466]).

**Verdict: H5 PARTLY SUPPORTED, for both models (computed by the rule above).**

* Supported part: a simple target-free change (give hallucinated-yes errors as much training weight as missed objects) repairs most of the transfer on Qwen existence (0.453 -> 0.820, indistinguishable from the best baseline, POPE cost -0.004) and on SmolVLM attribute (0.446 -> 0.703, indistinguishable from the baseline). This is what the prediction expected, and the POPE cost is small.
* Not supported part: on Qwen attribute the weighted probe only reaches chance (0.512, CI includes 0.5; still significantly better than the original 0.352 but far below the 0.812 baseline); relation does not move under re-weighting; SmolVLM existence is above 0.5 but its gain over the original is not significant. So the prediction fails for Qwen attribute, and re-weighting alone is not a general fix.
* E1b (target-domain statistics) never beat the target-free E1a, so the benefit comes from breaking the one-sided error mix, not from matching AMBER's mix.
* E2 (answer-conditioned probes) did **not** fix existence or attribute (Qwen 0.511, 0.347) but is the only variant that helps relation, in both models (+0.120 and +0.129, both significant).
* H5 cannot be the whole story: E0 shows the original probe is reversed even *within* the answered-no rows of Qwen existence and relation, i.e. also among errors of one type, so the probe learned more than a "said no" shortcut (for example features that are specific to POPE's images or question style).

**Caveats:** AMBER test positives are very small (Qwen 22 / 16 / 38; SmolVLM 39 / 17 / 51), the intervals are wide, and p-values are uncorrected across many comparisons. The AMBER-train error shares used by E1b come from small subsets. E1b uses target-domain statistics and is reported only as a labelled comparison, never as a headline. Per-layer AMBER numbers in the follow-up tables are descriptive and not used for selection.

## 4. Caveats

* **Few positives.** Task A has 28 test positives on POPE (4 / 8 / 16 per setting) and only 7 to 16 per AMBER subset; AMBER Task B has 16 to 38 (up to 51 for SmolVLM). Read subset-level numbers as indicative, and quote overall numbers with their CIs.
* **The best POPE layer is the final hidden state,** which is what the output layer reads, so a probe there can partly decode the answer distribution. The logit-only probe is the fair control for that.
* **Layer choice is unstable:** dev selection picked layer 28 for Task A, 5-fold CV picked 16, with overlapping test CIs, and it varies across MLP seeds and models. Do not interpret a specific layer.
* **P-values are uncorrected** for the many comparisons in the paired tables.
* **Scope:** two models (Qwen2-VL-2B, SmolVLM-500M), one POPE split, AMBER with 1,000 queries per subset, one extraction run per cache. The ablation runs have POPE results only.
* **Baseline definition:** margin and binary entropy are monotone in each other for a two-way choice, so their AUROCs are identical. Baselines score uncertainty against wrong answers (Tasks A and B); scoring P(yes) against hallucination would be trivially near AUROC 1.

## 5. Open items for you

1. **Git:** your local branch has commits that include `pope_full_features.pt` (802 MB) in history. `.gitignore` does not affect files already tracked and GitHub rejects files over 100 MB, so pushing as is will fail. Fix before pushing (for example rewrite those local commits so the `.pt` is not tracked). Do not commit any `.pt` file. `results/final/` is small (a couple of MB) and fine to commit.
2. **Stray file:** `results/pope_full_features.pt` (untracked, 802 MB, outside `cached_features/`) is a different extraction of the same 9,000 questions (same row order, but values differ: mean absolute difference 0.017, maximum 4.5). It is **not** the cache used here: the one used is `results/cached_features/pope_full_features.pt`, byte-identical to the original Drive download and consistent with `pope_metadata.parquet`. Delete or move the stray file, and make sure the team all uses the same one, or results will differ.
3. **Optional third model:** run `notebooks/04_colab_qwen25vl_3b_pope_amber.ipynb` on Colab (hours per dataset on a free T4), copy the two `qwen25vl-3b` caches in, then run `python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m qwen25vl-3b --probe logreg --out-dir results/final/cross_model3_logreg` and `python src/make_summary.py`.
4. **Optional:** re-run the PCA / no-class-weight / CV ablations now that the AMBER cache is present if you want their transfer tables.
5. **The error-mix explanation was tested** (section 3.4); the result is partial. A natural next step, not run, would be to train the probe on a mixture of POPE and a small AMBER-train sample, or to probe other layers/positions.

## 6. Where things are

| Item | Location |
|---|---|
| Full tables | `results/final/SUMMARY.md` |
| Per-run outputs, commands, configs, logs | `results/final/qwen_logreg/`, `qwen_mlp/`, `qwen_logreg_pca/`, `qwen_logreg_noclassweight/`, `qwen_logreg_cv5/`, `cross_model_logreg/`, `cross_model_mlp/` |
| Evaluation code | `src/probe_transfer_eval.py`, `src/probe_stats.py`, `src/make_summary.py`, `src/followup_errormix.py` |
| Follow-up outputs | `results/final/followup_errormix/qwen2vl-2b/`, `.../smolvlm-500m/` (CSVs, `followup_transfer_taskB.png`, `command.txt`, `run_config.json`, `run.log`) |
| Colab extraction | `notebooks/01_...` (Qwen POPE), `03_...` (any model, POPE or AMBER), `04_...` (Qwen2.5-VL-3B, not yet run) |
| Reproduction commands | `README.md`, section "Reproducing the report results" |
