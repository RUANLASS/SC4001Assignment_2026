# SC4001 VLM Hallucination Investigation

This repository contains the codebase for investigating whether Vision-Language Models (VLMs) possess internal representations in their hidden states that signal when they are hallucinating object descriptions.

The project investigates **Qwen2-VL-2B-Instruct** evaluated on the **POPE (Polling Object Probing Evaluation)** dataset.
Extra models (Qwen2.5-VL-3B, SmolVLM-500M, LLaVA-1.5-7B) and the **AMBER** benchmark (cross-dataset transfer) are covered in
[`docs/EXTRA_MODELS_AMBER.md`](docs/EXTRA_MODELS_AMBER.md).

## Local files and Git

The virtual environment (`.venv/`) and all contents of `data/` are excluded from Git. 
The datasets are local downloads and should not be committed. 
The `data/.gitkeep` placeholder keeps the empty `data/` directory in the repository. 
Run `python src/download_datasets.py` to download or load the POPE dataset under `data/pope/`.

Other project folders, including `notebooks/`, `results/`, and `src/`, are not excluded by these ignore rules.

---

## Getting the cache (needed for probe training)

The hidden-state cache is **not in Git** (the full file is about 800 MB). Download these from the shared folder and put them in `results/cached_features/`:

- `pope_full_features.pt` (about 802 MB): hidden states, shape [9000, 29, 1536], float16, last prompt token, all layers
- `pope_metadata.parquet`: one row per question (predictions, probabilities, entropy, hallucination label, image id, split)
- `pope_run_config.json`: model, prompt, token ids, seed, dtype

**Download link:** TODO - paste the shared Google Drive folder link here.

Notes:
- Use the `split` column in the metadata (train/dev/test, assigned by image). Do not re-split, so results stay comparable across the team.
- Load with `load_pope_cache()` and `get_split()` from `notebooks/02_pope_baseline_visuals.ipynb`.
- Do not commit `.pt` files; they are gitignored. To regenerate, see `COLAB.md` (about 5 hours on a free T4).
- `results/cached_features/pope_subset_features.pt` (the old 100-sample file) is superseded by the full cache.

---

## Reproducing the report results

Everything below is CPU-only and reads the caches in `results/cached_features/` (see "Getting the cache"). If a required `.pt` is missing the
scripts stop with `FileNotFoundError`; they never fall back to mock or synthetic data.

```bash
pip install -r requirements.txt -r requirements-extra.txt
python -m pytest tests                                   # unit tests on small synthetic caches (no real data, no GPU)

# --- baseline figures + table (notebook 02, runs on pope_full_features.pt + pope_metadata.parquet)
jupyter nbconvert --to notebook --execute notebooks/02_pope_baseline_visuals.ipynb --output 02_pope_baseline_visuals.ipynb
#   -> results/figures/*.png, results/figures/pope_baseline_results_table.csv

# --- Qwen2-VL-2B probes on POPE (all 9,000 questions; every run writes command.txt, run_config.json, run.log next to its tables)
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --recalibrate           --out-dir results/final/qwen_logreg
python src/probe_transfer_eval.py --models qwen2vl-2b --probe mlp --seeds 0 1 2             --out-dir results/final/qwen_mlp
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --pca 64 128 256       --out-dir results/final/qwen_logreg_pca
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --no-class-weight      --out-dir results/final/qwen_logreg_noclassweight
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --layer-cv 5 --recalibrate --out-dir results/final/qwen_logreg_cv5

# --- summary (every number is read from the CSVs above)
python src/make_summary.py                               # -> results/final/SUMMARY.md
```

| Output | Produced by |
|---|---|
| Layer-wise AUROC curves (`layerwise_auroc_task{A,B}.png`, CV band in `qwen_logreg_cv5`), `layerwise_pope.csv` | `probe_transfer_eval.py` |
| Probe vs baselines on POPE test, with CIs (`pope_test_summary.csv`), MLP seed mean ± std (`pope_test_seed_summary.csv`) | `probe_transfer_eval.py` |
| Paired probe-vs-baseline tests (`paired_tests.csv`) | `probe_transfer_eval.py` (`probe_stats.paired_auc_diff`) |
| Calibration (`calibration.csv`, `probe_reliability_task{A,B}.png`) | `probe_transfer_eval.py` (`--recalibrate` adds Platt scaling) |
| By-setting table (`pope_by_setting.csv`), AURC / AUACC (`aurc.csv`), `accuracy_vs_coverage.png` | `probe_transfer_eval.py` |
| Hidden-only vs logit-only vs combined probe | every run (methods `probe (...)`, `probe (logit-only logreg)`, `probe (hidden+logits, ...)`) |

**AMBER and cross-model runs** (need `amber_qwen2vl-2b_features.pt`, `pope_smolvlm-500m_features.pt`, `amber_smolvlm-500m_features.pt` in
`results/cached_features/`; the Qwen runs above pick up the AMBER cache automatically and then also write `transfer_amber.csv`):

```bash
python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m --probe logreg --out-dir results/final/cross_model_logreg   # restricts both models to their shared 3,000 questions / 3,000 AMBER queries
python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m --probe mlp --seeds 0 1 2 --out-dir results/final/cross_model_mlp
python src/make_summary.py
```

The PCA / no-class-weight / CV ablation runs above were done before the AMBER cache was available, so they contain POPE results only
(re-run them with AMBER present if you want their transfer tables).

**Follow-up: is the transfer failure caused by the error mix? (hypothesis H5; CPU only, Task B)** Needs the four POPE/AMBER caches for
the two models. The prediction is stated in the script's docstring and printed first in each `run.log`, before any result is computed:

```bash
python src/followup_errormix.py --models qwen2vl-2b smolvlm-500m --out-dir results/final/followup_errormix   # --seed 0 --n-boot 500 --n-boot-paired 1000 are the defaults
python src/make_summary.py                                                                                  # adds the "Follow-up: error-mix test" section to results/final/SUMMARY.md
```

Per model it writes `e0_sign_diagnostic.csv`, `e1_reweight.csv` (balanced E1a and AMBER-matched E1b, the latter uses target-domain statistics),
`e2_answer_conditioned.csv`, `followup_summary.csv`, `layerwise_e1a_e2.csv`, `followup_transfer_taskB.png`, `command.txt`, `run_config.json`, `run.log`
to `results/final/followup_errormix/<model>/`. Layer and C are chosen on POPE dev only; AMBER dev/test labels are never used for any choice.

Optional third model (GPU, Colab only, not yet run): `notebooks/04_colab_qwen25vl_3b_pope_amber.ipynb`, then add `qwen25vl-3b` to `--models`.
Options: `--class-weight/--no-class-weight` (default on), `--pca 64 128 256`, `--seeds`, `--layer-cv K`, `--recalibrate`, `--n-boot`, `--n-boot-paired`.

---

## File Structure
```text
SC4001/
├── .gitignore
├── .venv/                          # Local Python environment; excluded from Git
├── COLAB.md                        # step-by-step Colab guide for notebook 01
├── data/                           # Local datasets; contents excluded from Git
│   └── .gitkeep                    # Keeps this directory in the repository
├── docs/
│   └── EXTRA_MODELS_AMBER.md       # Extra models + AMBER: protocol, how to run, verification status
├── notebooks/
│   ├── 00_audit_existing_cache.ipynb            # audit of the old 100-sample cache + POPE structure (CPU)
│   ├── 01_colab_pope_feature_extraction.ipynb   # Qwen2-VL-2B on POPE (GPU, Colab)
│   ├── 02_pope_baseline_visuals.ipynb           # baselines + visuals on the POPE cache (CPU)
│   ├── 03_colab_extra_models_amber.ipynb        # any registered model on POPE / AMBER (GPU, Colab)
│   └── 04_colab_qwen25vl_3b_pope_amber.ipynb    # Qwen2.5-VL-3B on POPE then AMBER, smoke-test gated (GPU, Colab; not yet run)
├── results/
│   ├── cached_features/            # hidden-state caches (*.pt gitignored) + small metadata / run configs
│   ├── figures/                    # notebook 02 baseline figures and results table (real data)
│   └── final/                      # outputs of src/probe_transfer_eval.py, one sub-folder per run, plus SUMMARY.md
├── src/
│   ├── download_datasets.py        # POPE dataset downloading and disk caching
│   ├── setupverify.py              # Environment & GPU verification
│   ├── vlm_registry.py             # model list, prompt, yes/no scoring
│   ├── vlm_datasets.py             # POPE + AMBER adapters, image-level splits
│   ├── cache_io.py                 # cache file naming / loading
│   ├── extract_features_multi.py   # GPU extraction for any registered model, POPE or AMBER
│   ├── probe_transfer_eval.py      # CPU: layer sweep, probes, baselines, transfer, paired tests, calibration, AURC
│   ├── probe_stats.py              # pure statistics: paired image bootstrap, ECE/Brier, Platt, accuracy-coverage areas
│   ├── followup_errormix.py        # CPU: error-mix follow-up (E0 sign diagnostic, E1 re-weighting, E2 answer-conditioned probes)
│   └── old/                        # superseded legacy script (see src/old/README.md); the mock-data scripts were deleted
├── tests/                          # pytest, CPU only (python -m pytest tests)
├── requirements.txt                # Project dependencies
├── requirements-extra.txt          # Extra dependencies for extraction (Colab), evaluation and tests
└── README.md
```
