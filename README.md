# Do Vision-Language Models Know When They Hallucinate?

SC4001 Neural Networks and Deep Learning, Group Project (Topic E), Group 42:
Bui Khanh Hung, Sarin Lakshya Anuj, Low Jun Keat.

We test whether a linear probe on the hidden states of a small VLM (Qwen2-VL-2B-Instruct, with SmolVLM-500M-Instruct as a second model)
detects wrong yes/no answers better than output-level signals (token probability, entropy), where in the network the signal lives, and
whether a probe trained on POPE transfers to AMBER. The report (PDF) describes the experiments; this file explains how to run the code and
reproduce its tables and figures.

## Repository layout

| Path | Content |
|---|---|
| `src/probe_transfer_eval.py` | CPU evaluation: output baselines, per-layer probes, dev-only selection, POPE to AMBER transfer, paired tests, calibration, AURC |
| `src/probe_stats.py` | Paired image-level bootstrap, ECE/Brier, Platt scaling, risk-coverage areas |
| `src/followup_errormix.py` | CPU follow-up experiment on the error mix (E0 sign diagnostic, E1 re-weighting, E2 answer-conditioned probes) |
| `src/make_summary.py` | Collects every table into `results/final/SUMMARY.md` |
| `src/extract_features_multi.py`, `vlm_registry.py`, `vlm_datasets.py`, `cache_io.py`, `download_datasets.py` | Feature extraction (GPU, run from the notebooks), model/dataset adapters, cache loading |
| `notebooks/01_colab_pope_feature_extraction.ipynb` | Qwen2-VL-2B on all 9,000 POPE questions (GPU, Colab) |
| `notebooks/03_colab_extra_models_amber.ipynb` | Any registered model on POPE or AMBER (GPU, Colab) |
| `notebooks/04_colab_qwen25vl_3b_pope_amber.ipynb` | Qwen2.5-VL-3B, prepared but **not run** (no result in the report depends on it) |
| `notebooks/02_pope_baseline_visuals.ipynb` | CPU: output-baseline figures and table (report Section 2, Table C1) |
| `tests/` | Unit tests on small synthetic caches (no real data, no model) |
| `results/cached_features/` | Metadata and run configs (in the repo); the large `*_features.pt` files go here too (see below) |
| `results/final/`, `results/figures/` | All reported tables and figures; each run folder has `command.txt`, `run_config.json`, `run.log` |

## Setup

Python 3.10 or newer (tested with 3.11). All evaluation runs on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-extra.txt
pip install jupyter            # only needed to run the notebooks
```

## Data preparation: the feature caches

The analysis reads cached hidden states (last prompt token, every layer, float16); it never loads a model. Each cache is a pair
`{dataset}_{model}_features.pt` + `_metadata.parquet` (+ `_run_config.json`). The metadata and configs are in the repo. The four `.pt` files
are too large for Git or the ZIP and must be placed in `results/cached_features/` (the Qwen POPE cache is named `pope_full_features.pt`):

| File | Size | Content |
|---|---|---|
| `pope_full_features.pt` | 802 MB | Qwen2-VL-2B, POPE, 9,000 questions x 29 layers x 1536 |
| `amber_qwen2vl-2b_features.pt` | 267 MB | Qwen2-VL-2B, AMBER, 3,000 queries |
| `pope_smolvlm-500m_features.pt` | 190 MB | SmolVLM-500M, POPE, 3,000 questions x 33 x 960 |
| `amber_smolvlm-500m_features.pt` | 190 MB | SmolVLM-500M, AMBER, 3,000 queries |

**Download the four `.pt` files from the shared Google Drive folder:** <https://drive.google.com/drive/folders/1zQb8y5FydDJVZMvElPfZz5p7U07MFZKx?usp=sharing>
and copy them into `results/cached_features/` (do not rename them; do not commit them).

Alternatively, if the link is unavailable, regenerate them on a free Colab T4 (GPU, resumable, checkpoints every
100 questions): open the notebook in Colab (*Runtime > Change runtime type > GPU*), set `USE_DRIVE = True`, run it top to bottom (its first cell
has the full guide) and copy the outputs from Drive into `results/cached_features/`. Notebook 01 produces `pope_full_features.pt` (about 5 hours).
Notebook 03 produces the other three: set `MODEL_KEY` / `DATASET` to `qwen2vl-2b` + `amber`, `smolvlm-500m` + `pope`, `smolvlm-500m` + `amber`
(keep `POPE_SUBSET_SIZE = 3000`, `AMBER_PER_SUBSET = 1000`) and run its smoke-test cell first. POPE (`lmms-lab-encoder/POPE`) and AMBER
(github.com/junyangwang0410/AMBER) are downloaded by the notebooks. Keep the seed at 0: it fixes the question order and the image-level
70/15/15 train/dev/test split (an MD5 hash of the image id), which every result relies on.

## Reproducing the results

Run from the repository root. Approximate times are for an 8-core laptop.

```bash
# Baseline figures and table (report Section 2); writes results/figures/
jupyter nbconvert --to notebook --execute notebooks/02_pope_baseline_visuals.ipynb --output-dir /tmp/nb_out     # a few minutes

# Main probes, Qwen2-VL-2B on POPE (+ AMBER transfer, since the AMBER cache is present)
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --recalibrate        --out-dir results/final/qwen_logreg      # ~10-20 min
python src/probe_transfer_eval.py --models qwen2vl-2b --probe mlp --seeds 0 1 2           --out-dir results/final/qwen_mlp         # ~1 h
# Ablations (report Section 3.2)
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --pca 64 128 256     --out-dir results/final/qwen_logreg_pca
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --no-class-weight    --out-dir results/final/qwen_logreg_noclassweight
python src/probe_transfer_eval.py --models qwen2vl-2b --probe logreg --layer-cv 5 --recalibrate --out-dir results/final/qwen_logreg_cv5
# Smaller model, on the 3,000 questions both models share (report Section 6)
python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m --probe logreg          --out-dir results/final/cross_model_logreg
python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m --probe mlp --seeds 0 1 2 --out-dir results/final/cross_model_mlp
# Follow-up on the error mix (report Section 5.2); the prediction is printed first in each run.log
python src/followup_errormix.py --models qwen2vl-2b smolvlm-500m --out-dir results/final/followup_errormix                       # ~1 h per model

python src/make_summary.py                    # collects every table into results/final/SUMMARY.md (seconds)
```

Quick check that the pipeline works (a few minutes): add `--layer-stride 7 --n-boot 20 --n-boot-paired 20 --out-dir /tmp/smoke` to the first command.
Selection of layer, regularisation, PCA size and class weighting uses dev (or CV on train+dev) only; the test split is used for the reported numbers.
The reruns are deterministic: re-running the logistic-regression and MLP commands reproduced the earlier AUROCs exactly.

| Report | Where to look |
|---|---|
| Sec. 2, Table C1 (output baselines, POPE settings) | `results/figures/`, `results/final/qwen_logreg/pope_by_setting.csv` |
| Sec. 3, Tables 2-4 (probes, ablations, calibration, AURC) | `qwen_logreg/`, `qwen_mlp/`, ablation folders: `pope_test_summary.csv`, `paired_tests.csv`, `calibration.csv`, `aurc.csv` |
| Sec. 4, Fig. 1 and Appendix B (layer-wise) | `qwen_logreg/layerwise_pope.csv`, `layerwise_auroc_taskB.png` |
| Sec. 5, Tables 5, C2 (transfer to AMBER) | `qwen_logreg/transfer_amber.csv`, `qwen_mlp/`, `cross_model_*/transfer_amber.csv` |
| Sec. 5.2, Tables 6, C3 (error-mix follow-up) | `followup_errormix/<model>/followup_summary.csv`, `e0_*.csv`, `followup_transfer_taskB.png` |
| Sec. 6, Tables 7, C4 (smaller model) | `cross_model_logreg/`, `cross_model_mlp/` |

## Tests

```bash
python -m pytest tests          # 82 passed, 1 skipped, about 5 minutes; synthetic caches only
```

The skipped test checks the AMBER annotation counts and runs only if `AMBER_REPO` points to a clone of the AMBER repository.

## Existing code, models and our contribution

Used as is: PyTorch, Hugging Face `transformers`/`datasets`, scikit-learn, the pretrained Qwen2-VL-2B-Instruct and SmolVLM-500M-Instruct models, and
the POPE and AMBER benchmarks. Our contribution: the frozen answer-scoring and feature-extraction protocol, the image-level splits, the evaluation
pipeline (per-layer probes with dev-only selection, logit-only and combined controls, paired image-level bootstrap, calibration, selective
prediction, AMBER transfer), the error-mix follow-up, and the unit tests. See the AI Use Statement in the report for how generative AI was used.
