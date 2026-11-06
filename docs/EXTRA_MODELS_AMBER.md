# Extra models and the AMBER dataset

Scope of this part of the project: **does the hallucination signal in hidden states (and the POPE-trained probe) hold up
for other VLMs and on a different benchmark?** It adds more models and AMBER on top of the team's Qwen2-VL-2B / POPE run
(notebook 01), using the *same* protocol so every number is comparable.

| Piece | File |
|---|---|
| Model list + yes/no scoring | `src/vlm_registry.py` |
| POPE / AMBER adapters, image-level splits | `src/vlm_datasets.py` |
| Cache file naming + loading | `src/cache_io.py` |
| GPU extraction (resumable) | `src/extract_features_multi.py` |
| CPU evaluation: baselines, per-layer probes, POPE -> AMBER transfer | `src/probe_transfer_eval.py` |
| Colab driver | `notebooks/03_colab_extra_models_amber.ipynb` |
| Unit tests (CPU, no GPU/model needed) | `tests/` |

## Protocol (identical for every model; same as notebook 01)

* Prompt: `"{question} Answer yes or no."` in the model's chat format, one image, generation prompt on.
* Hidden state: the **last prompt token**, every transformer layer (+ embeddings), stored as float16, shape `[N, L, H]`.
* Answer: forced choice. `P(yes)` = summed probability of the yes token ids, `P(no)` likewise; predicted yes iff `P(yes) > P(no)`.
* Label: `hallucination = 1` iff predicted yes and ground truth no. `error = 1` iff the answer is wrong.
* Splits: `md5("0-{image_id}")` -> 70 / 15 / 15 train / dev / test, decided **by image** (no image in two splits). Never re-split.
* POPE questions: first N rows of the seeded shuffle, so a 3,000-question run is a subset of the same questions for every model.
* Layer **and** regularisation of every probe are chosen on dev only; test is touched once. AUROC CIs resample images.

Two tasks (same as notebook 02): **A** hallucinated-yes vs correct-yes among "yes" answers; **B** wrong vs right over all questions.

## Models (`python src/extract_features_multi.py --list-models`)

| key | Hugging Face id | notes |
|---|---|---|
| `qwen2vl-2b` | Qwen/Qwen2-VL-2B-Instruct | team reference model; use it to extract **AMBER** with identical settings |
| `qwen25vl-3b` | Qwen/Qwen2.5-VL-3B-Instruct | same family, newer and larger; bfloat16 |
| `smolvlm-500m` | HuggingFaceTB/SmolVLM-500M-Instruct | different architecture, cheapest; image splitting off |
| `llava15-7b` | llava-hf/llava-1.5-7b-hf | the classic POPE model; 4-bit quantised for a 15 GB T4 (so hidden states are those of the quantised model) |

To add a model: add a `ModelSpec` to `REGISTRY` in `src/vlm_registry.py`, run `--smoke 20`, and only start the full run if
`answer_mass` (P(yes)+P(no)) is close to 1.

## AMBER (discriminative part)

Metadata and ground truth come from <https://github.com/junyangwang0410/AMBER> (`data/query/query_discriminative.json`,
`data/annotations.json`); images from the Google-Drive zip in its README (file id is hard-coded in notebook 03), or from
the Hugging Face mirror `visual-preference/AMBER` (`--amber-hf-images`). Checked against the real files:

| subset | queries | truth yes | truth no |
|---|---|---|---|
| existence | 4,924 | 2,462 | 2,462 |
| attribute | 7,628 | 1,352 | 6,276 |
| relation | 1,664 | 975 | 689 |

14,216 queries over 1,004 images. Because every subset has both classes, Task A and Task B are both defined on each subset.
The default selection is 1,000 random queries per subset (3,000 total), then one seeded shuffle.

## How to reproduce

```bash
pip install -r requirements.txt -r requirements-extra.txt
python -m pytest tests                                   # CPU, ~15 s; set AMBER_REPO=<clone> to also check the real AMBER counts

# 1) GPU (Colab T4): open notebooks/03_colab_extra_models_amber.ipynb, set MODEL_KEY / DATASET, run top to bottom.
#    Run the smoke-test cell first. Order: qwen2vl-2b+amber (required for transfer), smolvlm-500m+pope/amber,
#    qwen25vl-3b+pope/amber, then llava15-7b if time allows. Same thing from a shell:
python src/extract_features_multi.py --model smolvlm-500m --dataset pope --subset-size 3000 --smoke 20
python src/extract_features_multi.py --model smolvlm-500m --dataset pope --subset-size 3000
python src/extract_features_multi.py --model qwen2vl-2b  --dataset amber --per-subset 1000

# 2) CPU: put the *_features.pt / _metadata.parquet files in results/cached_features/, then
python src/probe_transfer_eval.py --models qwen2vl-2b smolvlm-500m qwen25vl-3b --out-dir results/final/cross_model_logreg
```

Outputs of step 2 (small; commit them): `layerwise_pope.csv`, `pope_test_summary.csv`, `pope_test_seed_summary.csv`, `transfer_amber.csv`,
`paired_tests.csv`, `calibration.csv`, `pope_by_setting.csv`, `aurc.csv`, `layerwise_auroc_task{A,B}.png`, `transfer_task{A,B}.png`,
`accuracy_vs_coverage.png`, `probe_reliability_task{A,B}.png`, `command.txt`, `run_config.json`. See the README for every option.

Feature files are about `N x layers x hidden x 2 bytes` (e.g. 3,000 questions x 33 layers x 960 dims is roughly 0.2 GB for
SmolVLM-500M). **Never commit `.pt` files** (gitignored); share them through Drive.

## Status of verification (what was actually run)

**CPU, real data (outputs in `results/final/`, summarised in `results/final/SUMMARY.md`, generated from the CSVs):**
* Qwen2-VL-2B on POPE (all 9,000 questions): layer sweep, probe selection on dev (and by 5-fold image-grouped CV), baselines, paired
  image-bootstrap tests, calibration, per-setting breakdown, AURC, PCA / class-weight ablations, MLP over 3 seeds.
* POPE -> AMBER transfer and AMBER -> AMBER ceilings for Qwen2-VL-2B (logreg and MLP), on the 3,000 AMBER queries.
* Cross-model comparison Qwen2-VL-2B vs SmolVLM-500M on the 3,000 shared POPE questions and 3,000 shared AMBER queries (logreg and MLP).
* Cache integrity checks (shapes match the run configs, row ids match the metadata, `split` equals the md5 rule, no image in two splits).
  Re-running the Qwen POPE runs after the AMBER cache was added reproduced the earlier AUROCs exactly (difference 0.0).

**GPU, Colab T4 (done by teammates, per the committed `*_run_config.json`, all `status: complete`, no flags):** Qwen2-VL-2B on POPE (9,000) and AMBER (3,000);
SmolVLM-500M on POPE (3,000) and AMBER (3,000).

**Not run / not verified:**
* Qwen2.5-VL-3B and LLaVA-1.5-7B: never run. `notebooks/04_colab_qwen25vl_3b_pope_amber.ipynb` is ready (smoke test with an automatic `answer_mass` gate,
  then POPE and AMBER) but has not been run on a GPU. No cross-model result includes a third model.
* The Colab extraction code itself was not run by the author of the evaluation code; only its outputs were used.
* Unit tests cover pure logic, probe selection, leakage guards (selection never reads test labels or rows), PCA/CV/seed options, the paired bootstrap,
  calibration metrics, AURC and the summary generator on synthetic caches; they do not load any real model.

## Contribution / AI-use notes for the report

* Existing code and models to acknowledge: Hugging Face `transformers`, the four pretrained VLMs above, POPE, AMBER, scikit-learn.
* Our contribution here: a model-agnostic extraction pipeline with one frozen protocol, image-level splits shared with the
  Qwen2-VL-2B run, a per-layer probe evaluation that selects layer/regularisation on dev only, and the POPE -> AMBER transfer test
  with in-domain (AMBER -> AMBER) ceiling and output-level baselines on the same rows.
* If generative AI helped write or debug this code, say so in the AI Use Statement and describe how it was tested.
