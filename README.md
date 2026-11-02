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

## File Structure
```text
SC4001/
├── .gitignore
├── .venv/                          # Local Python environment; excluded from Git
├── data/                           # Local datasets; contents excluded from Git
│   └── .gitkeep                    # Keeps this directory in the repository
├── docs/
│   └── EXTRA_MODELS_AMBER.md       # Extra models + AMBER: protocol, how to run, verification status
├── notebooks/
│   ├── 00_audit_existing_cache.ipynb
│   ├── 01_colab_pope_feature_extraction.ipynb   # Qwen2-VL-2B on POPE (GPU)
│   ├── 02_pope_baseline_visuals.ipynb           # baselines + visuals on the POPE cache (CPU)
│   └── 03_colab_extra_models_amber.ipynb        # extra models and AMBER (GPU)
├── results/                        # figures, tables, small outputs; *.pt caches are gitignored
│   └── cached_features/            # Stored hidden states across layers
├── src/
│   ├── download_datasets.py        # POPE dataset downloading and disk caching
│   ├── feature_extraction.py       # OLD 100-sample script (reference only)
│   ├── setupverify.py              # Environment & GPU verification
│   ├── evaluation_baseline.py, abstention_curves.py, train_probe_regression.py   # baselines / probes on the Qwen2-VL cache
│   ├── vlm_registry.py             # model list, prompt, yes/no scoring
│   ├── vlm_datasets.py             # POPE + AMBER adapters, image-level splits
│   ├── cache_io.py                 # cache file naming / loading
│   ├── extract_features_multi.py   # GPU extraction for any registered model, POPE or AMBER
│   └── probe_transfer_eval.py      # CPU: layer-wise probes + POPE -> AMBER transfer
├── tests/                          # pytest, CPU only (python -m pytest tests)
├── requirements.txt                # Project dependencies
├── requirements-extra.txt          # Extra dependencies for src/extract_features_multi.py (Colab) and the evaluation script
└── README.md
```
