# SC4001 VLM Hallucination Investigation

This repository contains the codebase for investigating whether Vision-Language Models (VLMs) possess internal representations in their hidden states that signal when they are hallucinating object descriptions.

The project investigates **Qwen2-VL-2B-Instruct** evaluated on the **POPE (Polling Object Probing Evaluation)** dataset.

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
├── .venv/                # Local Python environment; excluded from Git
├── data/                 # Local datasets; contents excluded from Git
│   └── .gitkeep          # Keeps this directory in the repository
├── notebooks/            # Exploratory data analysis & visualization
├── results/              # Cached tensors (.pt) & probe models
│   └── cached_features/  # Stored hidden states across layers
├── src/                  # Python source scripts
│   ├── download_datasets.py    # POPE dataset downloading and disk caching
│   ├── feature_extraction.py   # VLM feature extraction & hidden state caching
│   └── setupverify.py           # Environment & GPU verification
├── requirements.txt      # Project dependencies
└── READ.ME               
```