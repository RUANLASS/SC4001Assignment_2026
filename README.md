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