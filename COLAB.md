# Running the POPE feature extraction on Google Colab

**GPU REQUIRED - RUN IN COLAB - NOT TESTED LOCALLY.** The model run lives in `notebooks/01_colab_pope_feature_extraction.ipynb`. Do not run it on a laptop, and do not use `src/feature_extraction.py` for this (it is the old 100-sample script, kept only for reference).

## Steps

1. **Get the notebook into Colab.** Commit and push only the notebooks and this file (never `data/`, `.venv/`, or any full `.pt` cache), then in Colab: *File > Open notebook > GitHub*, choose `notebooks/01_colab_pope_feature_extraction.ipynb`. Or: *File > Upload notebook*.
2. **Runtime.** *Runtime > Change runtime type > Python 3, Hardware accelerator: GPU* (T4 or better; L4 / A100 are faster). The 2B model needs about 6-8 GB of GPU memory at batch size 1.
3. **Config.** Edit only the first code cell. Defaults are a full run (all 9,000 questions), `BATCH_SIZE = 1`, checkpoint every 100 questions. Set `USE_DRIVE = True` to write outputs to Google Drive (recommended; survives disconnects).
4. **Run the cells top to bottom:** Config -> a. Colab setup (checks the GPU, installs packages, clones the repo, optional Drive mount) -> b. Dataset inspection -> c. Token check -> d. Prompt freeze -> e. Model load -> f. Scoring function -> g/h. Run extraction -> Merge -> i. Verification -> optional Download.
5. **Outputs** (in `results/cached_features/` of the cloned repo, or your Drive folder):
   * `pope_full_features.pt` - about 0.8 GB for 9,000 questions (29 hidden-state entries x 1536 dims x float16 per question)
   * `pope_metadata.parquet` - a few MB (CSV if parquet fails)
   * `pope_run_config.json` - model, prompt, token ids, seed, dataset info, date
   * `pope_full_shards/` - 100-question shards, about 9 MB each, used for resuming (same data as the .pt)
6. **Bring them back.** Copy the three files into your local repo's `results/cached_features/`. From Drive: download them from drive.google.com. Without Drive: set `DOWNLOAD_AT_END = True` and run the last cell (browser downloads of 0.8 GB can fail; Drive is more reliable). **Do not commit the `.pt` file.** Note `results/` is not in `.gitignore`.
7. **Verify.** The Verification cell must show: 0 missing rows, 0 duplicates, 0 non-finite values, file size close to the expected size (about 0.80 GB for 9,000 rows) with no `!!! FLAG` line, accuracy and false-yes rate per setting, and `answer_mass` close to 1. Then run `notebooks/02_pope_baseline_visuals.ipynb` locally (CPU).
8. **Disconnects.** Reconnect, re-run all cells from the top with the same config. Finished shards are skipped (by dataset row index); at most the last 100 questions are redone. If you used no Drive, the runtime's disk is wiped on a hard reset, so use Drive for the full run.
9. **Out of memory (CUDA OOM).** *Runtime > Restart session*, keep `BATCH_SIZE = 1`, re-run from the top (it resumes).
10. **T4 note.** T4 GPUs have no native bfloat16. If model loading or inference is very slow or errors, set `DTYPE = "float16"` in the config (this is also what the old 100-sample cache used). Start the run over (new `OUT_DIR`) if you change it after shards exist; the notebook refuses to mix dtypes.
