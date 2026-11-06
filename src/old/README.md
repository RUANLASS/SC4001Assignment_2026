# src/old - superseded, not used for any reported result

* `feature_extraction.py`: the original 100-sample Qwen2-VL-2B script. It saved only hidden states (no predictions, labels or
  metadata). It produced `pope_subset_features.pt`, which is superseded by the full POPE cache from
  `notebooks/01_colab_pope_feature_extraction.ipynb`. Kept for reference only.

The earlier `train_probe_regression.py`, `evaluation_baseline.py` and `abstention_curves.py` were deleted: they ran on randomly
generated mock data and their figures were fake. The real evaluation pipeline is `src/probe_transfer_eval.py`.
