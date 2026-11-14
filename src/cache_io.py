"""Cache file naming and loading shared by the extraction and evaluation scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEGACY = {("pope", "qwen2vl-2b"): ("pope_full_features.pt", "pope_metadata.parquet", "pope_run_config.json")}


def file_names(dataset: str, model_key: str) -> tuple[str, str, str]:
    s = f"{dataset}_{model_key}"
    return f"{s}_features.pt", f"{s}_metadata.parquet", f"{s}_run_config.json"


def find_cache(dataset: str, model_key: str, cache_dir) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    candidates = [file_names(dataset, model_key)]
    if (dataset, model_key) in LEGACY:
        candidates.append(LEGACY[(dataset, model_key)])
    for feats, meta, _ in candidates:
        f, m = cache_dir / feats, cache_dir / meta
        if not m.exists() and m.with_suffix(".csv").exists():
            m = m.with_suffix(".csv")
        if f.exists() and m.exists():
            return f, m
    raise FileNotFoundError(f"No cache for dataset={dataset!r}, model={model_key!r} in {cache_dir} "
                            f"(looked for {[c[0] for c in candidates]})")


def read_meta(path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_cache(dataset: str, model_key: str, cache_dir, mmap: bool = True) -> dict:
    """Returns dict(features=np.ndarray float16 [N, L, H], meta=DataFrame row-aligned with features, layers, n_entries)."""
    import torch
    f_path, m_path = find_cache(dataset, model_key, cache_dir)
    try:
        blob = torch.load(f_path, map_location="cpu", weights_only=True, mmap=mmap)
    except (TypeError, RuntimeError):
        blob = torch.load(f_path, map_location="cpu", weights_only=True)
    feats, row_idx = blob["features"].numpy(), blob["row_idx"].numpy()
    meta = read_meta(m_path)
    missing = set(row_idx.tolist()) - set(meta["row_idx"].tolist())
    if missing:
        raise ValueError(f"{len(missing)} cached feature rows have no metadata, e.g. {sorted(missing)[:5]}")
    meta = meta.set_index("row_idx", drop=False).loc[row_idx].reset_index(drop=True)
    layers = [int(x) for x in blob.get("layers", range(feats.shape[1]))]
    n_entries = int(blob.get("n_hidden_state_entries", feats.shape[1]))
    return dict(features=feats, meta=derive_columns(meta), layers=layers, n_entries=n_entries,
                dataset=dataset, model_key=model_key)


def derive_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """Add split, gt_yes, pred_yes, correct, error, conf and raw_conf columns."""
    m = meta.copy()
    if "split" not in m.columns or m["split"].isna().any():
        from vlm_datasets import split_of
        m["split"] = [split_of(i) for i in m["image_id"]]
    m["gt_yes"] = (m["gt_answer"] == "yes").astype(int)
    m["pred_yes"] = (m["pred_answer"] == "yes").astype(int)
    m["correct"] = (m["pred_answer"] == m["gt_answer"]).astype(int)
    m["error"] = 1 - m["correct"]
    m["conf"] = np.maximum(m["p_yes_norm"], 1 - m["p_yes_norm"])
    m["raw_conf"] = np.where(m["pred_yes"] == 1, m["p_yes"], m["p_no"])
    return m
