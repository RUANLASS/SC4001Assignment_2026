"""Dataset adapters (POPE, AMBER) for the multi-model extraction.

Conventions copied EXACTLY from notebook 01 so that rows and splits line up with the team's Qwen2-VL-2B cache:
  * ``row_idx``  = position in the dataset (POPE: row in the HF 'test' split)
  * processing order = ``np.random.default_rng(seed).shuffle(arange(n))``; a subset of size N is the FIRST N of that order,
    so a smaller run on any model is a subset of the same questions
  * split = md5('{seed}-{image_id}') -> train / dev / test = 70 / 15 / 15 %, decided BY IMAGE (no image in two splits)

Heavy imports (datasets, PIL) are done lazily so the pure logic can be unit-tested without them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SPLIT_FRACS = (0.70, 0.15, 0.15)


def split_of(image_id, seed: int = 0, fracs=SPLIT_FRACS) -> str:
    u = int(hashlib.md5(f"{seed}-{image_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < fracs[0] else ("dev" if u < fracs[0] + fracs[1] else "test")


def seeded_order(row_ids, seed: int = 0) -> list[int]:
    order = np.asarray(row_ids).copy()
    np.random.default_rng(seed).shuffle(order)
    return [int(r) for r in order]


@dataclass
class DatasetBundle:
    name: str                              # "pope" | "amber"
    meta: pd.DataFrame                     # row_idx, id, question_id, image_id, setting, question, gt_answer
    get_image: Callable[[int], object]     # row_idx -> PIL.Image
    selected: list[int]                    # row_idx in processing order
    selection_desc: dict                   # what was selected (goes into the run config)


REQUIRED_COLS = ["row_idx", "id", "question_id", "image_id", "setting", "question", "gt_answer"]


# --------------------------------------------------------------------------------------------------------------------
# POPE
# --------------------------------------------------------------------------------------------------------------------
def load_pope(local_path: str = "data/pope", hf_name: str = "lmms-lab-encoder/POPE") -> DatasetBundle:
    from datasets import load_dataset, load_from_disk
    p = Path(local_path)
    if p.exists() and any(p.iterdir()):
        ds = load_from_disk(str(p))
    else:
        ds = load_dataset(hf_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(p))
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    split_ds = ds[split_name]
    meta = split_ds.remove_columns("image").to_pandas()
    meta["row_idx"] = np.arange(len(meta))
    meta = meta.rename(columns={"image_source": "image_id", "category": "setting", "answer": "gt_answer"})
    meta["gt_answer"] = meta["gt_answer"].astype(str).str.strip().str.lower()
    if not set(meta["gt_answer"]) <= {"yes", "no"}:
        raise ValueError(f"unexpected POPE answers: {set(meta['gt_answer'])}")
    meta = meta[REQUIRED_COLS]
    return DatasetBundle("pope", meta, lambda r: split_ds[int(r)]["image"], [], {})


def select_pope(bundle: DatasetBundle, subset_size: int | None, seed: int = 0) -> DatasetBundle:
    order = seeded_order(bundle.meta["row_idx"].to_numpy(), seed)
    if subset_size is not None:
        order = order[:subset_size]
    bundle.selected = order
    bundle.selection_desc = dict(dataset="pope", subset_size=subset_size, seed=seed, order="seeded shuffle of all rows, first N")
    return bundle


# --------------------------------------------------------------------------------------------------------------------
# AMBER (discriminative part only: yes/no queries)
# --------------------------------------------------------------------------------------------------------------------
# Verified arithmetic: discriminative ids run 1005..15220 (14,216 queries = 4,924 + 7,628 + 1,664, AMBER paper Table 8);
# the first relation query ("Is there direct contact between ...") has id 13557 = 1005 + 4924 + 7628.
AMBER_RANGES = {"existence": (1005, 5928), "attribute": (5929, 13556), "relation": (13557, 15220)}
AMBER_EXPECTED = {"existence": 4924, "attribute": 7628, "relation": 1664}


def amber_subset_of(qid: int) -> str | None:
    for name, (lo, hi) in AMBER_RANGES.items():
        if lo <= qid <= hi:
            return name
    return None


def _iter_annotations(ann):
    """annotations.json may be a list of dicts (with 'id') or a dict keyed by id."""
    if isinstance(ann, dict):
        for k, v in ann.items():
            if isinstance(v, dict):
                yield int(v.get("id", k)), v
    else:
        for v in ann:
            if isinstance(v, dict) and "id" in v:
                yield int(v["id"]), v


def build_amber_meta(queries: list, annotations) -> pd.DataFrame:
    """queries: list of {'id','image','query'}; annotations: contents of annotations.json (must give 'truth' per id)."""
    truth = {}
    for qid, entry in _iter_annotations(annotations):
        if "truth" in entry:
            truth[qid] = str(entry["truth"]).strip().lower()
    if not truth:
        first = next(iter(_iter_annotations(annotations)), (None, {}))[1]
        raise ValueError("annotations.json has no 'truth' field per entry. Keys of the first entry: "
                         f"{sorted(first.keys())}. Run `--inspect` and adapt build_amber_meta.")
    rows = []
    for q in queries:
        qid = int(q["id"])
        sub = amber_subset_of(qid)
        if sub is None:
            continue                      # generative queries (ids 1..1004) are not yes/no
        if qid not in truth:
            raise KeyError(f"query id {qid} has no ground truth in annotations.json")
        rows.append(dict(id=qid, question_id=qid, image_id=str(q["image"]), setting=sub,
                         question=str(q["query"]), gt_answer=truth[qid]))
    meta = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    meta.insert(0, "row_idx", np.arange(len(meta)))
    bad = set(meta["gt_answer"]) - {"yes", "no"}
    if bad:
        raise ValueError(f"unexpected AMBER truth values: {bad}")
    return meta[REQUIRED_COLS]


def select_amber(bundle: DatasetBundle, subsets, per_subset: int | None, seed: int = 0) -> DatasetBundle:
    """Stratified: up to ``per_subset`` random queries from each requested subset, then one seeded shuffle."""
    rng = np.random.default_rng(seed)
    chosen = []
    for sub in subsets:
        ids = bundle.meta.loc[bundle.meta["setting"] == sub, "row_idx"].to_numpy()
        if len(ids) == 0:
            raise ValueError(f"AMBER subset '{sub}' is empty")
        take = len(ids) if per_subset is None else min(per_subset, len(ids))
        chosen.extend(rng.choice(ids, size=take, replace=False).tolist())
    order = np.asarray(chosen)
    rng.shuffle(order)
    bundle.selected = [int(r) for r in order]
    bundle.selection_desc = dict(dataset="amber", subsets=list(subsets), per_subset=per_subset, seed=seed,
                                 order="random sample per subset, then seeded shuffle")
    return bundle


def amber_report(meta: pd.DataFrame) -> str:
    """Human-readable schema/label summary; printed by --inspect and at the start of every AMBER run."""
    lines = [f"AMBER discriminative queries: {len(meta)}"]
    for sub, n_exp in AMBER_EXPECTED.items():
        d = meta[meta.setting == sub]
        lines.append(f"  {sub:10s} n={len(d):5d} (paper: {n_exp})  truth: {d.gt_answer.value_counts().to_dict()}")
    lines.append(f"  unique images: {meta.image_id.nunique()}")
    return "\n".join(lines)


def load_amber(repo_dir: str = "data/amber_repo", images_dir: str | None = "data/amber_images",
               hf_images: str | None = None) -> DatasetBundle:
    """Metadata from a clone of github.com/junyangwang0410/AMBER (data/query + data/annotations.json).

    Pixels from ``images_dir`` (the AMBER_*.jpg files from the Google-Drive zip, searched recursively), or, if
    ``hf_images`` is given (e.g. 'visual-preference/AMBER'), from that Hugging Face mirror matched by query id.
    """
    repo = Path(repo_dir)
    q_path = repo / "data" / "query" / "query_discriminative.json"
    a_path = repo / "data" / "annotations.json"
    for p in (q_path, a_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. `git clone https://github.com/junyangwang0410/AMBER {repo}` first.")
    queries = json.loads(q_path.read_text())
    meta = build_amber_meta(queries, json.loads(a_path.read_text()))

    if hf_images:
        from datasets import load_dataset
        ds = load_dataset(hf_images, split="query_discriminative")
        pos = {int(i): k for k, i in enumerate(ds["id"])}
        qid_of_row = dict(zip(meta["row_idx"], meta["id"]))

        def get_image(r):
            return ds[pos[int(qid_of_row[int(r)])]]["image"]
    else:
        from PIL import Image
        root = Path(images_dir)
        if not root.exists():
            raise FileNotFoundError(f"AMBER image folder {root} not found (see docs/EXTRA_MODELS_AMBER.md).")
        index = {p.name: p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        missing = sorted(set(meta["image_id"]) - set(index))
        if missing:
            raise FileNotFoundError(f"{len(missing)} AMBER images missing under {root}, e.g. {missing[:3]}")
        img_of_row = dict(zip(meta["row_idx"], meta["image_id"]))

        def get_image(r):
            return Image.open(index[img_of_row[int(r)]])

    return DatasetBundle("amber", meta, get_image, [], {})
