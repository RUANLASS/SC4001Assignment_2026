#!/usr/bin/env python
"""Extract last-prompt-token hidden states + yes/no scores from a VLM on POPE or AMBER.

Generalises notebook 01 (Qwen2-VL-2B on POPE) to more models and to AMBER, with the SAME protocol and the same output
columns, so caches from different models/datasets can be probed and compared with one evaluation script.

    python src/extract_features_multi.py --list-models
    python src/extract_features_multi.py --dataset amber --inspect                       # check AMBER files, no GPU
    python src/extract_features_multi.py --model smolvlm-500m --dataset pope --smoke 20   # 20 questions, prints diagnostics
    python src/extract_features_multi.py --model smolvlm-500m --dataset pope --subset-size 3000
    python src/extract_features_multi.py --model qwen2vl-2b  --dataset amber --per-subset 1000

GPU REQUIRED for the model run (use notebooks/03_colab_extra_models_amber.ipynb). Resumable: finished 100-question
shards are skipped on restart. Outputs (in --out-dir, default results/cached_features/), with stem = {dataset}_{model}:
    {stem}_features.pt        {"features": [N, L, H] float16, "row_idx": [N], "layers": [...], ...}
    {stem}_metadata.parquet   one row per question (same columns as pope_metadata.parquet, plus dataset/model/n_input_tokens)
    {stem}_run_config.json    model, prompt, token ids, seed, versions, status
    {stem}_shards/            resume files (npz); safe to delete once the run is complete
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cache_io import file_names
from vlm_datasets import (DatasetBundle, amber_report, load_amber, load_pope, select_amber, select_pope, split_of)
from vlm_registry import REGISTRY, ModelSpec, make_prompt, parse_layers, resolve_yes_no_ids, score_logits


# ====================================================================================================================
# Model backend (GPU, transformers). Anything model-specific lives here; the rest of the file is backend-agnostic.
# ====================================================================================================================
class HFBackend:
    """forward(question, PIL image) -> (hidden [len(layers), H] float32, logits [V] float32, n_input_tokens)."""

    def __init__(self, spec: ModelSpec, device: str = "cuda", layers: str = "all"):
        import torch
        import transformers
        from transformers import AutoProcessor
        self.torch, self.spec, self.device = torch, spec, device
        self.torch_dtype = getattr(torch, spec.dtype)
        if device == "cuda":
            assert torch.cuda.is_available(), "No GPU. Colab: Runtime > Change runtime type > GPU."
            if spec.dtype == "bfloat16" and torch.cuda.get_device_capability(0)[0] < 8:
                print("NOTE: this GPU has no native bfloat16 (T4). It works but is slower; the Qwen2-VL-2B run did the same.")

        self.processor = AutoProcessor.from_pretrained(spec.hf_id, **spec.processor_kwargs)
        self.tok = self.processor.tokenizer
        self.yes_map, self.no_map = resolve_yes_no_ids(
            lambda s: self.tok.encode(s, add_special_tokens=False), spec.space_variants)
        self.yes_ids, self.no_ids = sorted(set(self.yes_map.values())), sorted(set(self.no_map.values()))
        print("yes tokens:", self.yes_map, "\nno tokens: ", self.no_map)

        self.model = self._load_model()
        cfg = self.model.config
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
        self.n_entries = int(text_cfg.num_hidden_layers) + 1          # embeddings + every transformer layer
        self.hidden_size = int(text_cfg.hidden_size)
        self.layers = parse_layers(layers, self.n_entries)
        self.info = dict(hf_id=spec.hf_id, dtype=spec.dtype, quant_4bit=spec.quant_4bit,
                         yes_token_ids=self.yes_map, no_token_ids=self.no_map,
                         n_hidden_state_entries=self.n_entries, hidden_size=self.hidden_size, layers_kept=self.layers,
                         torch=torch.__version__, transformers=transformers.__version__,
                         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        print(f"Loaded {spec.hf_id} ({spec.dtype}{', 4-bit' if spec.quant_4bit else ''}); hidden_states entries: "
              f"{self.n_entries} x {self.hidden_size}; keeping {len(self.layers)}")

    def _load_model(self):
        torch, spec = self.torch, self.spec
        try:
            from transformers import AutoModelForImageTextToText as Auto
        except ImportError:                                            # older transformers
            from transformers import AutoModelForVision2Seq as Auto
        kw = dict(torch_dtype=self.torch_dtype)
        if spec.quant_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=self.torch_dtype)
            kw["device_map"] = {"": 0}
        model = Auto.from_pretrained(spec.hf_id, **kw)
        if not spec.quant_4bit:
            model = model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def render(self, question: str) -> str:
        prompt = make_prompt(question)
        if self.spec.prompt_format is not None:
            return self.spec.prompt_format.replace("{prompt}", prompt)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def decode(self, token_id: int) -> str:
        return self.tok.decode([int(token_id)])

    def forward(self, question: str, image):
        torch = self.torch
        text = self.render(question)
        inputs = self.processor(text=text, images=[image.convert("RGB")], return_tensors="pt")
        inputs = inputs.to(self.device, self.torch_dtype)             # moves everything, casts only float tensors
        with torch.inference_mode():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        if hs is None or len(hs) != self.n_entries:
            raise RuntimeError(f"expected {self.n_entries} hidden_states entries (embeddings + layers) from the language "
                               f"model, got {None if hs is None else len(hs)}. This model needs a custom hidden-state accessor.")
        hidden = torch.stack([hs[i][0, -1, :] for i in self.layers]).float().cpu().numpy()
        logits = out.logits[0, -1, :].float().cpu().numpy()
        return hidden, logits, int(inputs["input_ids"].shape[1])

    def cleanup(self):
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


# ====================================================================================================================
# Backend-agnostic extraction loop with shard checkpointing (unit-tested with a fake backend)
# ====================================================================================================================
def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def make_record(m, scores: dict, backend, seed: int, dataset: str, model_key: str, n_tokens: int) -> dict:
    p_yes, p_no = scores["p_yes"], scores["p_no"]
    pred = "yes" if p_yes > p_no else "no"
    return dict(
        row_idx=int(m["row_idx"]), id=m["id"], question_id=m["question_id"], image_id=m["image_id"], setting=m["setting"],
        question=m["question"], gt_answer=m["gt_answer"], pred_answer=pred,
        hallucination=int(pred == "yes" and m["gt_answer"] == "no"),
        p_yes=p_yes, p_no=p_no, p_yes_norm=scores["p_yes_norm"], margin=scores["margin"],
        binary_entropy=scores["binary_entropy"], vocab_entropy=scores["vocab_entropy"], answer_mass=p_yes + p_no,
        top1_token=backend.decode(scores["top1_id"]), split=split_of(m["image_id"], seed), prompt=make_prompt(m["question"]),
        dataset=dataset, model_key=model_key, n_input_tokens=int(n_tokens))


def process_question(backend, bundle_meta_by_row, get_image, r, seed, dataset, model_key):
    m = bundle_meta_by_row.loc[r]
    hidden, logits, n_tokens = backend.forward(m["question"], get_image(r))
    scores = score_logits(logits, backend.yes_ids, backend.no_ids)
    return hidden, make_record(m, scores, backend, seed, dataset, model_key, n_tokens)


def list_shards(shard_dir: Path):
    return [p for p in sorted(Path(shard_dir).glob("shard_*.npz")) if ".tmp" not in p.name]


def save_shard(shard_dir: Path, k: int, row_idx, features: np.ndarray, recs: list):
    tmp, final = shard_dir / f"shard_{k:05d}.tmp.npz", shard_dir / f"shard_{k:05d}.npz"
    np.savez(tmp, row_idx=np.asarray(row_idx, dtype=np.int64), features=features,
             meta_json=np.array(json.dumps(recs, default=_json_default)))
    os.replace(tmp, final)                                            # a shard is either complete or absent


def load_done(shard_dir: Path):
    """Row indices already stored, and the next shard number. Corrupt / half-written shards are deleted and redone."""
    for stale in Path(shard_dir).glob("shard_*.tmp.npz"):
        stale.unlink()
    done, next_k = set(), 0
    for p in list_shards(shard_dir):
        next_k = max(next_k, int(p.stem.split("_")[1]) + 1)
        try:
            with np.load(p, allow_pickle=False) as z:
                done.update(int(r) for r in z["row_idx"])
        except Exception as e:                                        # noqa: BLE001
            print(f"  corrupt shard {p.name} ({e}); deleting, its rows will be redone")
            p.unlink()
    return done, next_k


def run_extraction(backend, bundle: DatasetBundle, shard_dir: Path, model_key: str, checkpoint_every: int = 100,
                   seed: int = 0, limit: int | None = None, store_dtype: str = "float16", log=print) -> dict:
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    selected = bundle.selected if limit is None else bundle.selected[:limit]
    meta_by_row = bundle.meta.set_index("row_idx", drop=False)
    done, next_k = load_done(shard_dir)
    todo = [r for r in selected if r not in done]
    log(f"selected {len(selected)} | already done {len(set(selected) & done)} | to do {len(todo)}")
    t0, n_done = time.time(), 0
    for s in range(0, len(todo), checkpoint_every):
        chunk = todo[s:s + checkpoint_every]
        feats, recs = [], []
        for r in chunk:
            hidden, rec = process_question(backend, meta_by_row, bundle.get_image, r, seed, bundle.name, model_key)
            feats.append(hidden)
            recs.append(rec)
        F = np.stack(feats)
        if not np.isfinite(F).all():
            raise RuntimeError("non-finite hidden states (try dtype bfloat16 instead of float16)")
        if store_dtype == "float16" and np.abs(F).max() >= 65000:
            raise RuntimeError(f"hidden-state magnitude {np.abs(F).max():.0f} would overflow float16; use --store-dtype float32")
        save_shard(shard_dir, next_k, chunk, F.astype(store_dtype), recs)
        next_k += 1
        n_done += len(chunk)
        el = time.time() - t0
        log(f"  {n_done}/{len(todo)} done | {el / n_done:.2f} s/question | eta {(len(todo) - n_done) * el / n_done / 60:.0f} min")
        if hasattr(backend, "cleanup"):
            backend.cleanup()
    return dict(n_selected=len(selected), n_new=n_done, seconds=time.time() - t0)


def merge_shards(shard_dir: Path, selected):
    sel = set(selected)
    rows, feats, recs, seen = [], [], [], set()
    for p in list_shards(shard_dir):
        with np.load(p, allow_pickle=False) as z:
            ri, F, M = z["row_idx"], z["features"], json.loads(z["meta_json"].item())
        for j, r in enumerate(ri.tolist()):
            if r in seen or r not in sel:
                continue
            seen.add(r)
            rows.append(r)
            feats.append(F[j])
            recs.append(M[j])
    if not rows:
        raise RuntimeError(f"no finished shards found in {shard_dir}")
    perm = np.argsort(rows)
    return (np.asarray(rows, dtype=np.int64)[perm], np.stack([feats[i] for i in perm]),
            pd.DataFrame([recs[i] for i in perm]))


def write_pt(path: Path, row_idx, features: np.ndarray, layers, n_entries: int):
    import torch
    tmp = Path(str(path) + ".tmp")
    torch.save(dict(features=torch.from_numpy(np.ascontiguousarray(features)),
                    row_idx=torch.from_numpy(np.asarray(row_idx, dtype=np.int64)),
                    layers=[int(i) for i in layers], n_hidden_state_entries=int(n_entries),
                    token_position="last_prompt_token",
                    note="features[i] <-> metadata row with row_idx[i]; shape [N, len(layers), hidden]; "
                         "layers = indices into the model's hidden_states tuple (0 = embeddings)"), tmp)
    os.replace(tmp, path)


def write_meta(meta: pd.DataFrame, path: Path) -> Path:
    try:
        meta.to_parquet(path, index=False)
        return path
    except Exception as e:                                            # noqa: BLE001
        csv = path.with_suffix(".csv")
        print(f"parquet unavailable ({e}); writing CSV instead")
        meta.to_csv(csv, index=False)
        return csv


# ====================================================================================================================
# Verification and smoke test
# ====================================================================================================================
def summarise(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    m["correct"] = (m["pred_answer"] == m["gt_answer"]).astype(int)
    g = m.groupby("setting")
    return pd.DataFrame({
        "n": g.size(), "accuracy": g["correct"].mean(),
        "yes_rate": g["pred_answer"].apply(lambda s: (s == "yes").mean()),
        "gt_yes_rate": g["gt_answer"].apply(lambda s: (s == "yes").mean()),
        "false_yes_rate": m[m.gt_answer == "no"].groupby("setting")["hallucination"].mean(),
    }).round(4)


def verify(meta: pd.DataFrame, features: np.ndarray, row_idx, selected, features_path: Path | None = None) -> list[str]:
    flags = []
    print(f"features {tuple(features.shape)} {features.dtype} | metadata {meta.shape}")
    if not (len(meta) == len(features) == len(row_idx) and (meta["row_idx"].to_numpy() == np.asarray(row_idx)).all()):
        flags.append("features / metadata misaligned")
    missing = sorted(set(selected) - set(np.asarray(row_idx).tolist()))
    dup = int(pd.Series(np.asarray(row_idx)).duplicated().sum())
    nonfinite = int((~np.isfinite(features.astype(np.float32))).sum())
    print(f"missing rows: {len(missing)} | duplicates: {dup} | non-finite values: {nonfinite}")
    if missing or dup or nonfinite:
        flags.append(f"missing={len(missing)} dup={dup} nonfinite={nonfinite}")
    if features_path is not None and Path(features_path).exists():
        size, expected = Path(features_path).stat().st_size / 1e9, features.nbytes / 1e9
        print(f"file size {size:.3f} GB (expected ~{expected:.3f} GB)")
        if not 0.9 * expected <= size <= 1.2 * expected:
            flags.append("feature file size far from expectation")
    mass = meta["answer_mass"]
    print(f"answer_mass P(yes)+P(no): mean {mass.mean():.3f} | share < 0.9: {(mass < 0.9).mean():.3f}")
    print("most common first tokens:", meta["top1_token"].value_counts().head(6).to_dict())
    if mass.mean() < 0.8:
        flags.append("answer_mass is low: the model often does not start with yes/no -> check yes/no token ids and the prompt")
    print("\n", summarise(meta).to_string())
    print("split sizes:", meta["split"].value_counts().to_dict())
    if meta.groupby("image_id")["split"].nunique().max() > 1:
        flags.append("an image appears in two splits")
    absent = meta[meta.gt_answer == "no"]
    if len(absent) and absent["hallucination"].sum() < 20:
        flags.append(f"only {int(absent['hallucination'].sum())} hallucinations: AUROC on this data will be very noisy")
    if (meta["gt_answer"] == "no").all():
        print("NOTE: every ground-truth answer in this cache is 'no': Task A (hallucinated-yes vs correct-yes) is undefined "
              "here; only Task B can be evaluated and an error is exactly a 'yes' answer. (The full AMBER subsets all contain "
              "'yes' truths, so this only happens with a very small or filtered selection.)")
    for f in flags:
        print("!!! FLAG:", f)
    return flags


def smoke_test(backend, bundle: DatasetBundle, n: int, model_key: str, seed: int):
    meta_by_row = bundle.meta.set_index("row_idx", drop=False)
    recs, secs = [], []
    for r in bundle.selected[:n]:
        t = time.time()
        hidden, rec = process_question(backend, meta_by_row, bundle.get_image, r, seed, bundle.name, model_key)
        secs.append(time.time() - t)
        recs.append(rec)
        assert np.isfinite(hidden).all(), "non-finite hidden state"
        print(f"  row {rec['row_idx']:5d} gt={rec['gt_answer']:3s} pred={rec['pred_answer']:3s} p_yes_norm={rec['p_yes_norm']:.3f} "
              f"mass={rec['answer_mass']:.3f} top1={rec['top1_token']!r} tokens={rec['n_input_tokens']}")
    mass = np.mean([x["answer_mass"] for x in recs])
    spq = float(np.mean(secs[1:] if len(secs) > 1 else secs))             # skip the first (warm-up) call
    print(f"\nhidden state shape per question: {hidden.shape} | mean answer_mass {mass:.3f} | {spq:.2f} s/question")
    print(f"estimated time for the {len(bundle.selected)} selected questions: {spq * len(bundle.selected) / 3600:.1f} h")
    if mass < 0.8:
        print("!!! answer_mass is low. The first generated token is not yes/no. Do NOT start the full run; check the "
              "rendered prompt and the yes/no token ids above.")
    else:
        print("Smoke test OK.")
    return mass, spq


# ====================================================================================================================
# CLI
# ====================================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(REGISTRY), help="model key (see --list-models)")
    p.add_argument("--dataset", choices=["pope", "amber"], default="pope")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--inspect", action="store_true", help="load and describe the dataset only (no model, no GPU)")
    p.add_argument("--smoke", type=int, default=0, metavar="N", help="run N questions, print diagnostics, write nothing")
    p.add_argument("--subset-size", type=int, default=None, help="POPE: first N rows of the seeded order (default: all 9000)")
    p.add_argument("--per-subset", type=int, default=1000, help="AMBER: random queries per subset (default 1000; None=all via 0)")
    p.add_argument("--amber-subsets", default="existence,attribute,relation")
    p.add_argument("--pope-path", default="data/pope")
    p.add_argument("--amber-repo", default="data/amber_repo", help="clone of github.com/junyangwang0410/AMBER")
    p.add_argument("--amber-images", default="data/amber_images", help="folder with AMBER_*.jpg (searched recursively)")
    p.add_argument("--amber-hf-images", default=None, help="take pixels from this HF mirror instead, e.g. visual-preference/AMBER")
    p.add_argument("--out-dir", default="results/cached_features")
    p.add_argument("--layers", default="all", help="'all' | 'every:4' | comma list of hidden_states indices (negative ok)")
    p.add_argument("--store-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="keep 0: it defines the question order and the image-level splits")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def build_bundle(a) -> DatasetBundle:
    if a.dataset == "pope":
        return select_pope(load_pope(a.pope_path), a.subset_size, a.seed)
    bundle = load_amber(a.amber_repo, a.amber_images, a.amber_hf_images)
    print(amber_report(bundle.meta))
    per = a.per_subset if a.per_subset and a.per_subset > 0 else None
    bundle = select_amber(bundle, [s.strip() for s in a.amber_subsets.split(",") if s.strip()], per, a.seed)
    sel = bundle.meta.set_index("row_idx").loc[bundle.selected]
    print("selected:", len(sel), "| per subset / truth:", sel.groupby("setting")["gt_answer"].value_counts().to_dict())
    return bundle


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.list_models:
        for k, s in REGISTRY.items():
            print(f"{k:14s} {s.hf_id:38s} {s.dtype:9s} {'4-bit ' if s.quant_4bit else ''}{s.notes}")
        return 0
    bundle = build_bundle(a)
    if a.inspect:
        print(bundle.meta.head(3).T if a.dataset == "amber" else pd.crosstab(bundle.meta["setting"], bundle.meta["gt_answer"]))
        print("selected rows:", len(bundle.selected), bundle.selection_desc)
        return 0
    if not a.model:
        sys.exit("--model is required (see --list-models)")
    spec = REGISTRY[a.model]
    backend = HFBackend(spec, a.device, a.layers)

    if a.smoke:
        smoke_test(backend, bundle, a.smoke, spec.key, a.seed)
        return 0

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_name, m_name, c_name = file_names(a.dataset, spec.key)
    shard_dir = out / f"{a.dataset}_{spec.key}_shards"
    config_path = out / c_name
    config = dict(model_key=spec.key, **backend.info, dataset=a.dataset, selection=bundle.selection_desc,
                  n_selected=len(bundle.selected), prompt_template=make_prompt("{question}"), seed=a.seed,
                  prompt_format=spec.prompt_format or "processor.apply_chat_template (1 image, generation prompt)",
                  processor_kwargs=spec.processor_kwargs, store_dtype=a.store_dtype,
                  token_position="last prompt token", split_rule="md5('{seed}-{image_id}') -> 70/15/15 by image",
                  scoring="forced choice over yes/no token ids at the last prompt token; pred yes iff P(yes) > P(no)")
    check = ["model_key", "hf_id", "dtype", "yes_token_ids", "no_token_ids", "seed", "selection", "layers_kept", "prompt_format"]
    if config_path.exists():
        old = json.loads(config_path.read_text())
        bad = [k for k in check if old.get(k) != json.loads(json.dumps(config.get(k)))]
        if bad:
            sys.exit(f"Existing run in {out} used different settings for {bad}. Use another --out-dir or delete the shards.")
        print("Resuming a previous run (config matches).")
    config.update(status="running", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    config_path.write_text(json.dumps(config, indent=2, default=_json_default))

    stats = run_extraction(backend, bundle, shard_dir, spec.key, a.checkpoint_every, a.seed, None, a.store_dtype)
    row_idx, features, meta = merge_shards(shard_dir, bundle.selected)
    write_pt(out / f_name, row_idx, features, backend.layers, backend.n_entries)
    meta_path = write_meta(meta, out / m_name)
    print(f"wrote {out / f_name}, {meta_path}")
    flags = verify(meta, features, row_idx, bundle.selected, out / f_name)
    config.update(status="complete" if len(meta) == len(bundle.selected) and not flags else "complete_with_flags",
                  flags=flags, n_rows_saved=int(len(meta)), seconds_this_session=round(stats["seconds"]),
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    config_path.write_text(json.dumps(config, indent=2, default=_json_default))
    print("Done. Do NOT commit the .pt file (it is gitignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
