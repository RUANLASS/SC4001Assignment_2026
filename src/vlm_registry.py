"""Model registry and answer scoring for the multi-model hallucination-probing experiments.

Everything here is pure Python / numpy (no torch, no transformers), so it can be unit-tested on a laptop.
The GPU-only parts (loading a model and running it) live in ``extract_features_multi.py``.

Protocol (identical for every model, and identical to the Qwen2-VL-2B run in notebook 01):
  * prompt          = "{question} Answer yes or no."  wrapped in the model's chat format, one image, generation prompt on
  * scored position = the LAST PROMPT TOKEN (the position that predicts the first answer token)
  * yes/no score    = forced choice: P(yes) = sum of probabilities of the "yes" token ids, P(no) likewise;
                      predicted answer is "yes" iff P(yes) > P(no)
  * hallucination   = 1 iff predicted yes AND ground truth no
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PROMPT_SUFFIX = " Answer yes or no."  # frozen; same string as src/feature_extraction.py and notebook 01


def make_prompt(question: str) -> str:
    return f"{question}{PROMPT_SUFFIX}"


@dataclass(frozen=True)
class ModelSpec:
    key: str                         # short name used in file names and on the command line
    hf_id: str                       # Hugging Face model id
    family: str
    dtype: str = "bfloat16"          # compute dtype ("bfloat16" | "float16")
    prompt_format: str | None = None  # None -> processor.apply_chat_template; else a format string containing {prompt}
    space_variants: bool = False     # also score " Yes"/" No" (needed when the answer token carries a leading space)
    processor_kwargs: dict = field(default_factory=dict)
    quant_4bit: bool = False         # load with bitsandbytes 4-bit (needs `pip install bitsandbytes`)
    notes: str = ""


# Qwen processors resize to a pixel budget. POPE images (COCO, <= 640x640 = 0.41 MP) are untouched by this cap, so the
# Qwen2-VL-2B features stay identical to the teammate run; the cap only protects against the very large AMBER images.
_QWEN_PIXELS = {"max_pixels": 1280 * 28 * 28}

REGISTRY: dict[str, ModelSpec] = {
    "qwen2vl-2b": ModelSpec(
        key="qwen2vl-2b", hf_id="Qwen/Qwen2-VL-2B-Instruct", family="qwen-vl", dtype="bfloat16",
        processor_kwargs=dict(_QWEN_PIXELS),
        notes="Reference model of the team (cache from notebook 01). Use this key to extract AMBER with the same settings."),
    "qwen25vl-3b": ModelSpec(
        key="qwen25vl-3b", hf_id="Qwen/Qwen2.5-VL-3B-Instruct", family="qwen-vl", dtype="bfloat16",
        processor_kwargs=dict(_QWEN_PIXELS),
        notes="Same family, one generation newer and larger. Needs transformers >= 4.49. Keep bfloat16 (fp16 can overflow)."),
    "smolvlm-500m": ModelSpec(
        key="smolvlm-500m", hf_id="HuggingFaceTB/SmolVLM-500M-Instruct", family="idefics3", dtype="bfloat16",
        space_variants=True, processor_kwargs={"do_image_splitting": False},
        notes="Small, different architecture (SigLIP + SmolLM2); cheapest model to run. Image splitting is off so each image is one tile."),
    "llava15-7b": ModelSpec(
        key="llava15-7b", hf_id="llava-hf/llava-1.5-7b-hf", family="llava", dtype="float16",
        prompt_format="USER: <image>\n{prompt} ASSISTANT:", space_variants=True, quant_4bit=True,
        notes="The classic POPE model (CLIP + Llama-2). 4-bit quantised so it fits a 15 GB T4; hidden states are therefore "
              "those of the quantised model. Slowest option."),
}


# --------------------------------------------------------------------------------------------------------------------
# yes / no token ids
# --------------------------------------------------------------------------------------------------------------------
_YES = ["Yes", "yes", "YES"]
_NO = ["No", "no", "NO"]


def _single_token_ids(encode, strings):
    """encode: str -> list[int]. Keep only strings that are exactly one token."""
    out = {}
    for s in strings:
        ids = list(encode(s))
        if len(ids) == 1:
            out[s] = int(ids[0])
    return out


def resolve_yes_no_ids(encode, space_variants: bool = False):
    """Return (yes_map, no_map): {string: token id} for the single-token spellings of yes / no.

    ``encode`` is ``lambda s: tokenizer.encode(s, add_special_tokens=False)``.
    With ``space_variants`` the leading-space spellings (" Yes", ...) are scored as well. Raises if either side is empty or
    the two sides share a token id. Always print the result and check ``answer_mass`` (see ``smoke`` mode): if the first
    answer token is not one of these ids, P(yes) + P(no) will be far below 1.
    """
    yes_s = _YES + ([" " + s for s in _YES] if space_variants else [])
    no_s = _NO + ([" " + s for s in _NO] if space_variants else [])
    yes_map, no_map = _single_token_ids(encode, yes_s), _single_token_ids(encode, no_s)
    if not yes_map or not no_map:
        raise ValueError(f"No single-token yes/no spellings found (yes={yes_map}, no={no_map}).")
    if set(yes_map.values()) & set(no_map.values()):
        raise ValueError(f"yes and no share a token id: yes={yes_map}, no={no_map}")
    return yes_map, no_map


# --------------------------------------------------------------------------------------------------------------------
# scoring (numpy, float64) - mirrors score_logits() in notebook 01
# --------------------------------------------------------------------------------------------------------------------
def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def score_logits(logits: np.ndarray, yes_ids, no_ids) -> dict:
    """logits: [V] next-token logits at the last prompt token. Returns plain floats (and the top-1 token id)."""
    x = np.asarray(logits, dtype=np.float64)
    logp = x - _logsumexp(x)
    lp_yes = _logsumexp(logp[list(yes_ids)])
    lp_no = _logsumexp(logp[list(no_ids)])
    margin = lp_yes - lp_no
    p_yes_norm = 1.0 / (1.0 + np.exp(-margin))
    q = min(max(p_yes_norm, 1e-12), 1 - 1e-12)
    binary_entropy = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    vocab_entropy = float(-(np.exp(logp) * logp).sum())          # natural log, as in notebook 01
    return dict(p_yes=float(np.exp(lp_yes)), p_no=float(np.exp(lp_no)), p_yes_norm=float(p_yes_norm),
                margin=float(margin), binary_entropy=float(binary_entropy), vocab_entropy=vocab_entropy,
                top1_id=int(np.argmax(x)))


def parse_layers(arg: str, n_entries: int) -> list[int]:
    """'all' | 'every:4' | '0,8,16,-1'  ->  sorted list of hidden_states indices to keep (negative = from the end)."""
    arg = (arg or "all").strip().lower()
    if arg == "all":
        return list(range(n_entries))
    if arg.startswith("every:"):
        k = int(arg.split(":")[1])
        keep = set(range(0, n_entries, k)) | {n_entries - 1}
        return sorted(keep)
    keep = set()
    for tok in arg.split(","):
        i = int(tok)
        i = i + n_entries if i < 0 else i
        if not 0 <= i < n_entries:
            raise ValueError(f"layer {tok} out of range for {n_entries} hidden-state entries")
        keep.add(i)
    return sorted(keep)
