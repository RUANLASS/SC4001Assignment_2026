import numpy as np
import pytest
from scipy.special import logsumexp, softmax

from vlm_registry import (REGISTRY, make_prompt, parse_layers, resolve_yes_no_ids, score_logits)


def test_prompt_is_frozen():
    assert make_prompt("Is there a dog in the image?") == "Is there a dog in the image? Answer yes or no."


def test_registry_is_consistent():
    for k, s in REGISTRY.items():
        assert s.key == k
        assert s.dtype in ("bfloat16", "float16")
        assert s.prompt_format is None or "{prompt}" in s.prompt_format
    assert REGISTRY["qwen2vl-2b"].hf_id == "Qwen/Qwen2-VL-2B-Instruct"       # must match the team's cache
    assert REGISTRY["qwen2vl-2b"].dtype == "bfloat16" and not REGISTRY["qwen2vl-2b"].space_variants


def test_score_logits_matches_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=1000) * 3
    yes, no = [3, 10], [7, 99]
    s = score_logits(logits, yes, no)
    p = softmax(logits.astype(np.float64))
    assert s["p_yes"] == pytest.approx(p[yes].sum(), rel=1e-9)
    assert s["p_no"] == pytest.approx(p[no].sum(), rel=1e-9)
    assert s["margin"] == pytest.approx(np.log(p[yes].sum() / p[no].sum()), rel=1e-9)
    assert s["p_yes_norm"] == pytest.approx(p[yes].sum() / (p[yes].sum() + p[no].sum()), rel=1e-9)
    assert s["vocab_entropy"] == pytest.approx(-(p * np.log(p)).sum(), rel=1e-9)
    assert s["top1_id"] == int(np.argmax(logits))


def test_score_logits_binary_entropy_extremes():
    logits = np.zeros(50)
    logits[1] = 30.0                      # overwhelming "yes"
    s = score_logits(logits, [1], [2])
    assert s["binary_entropy"] < 1e-6 and s["p_yes_norm"] > 0.999999
    logits = np.zeros(50)                 # tie -> max binary entropy
    s = score_logits(logits, [1], [2])
    assert s["binary_entropy"] == pytest.approx(1.0) and s["p_yes_norm"] == pytest.approx(0.5)


def test_score_logits_handles_huge_logits_without_overflow():
    logits = np.full(100, -1e4)
    logits[5], logits[6] = 1e4, 1e4 - 2
    s = score_logits(logits, [5], [6])
    assert np.isfinite(list(v for k, v in s.items() if k != "top1_id")).all()
    assert s["margin"] == pytest.approx(2.0)


def _fake_encode(vocab):
    # sentencepiece-like: unknown strings split into several tokens
    return lambda s: [vocab[s]] if s in vocab else [1000 + i for i, _ in enumerate(s)]


def test_resolve_yes_no_ids_qwen_style_no_space():
    enc = _fake_encode({"Yes": 1, "yes": 2, "YES": 3, "No": 4, "no": 5, "NO": 6, " Yes": 7, " No": 8})
    y, n = resolve_yes_no_ids(enc, space_variants=False)
    assert set(y.values()) == {1, 2, 3} and set(n.values()) == {4, 5, 6}


def test_resolve_yes_no_ids_with_space_variants_and_drops_multitoken():
    enc = _fake_encode({"Yes": 1, "yes": 2, "No": 4, "no": 5, " Yes": 7, " No": 8})   # "YES"/"NO"/" yes" are multi-token
    y, n = resolve_yes_no_ids(enc, space_variants=True)
    assert set(y.values()) == {1, 2, 7} and set(n.values()) == {4, 5, 8}


def test_resolve_yes_no_ids_rejects_collisions_and_empty():
    with pytest.raises(ValueError):
        resolve_yes_no_ids(_fake_encode({"Yes": 1, "No": 1}))
    with pytest.raises(ValueError):
        resolve_yes_no_ids(_fake_encode({}))


def test_parse_layers():
    assert parse_layers("all", 5) == [0, 1, 2, 3, 4]
    assert parse_layers("every:2", 6) == [0, 2, 4, 5]          # always keeps the last entry
    assert parse_layers("0,-1,3", 6) == [0, 3, 5]
    with pytest.raises(ValueError):
        parse_layers("9", 6)
