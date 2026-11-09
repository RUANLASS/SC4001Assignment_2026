"""Pure statistics helpers for the probe evaluation (no model, no I/O): paired image-level bootstrap, calibration
metrics, Platt scaling and accuracy-coverage areas.

Everything here takes plain numpy arrays so it can be unit-tested on tiny synthetic inputs (see tests/test_probe_stats.py).
`probe_transfer_eval.py` imports these and applies them to the real caches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _auc(y, s) -> float:
    """AUROC, NaN when only one class is present."""
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(y) and len(np.unique(y)) == 2 else float("nan")


def _split_by_group(groups) -> tuple[list, int]:
    """Row indices of every group (image), so a bootstrap draw can resample whole images."""
    codes, uniq = pd.factorize(np.asarray(groups))
    by_group = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    return by_group, len(uniq)


# --------------------------------------------------------------------------------------------------------------------
# paired comparison of two scores on the SAME rows
# --------------------------------------------------------------------------------------------------------------------
def paired_auc_diff(y, s_probe, s_baseline, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired image-level bootstrap of AUROC(s_probe) - AUROC(s_baseline).

    Each bootstrap draw resamples whole images (groups) with replacement and evaluates BOTH scores on exactly the same
    resampled rows, so the shared sampling noise cancels in the difference. Returns the observed difference, the 95%
    percentile CI and a two-sided bootstrap p-value for H0: difference == 0.

    p-value: 2 * min(P*(d <= 0), P*(d >= 0)) with +1 smoothing, i.e. (1 + #draws on that side) / (B + 1), capped at 1.
    It can therefore not be exactly 0; the smallest value is about 2 / (B + 1). Draws where either AUROC is undefined
    (a resample with a single class) are dropped. Positive `diff` means the first score (the probe) is better.
    """
    y = np.asarray(y)
    s_probe, s_baseline = np.asarray(s_probe, dtype=float), np.asarray(s_baseline, dtype=float)
    out = dict(auc_probe=_auc(y, s_probe), auc_baseline=_auc(y, s_baseline), diff=np.nan, ci_lo=np.nan, ci_hi=np.nan,
               p_value=np.nan, n=int(len(y)), n_pos=int(y.sum()), n_boot_valid=0)
    if np.isnan(out["auc_probe"]) or np.isnan(out["auc_baseline"]) or n_boot <= 0:
        return out
    out["diff"] = out["auc_probe"] - out["auc_baseline"]
    by_group, n_groups = _split_by_group(groups)
    rng, d = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[g] for g in rng.integers(0, n_groups, n_groups)])
        a, b = _auc(y[idx], s_probe[idx]), _auc(y[idx], s_baseline[idx])
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    out["n_boot_valid"] = len(d)
    if len(d) >= 20:
        d = np.asarray(d)
        out["ci_lo"], out["ci_hi"] = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        p_le, p_ge = (1 + np.sum(d <= 0)) / (len(d) + 1), (1 + np.sum(d >= 0)) / (len(d) + 1)
        out["p_value"] = float(min(1.0, 2 * min(p_le, p_ge)))
    return out


# --------------------------------------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------------------------------------
def reliability_bins(p, y, n_bins: int = 15) -> pd.DataFrame:
    """Equal-width bins on [0, 1] of the predicted probability of the positive class."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    b = np.minimum((np.clip(p, 0, 1) * n_bins).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append(dict(bin=k, lo=k / n_bins, hi=(k + 1) / n_bins, n=int(m.sum()),
                         mean_p=float(p[m].mean()) if m.any() else np.nan, frac_pos=float(y[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def ece_score(p, y, n_bins: int = 15) -> float:
    """Expected calibration error of predicted probabilities of the positive class: sum_b (n_b/n) |frac_pos_b - mean_p_b|."""
    t = reliability_bins(p, y, n_bins)
    t = t[t.n > 0]
    return float((t.n / t.n.sum() * (t.frac_pos - t.mean_p).abs()).sum()) if len(t) else float("nan")


def brier_score(p, y) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(scores, y):
    """Platt scaling: logistic regression of the label on logit(score). Fit it on DEV scores only. None if one class."""
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(C=1e6, max_iter=200).fit(_logit(scores).reshape(-1, 1), y)


def platt_apply(platt, scores) -> np.ndarray:
    return platt.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


# --------------------------------------------------------------------------------------------------------------------
# accuracy-coverage / risk-coverage
# --------------------------------------------------------------------------------------------------------------------
def aurc_metrics(err_score, correct) -> dict:
    """Areas for the selective-prediction curve. `err_score`: higher = less trustworthy; `correct`: 1/0 per answer.

    Answers are kept most-trusted first; at coverage k/n the risk is 1 - (accuracy of the kept k). AURC = mean over k of
    that risk (lower is better) and AUACC = 1 - AURC (area under the accuracy-coverage curve). Reference values:
    `random_aurc` = overall error rate (random abstention) and `oracle_aurc` (all errors abstained on first).
    """
    err_score, correct = np.asarray(err_score, dtype=float), np.asarray(correct, dtype=float)
    n = len(correct)
    if n == 0:
        return dict(aurc=np.nan, auacc=np.nan, oracle_aurc=np.nan, random_aurc=np.nan, n=0)
    k = np.arange(1, n + 1)
    risk = 1 - np.cumsum(correct[np.argsort(err_score, kind="stable")]) / k
    n_ok = int(correct.sum())
    oracle_risk = 1 - np.minimum(k, n_ok) / k
    return dict(aurc=float(risk.mean()), auacc=float(1 - risk.mean()), oracle_aurc=float(oracle_risk.mean()),
                random_aurc=float(1 - correct.mean()), n=int(n))
