"""Unit tests for src/probe_stats.py on tiny hand-checkable inputs (CPU, no caches needed)."""
import numpy as np
import pytest

import probe_stats as ps


def _clustered_problem(seed=0, n_img=60, per=6, signal=1.0):
    """Labels are constant within an image (like questions on one image); two scores of different quality."""
    rng = np.random.default_rng(seed)
    y_img = rng.integers(0, 2, n_img)
    y = np.repeat(y_img, per)
    groups = np.repeat(np.arange(n_img), per)
    good = y * signal + rng.normal(0, 1, len(y))
    return y, good, groups, rng


# ---------------------------------------------------------------- paired_auc_diff
def test_paired_diff_identical_scores_is_zero_and_not_significant():
    y, s, g, _ = _clustered_problem()
    r = ps.paired_auc_diff(y, s, s.copy(), g, n_boot=200, seed=0)
    assert r["diff"] == 0 and r["ci_lo"] == r["ci_hi"] == 0 and r["p_value"] == 1.0


def test_paired_diff_detects_a_clearly_better_score():
    y, good, g, rng = _clustered_problem(signal=2.0)
    bad = rng.normal(0, 1, len(y))                                  # uninformative baseline
    r = ps.paired_auc_diff(y, good, bad, g, n_boot=400, seed=1)
    assert r["diff"] > 0.2 and r["ci_lo"] > 0 and r["p_value"] < 0.01
    assert r["auc_probe"] == pytest.approx(ps._auc(y, good)) and r["n"] == len(y) and r["n_pos"] == int(y.sum())
    flipped = ps.paired_auc_diff(y, bad, good, g, n_boot=400, seed=1)      # swapping the arguments flips the sign
    assert flipped["diff"] == pytest.approx(-r["diff"]) and flipped["ci_hi"] < 0


def test_paired_diff_is_tighter_than_two_independent_cis():
    """Same resampled images for both scores => the shared noise cancels. Near-duplicate scores give a tiny paired CI."""
    y, a, g, rng = _clustered_problem(signal=1.0)
    b = a + rng.normal(0, 0.05, len(a))
    r = ps.paired_auc_diff(y, a, b, g, n_boot=300, seed=2)
    width_paired = r["ci_hi"] - r["ci_lo"]
    # width of the CI of the AUROC of `a` alone (one score) is much larger than the CI of the paired difference
    by_group, ng = ps._split_by_group(g)
    rng2, aucs = np.random.default_rng(2), []
    for _ in range(300):
        idx = np.concatenate([by_group[k] for k in rng2.integers(0, ng, ng)])
        aucs.append(ps._auc(y[idx], a[idx]))
    assert width_paired < 0.3 * (np.percentile(aucs, 97.5) - np.percentile(aucs, 2.5))


def test_paired_diff_resamples_whole_images():
    """If a single image holds all of one class, draws that miss it have a single class and are dropped (never crash)."""
    y = np.array([0] * 40 + [1] * 4)
    g = np.array([f"i{k}" for k in range(40)] + ["pos"] * 4)
    s = np.random.default_rng(0).normal(size=44)
    r = ps.paired_auc_diff(y, s, -s, g, n_boot=100, seed=0)
    assert r["n_boot_valid"] < 100                                   # some draws lost the only positive image


def test_paired_diff_single_class_is_nan_not_an_error():
    r = ps.paired_auc_diff(np.ones(10), np.arange(10.0), np.arange(10.0), np.arange(10), n_boot=50)
    assert np.isnan(r["diff"]) and np.isnan(r["p_value"])


# ---------------------------------------------------------------- calibration
def test_ece_and_brier_known_values():
    p, y = np.full(10, 0.9), np.array([1] * 5 + [0] * 5)
    assert ps.ece_score(p, y) == pytest.approx(0.4)                  # one bin: |0.5 - 0.9|
    assert ps.brier_score(p, y) == pytest.approx(0.5 * 0.01 + 0.5 * 0.81)
    assert ps.ece_score([], []) != ps.ece_score([], [])              # NaN on empty input
    assert ps.ece_score([0.0, 1.0], [0, 1]) == 0                     # perfectly confident and right


def test_ece_is_small_for_calibrated_probabilities_and_large_for_overconfident_ones():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)                          # labels drawn from the probabilities
    assert ps.ece_score(p, y) < 0.02
    assert ps.ece_score(np.clip(p * 0.3 + 0.7, 0, 1), y) > 0.2        # same labels, shifted probabilities


def test_reliability_bins_counts_add_up():
    t = ps.reliability_bins(np.array([0.05, 0.06, 0.95, 1.0]), np.array([0, 1, 1, 1]), n_bins=10)
    assert t.n.sum() == 4 and t.loc[0, "n"] == 2 and t.loc[9, "n"] == 2 and t.loc[0, "frac_pos"] == 0.5


def test_platt_scaling_repairs_a_shifted_probe_when_fitted_on_dev_only():
    rng = np.random.default_rng(1)
    true_p = rng.random(6000)
    y = (rng.random(6000) < true_p).astype(int)
    shifted = np.clip(true_p * 0.4 + 0.5, 0.01, 0.99)               # monotone, so AUROC is unchanged, calibration is bad
    dev, test = slice(0, 3000), slice(3000, 6000)
    platt = ps.platt_fit(shifted[dev], y[dev])
    fixed = ps.platt_apply(platt, shifted[test])
    assert ps.ece_score(fixed, y[test]) < 0.5 * ps.ece_score(shifted[test], y[test])
    assert ps.platt_fit(np.array([0.2, 0.3]), np.array([1, 1])) is None      # one class -> cannot fit


# ---------------------------------------------------------------- accuracy-coverage areas
def test_aurc_perfect_ranking_matches_oracle_and_random_is_the_error_rate():
    correct = np.array([1, 1, 1, 1, 0, 0])
    perfect = ps.aurc_metrics(np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]), correct)
    assert perfect["aurc"] == pytest.approx(perfect["oracle_aurc"]) and perfect["random_aurc"] == pytest.approx(2 / 6)
    assert perfect["auacc"] == pytest.approx(1 - perfect["aurc"])
    # hand value: risks at k=1..6 are 0,0,0,0,1/5,2/6 -> mean 0.0888...
    assert perfect["aurc"] == pytest.approx((0.2 + 2 / 6) / 6)
    worst = ps.aurc_metrics(np.array([0.9, 0.8, 0.3, 0.4, 0.1, 0.2]), correct)   # errors ranked as most trusted
    assert worst["aurc"] > perfect["aurc"]
    assert np.isnan(ps.aurc_metrics([], [])["aurc"])
