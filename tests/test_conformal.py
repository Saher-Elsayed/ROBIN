"""Unit tests for split-conformal sign-off."""

import numpy as np
import pytest

from robin.conformal import ConformalSignoff


def test_calibrate_returns_positive_quantile():
    rng = np.random.default_rng(0)
    pred = rng.normal(0.1, 0.05, 200)
    obs = pred + rng.normal(0, 0.05, 200)
    s = ConformalSignoff(alpha=0.05)
    res = s.calibrate(pred, obs)
    assert res.q_alpha > 0
    assert res.n == 200


def test_empirical_coverage_close_to_target():
    rng = np.random.default_rng(1)
    pred_cal = rng.normal(0.1, 0.05, 500)
    obs_cal = pred_cal + rng.normal(0, 0.05, 500)
    pred_ho = rng.normal(0.1, 0.05, 500)
    obs_ho = pred_ho + rng.normal(0, 0.05, 500)

    s = ConformalSignoff(alpha=0.05)
    s.calibrate(pred_cal, obs_cal)
    cov = s.empirical_coverage(pred_ho, obs_ho)
    assert 0.90 <= cov <= 1.0


def test_signoff_acceptance_rule():
    rng = np.random.default_rng(2)
    pred = rng.normal(0.2, 0.05, 200)
    obs = pred + rng.normal(0, 0.03, 200)
    s = ConformalSignoff(alpha=0.05)
    s.calibrate(pred, obs)

    # WNS_hat > q -> accepted
    decision_pos = s.signoff(wns_hat=0.5)
    assert decision_pos.accepted

    # WNS_hat = 0 -> lower < 0 -> rejected
    decision_neg = s.signoff(wns_hat=0.0)
    assert not decision_neg.accepted
    assert decision_neg.lower < 0


def test_signoff_before_calibrate_raises():
    s = ConformalSignoff(alpha=0.05)
    with pytest.raises(RuntimeError):
        s.signoff(wns_hat=0.5)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        ConformalSignoff(alpha=1.5)
    with pytest.raises(ValueError):
        ConformalSignoff(alpha=0.0)


def test_calibration_too_small():
    s = ConformalSignoff(alpha=0.05)
    with pytest.raises(ValueError):
        s.calibrate(np.zeros(5), np.zeros(5))
