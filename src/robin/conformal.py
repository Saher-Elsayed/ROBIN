"""Split-conformal calibration and sign-off rule for FPGA timing closure.

Reference: Vovk, Gammerman, Shafer, "Algorithmic Learning in a Random World"
(2nd ed., Springer 2022); Angelopoulos & Bates, "A gentle introduction to
conformal prediction and distribution-free uncertainty quantification" (2023).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    """Output of split-conformal calibration."""

    q_alpha: float
    n: int
    alpha: float
    cal_residual_mean: float
    cal_residual_std: float


@dataclass
class SignoffDecision:
    """Per-design sign-off envelope and accept/reject decision."""

    point_estimate: float
    lower: float
    upper: float
    accepted: bool
    margin: float
    alpha: float


class ConformalSignoff:
    """Split-conformal predictor over post-route WNS residuals.

    Calibrate once on a held-out set of (design_features, true_WNS) pairs;
    apply at sign-off time. The acceptance rule:

        accept iff  WNS_hat(z) - q_{1-alpha} >= 0

    yields the distribution-free coverage guarantee

        Pr[ true WNS in [WNS_hat - q, WNS_hat + q] ] >= 1 - alpha

    under exchangeability of the calibration set and the test point.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.q_alpha: float | None = None
        self.calibration_size: int | None = None
        self._cal_hash: str | None = None

    def calibrate(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> CalibrationResult:
        """Compute the split-conformal quantile from a calibration set.

        Parameters
        ----------
        predictions : np.ndarray
            Point estimates WNS_hat(z_i) from the slack-regression head.
        observations : np.ndarray
            Measured post-route WNS values aligned with predictions.

        Returns
        -------
        CalibrationResult
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        observations = np.asarray(observations, dtype=np.float64)
        if predictions.shape != observations.shape or predictions.ndim != 1:
            raise ValueError("predictions and observations must be 1-D arrays of equal length")

        residuals = np.abs(observations - predictions)
        n = residuals.size
        if n < 20:
            raise ValueError(f"calibration set too small (n={n}); need n>=20 for stable quantile")

        # split-conformal quantile of order ceil((n+1)(1-alpha)) / n
        rank = int(math.ceil((n + 1) * (1 - self.alpha)))
        rank = min(rank, n)
        sorted_resid = np.sort(residuals)
        self.q_alpha = float(sorted_resid[rank - 1])
        self.calibration_size = int(n)

        # record a hash so we can detect calibration-set drift at sign-off time
        cal_bytes = np.concatenate([predictions, observations]).tobytes()
        self._cal_hash = hashlib.sha256(cal_bytes).hexdigest()[:16]

        return CalibrationResult(
            q_alpha=self.q_alpha,
            n=self.calibration_size,
            alpha=self.alpha,
            cal_residual_mean=float(residuals.mean()),
            cal_residual_std=float(residuals.std(ddof=1)),
        )

    def signoff(self, wns_hat: float) -> SignoffDecision:
        """Apply the sign-off rule to a single point estimate.

        Returns the envelope and accept/reject decision.
        """
        if self.q_alpha is None:
            raise RuntimeError("calibrate() must be called before signoff()")

        lower = wns_hat - self.q_alpha
        upper = wns_hat + self.q_alpha
        accepted = lower >= 0.0
        return SignoffDecision(
            point_estimate=float(wns_hat),
            lower=float(lower),
            upper=float(upper),
            accepted=bool(accepted),
            margin=float(lower),
            alpha=self.alpha,
        )

    def empirical_coverage(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Compute empirical coverage on a held-out set after calibration."""
        if self.q_alpha is None:
            raise RuntimeError("calibrate() must be called before empirical_coverage()")

        predictions = np.asarray(predictions, dtype=np.float64)
        observations = np.asarray(observations, dtype=np.float64)
        inside = np.abs(observations - predictions) <= self.q_alpha
        return float(inside.mean())

    def __repr__(self) -> str:
        if self.q_alpha is None:
            return f"ConformalSignoff(alpha={self.alpha}, uncalibrated)"
        return (
            f"ConformalSignoff(alpha={self.alpha}, q={self.q_alpha:.4f}, "
            f"n={self.calibration_size}, cal_hash={self._cal_hash})"
        )
