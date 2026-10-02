"""Truncated exponential distribution on [0, M] and configuration dataclass.

The base distribution F0 is a truncated Exp(1) on [0, M].  All public
functions accept both scalar and array-valued inputs via numpy broadcasting.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Union

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Centralised parameter store for distributionally robust insurance experiments.

    Attributes
    ----------
    M : float
        Upper bound of the loss support [0, M].
    alpha : float
        Confidence level for TVaR / quantile risk measures (e.g. 0.95).
    theta : float
        Safety loading factor for the insurance premium (e.g. 0.5).
    kappa : float
        Indemnity slope cap above the quantile threshold (e.g. 0.9).
    epsilon : float
        Bregman-Wasserstein ambiguity radius (e.g. 0.5).
    quad_abs_tol : float
        Absolute tolerance passed to ``scipy.integrate.quad``.
    quad_rel_tol : float
        Relative tolerance passed to ``scipy.integrate.quad``.
    root_tol : float
        Tolerance for root-finding routines (e.g. ``scipy.optimize.brentq``).
    monotonicity_tol : float
        Slack allowed when checking monotonicity of numerical arrays.
    """

    M: float = 100.0
    alpha: float = 0.95
    theta: float = 0.5
    kappa: float = 0.9
    epsilon: float = 0.5
    quad_abs_tol: float = 1e-10
    quad_rel_tol: float = 1e-10
    root_tol: float = 1e-10
    monotonicity_tol: float = 1e-8


# ---------------------------------------------------------------------------
# Truncated exponential distribution helpers
# ---------------------------------------------------------------------------


def _normalising_constant(M: float) -> float:
    """Return 1 - exp(-M), the normalising constant for Exp(1) truncated to [0, M]."""
    return 1.0 - np.exp(-M)


def F0(x: ArrayLike, M: float = 100.0) -> ArrayLike:
    """CDF of the truncated Exp(1) distribution on [0, M].

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).  Values outside [0, M] are clamped to the
        boundary values 0 and 1 respectively.
    M : float
        Upper truncation point (default 100).

    Returns
    -------
    float or ndarray
        CDF value(s) in [0, 1].

    Notes
    -----
    .. math::
        F_0(x) = \\frac{1 - e^{-x}}{1 - e^{-M}}, \\quad x \\in [0, M].
    """
    x = np.asarray(x, dtype=float)
    nc = _normalising_constant(M)
    result = (1.0 - np.exp(-x)) / nc
    return float(np.clip(result, 0.0, 1.0)) if result.ndim == 0 else np.clip(result, 0.0, 1.0)


def S0(x: ArrayLike, M: float = 100.0) -> ArrayLike:
    """Survival function of the truncated Exp(1) distribution on [0, M].

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).
    M : float
        Upper truncation point (default 100).

    Returns
    -------
    float or ndarray
        Survival value(s) in [0, 1].

    Notes
    -----
    .. math::
        S_0(x) = \\frac{e^{-x} - e^{-M}}{1 - e^{-M}}, \\quad x \\in [0, M].
    """
    x = np.asarray(x, dtype=float)
    nc = _normalising_constant(M)
    result = (np.exp(-x) - np.exp(-M)) / nc
    return float(np.clip(result, 0.0, 1.0)) if result.ndim == 0 else np.clip(result, 0.0, 1.0)


def F0_inv(t: ArrayLike, M: float = 100.0) -> ArrayLike:
    """Left (generalised) quantile inverse of F0.

    Parameters
    ----------
    t : float or ndarray
        Probability level(s) in [0, 1].
    M : float
        Upper truncation point (default 100).

    Returns
    -------
    float or ndarray
        Quantile value(s) in [0, M].

    Notes
    -----
    Derived by inverting F0:

    .. math::
        F_0^{-1}(t) = -\\log\\!\\bigl(1 - t\\,(1 - e^{-M})\\bigr).
    """
    t = np.asarray(t, dtype=float)
    nc = _normalising_constant(M)
    argument = 1.0 - t * nc
    # Guard against floating-point noise pushing argument outside (0, 1]
    argument = np.clip(argument, np.exp(-M), 1.0)
    result = -np.log(argument)
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------


def validate_f0(M: float = 100.0, n_points: int = 1_000, tol: float = 1e-8) -> dict:
    """Validate boundary conditions and monotonicity of F0, S0, and F0_inv.

    Parameters
    ----------
    M : float
        Upper truncation point.
    n_points : int
        Number of interior evaluation points.
    tol : float
        Numerical tolerance for boundary checks.

    Returns
    -------
    dict
        Dictionary with boolean fields:

        * ``cdf_left_boundary`` – F0(0) ≈ 0
        * ``cdf_right_boundary`` – F0(M) ≈ 1
        * ``survival_left_boundary`` – S0(0) ≈ 1
        * ``survival_right_boundary`` – S0(M) ≈ 0
        * ``cdf_monotone`` – F0 is non-decreasing on [0, M]
        * ``survival_monotone`` – S0 is non-increasing on [0, M]
        * ``cdf_survival_sum`` – F0(x) + S0(x) ≈ 1 everywhere
        * ``inverse_roundtrip`` – F0(F0_inv(t)) ≈ t for t in (0, 1)
        * ``inverse_boundary_low`` – F0_inv(0) ≈ 0
        * ``inverse_boundary_high`` – F0_inv(1) ≈ M
        * ``all_passed`` – conjunction of all the above
    """
    xs = np.linspace(0.0, M, n_points)
    ts = np.linspace(0.0, 1.0, n_points)

    cdf_vals = F0(xs, M)
    surv_vals = S0(xs, M)
    inv_vals = F0_inv(ts, M)

    results = {
        "cdf_left_boundary": bool(abs(F0(0.0, M)) < tol),
        "cdf_right_boundary": bool(abs(F0(M, M) - 1.0) < tol),
        "survival_left_boundary": bool(abs(S0(0.0, M) - 1.0) < tol),
        "survival_right_boundary": bool(abs(S0(M, M)) < tol),
        "cdf_monotone": bool(np.all(np.diff(cdf_vals) >= -tol)),
        "survival_monotone": bool(np.all(np.diff(surv_vals) <= tol)),
        "cdf_survival_sum": bool(np.allclose(cdf_vals + surv_vals, 1.0, atol=tol)),
        "inverse_roundtrip": bool(np.allclose(F0(inv_vals, M), ts, atol=tol)),
        "inverse_boundary_low": bool(abs(F0_inv(0.0, M)) < tol),
        "inverse_boundary_high": bool(abs(F0_inv(1.0, M) - M) < tol),
    }
    results["all_passed"] = all(results.values())
    return results
