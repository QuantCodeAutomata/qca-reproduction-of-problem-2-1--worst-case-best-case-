"""Validation utilities for distributions, indemnities, and optimality conditions.

These helpers are intentionally pure-numpy/pure-Python so that they can be
used as lightweight sanity checks inside any solver loop without introducing
heavy dependencies.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Array-level monotonicity helpers
# ---------------------------------------------------------------------------


def check_monotone_nondecreasing(arr: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if *arr* is non-decreasing up to *tol*.

    Parameters
    ----------
    arr : ndarray
        One-dimensional array of real values.
    tol : float
        Slack: consecutive pairs may decrease by at most *tol* before the
        check fails.

    Returns
    -------
    bool
        ``True`` iff :math:`a_{i+1} \\ge a_i - \\text{tol}` for all *i*.
    """
    arr = np.asarray(arr, dtype=float)
    return bool(np.all(np.diff(arr) >= -tol))


def check_monotone_nonincreasing(arr: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if *arr* is non-increasing up to *tol*.

    Parameters
    ----------
    arr : ndarray
        One-dimensional array of real values.
    tol : float
        Slack: consecutive pairs may increase by at most *tol* before the
        check fails.

    Returns
    -------
    bool
        ``True`` iff :math:`a_{i+1} \\le a_i + \\text{tol}` for all *i*.
    """
    arr = np.asarray(arr, dtype=float)
    return bool(np.all(np.diff(arr) <= tol))


# ---------------------------------------------------------------------------
# Distributional checks
# ---------------------------------------------------------------------------


def check_survival_function(S_values: np.ndarray, tol: float = 1e-8) -> bool:
    """Verify that *S_values* looks like a valid survival function.

    Checks:

    * All values are in [0, 1] (up to *tol*).
    * The sequence is non-increasing (up to *tol*).

    Parameters
    ----------
    S_values : ndarray
        Survival function evaluated on a grid, ordered from small to large *x*.
    tol : float
        Numerical tolerance.

    Returns
    -------
    bool
        ``True`` iff both conditions hold.
    """
    S_values = np.asarray(S_values, dtype=float)
    in_unit_interval = bool(np.all(S_values >= -tol) and np.all(S_values <= 1.0 + tol))
    nonincreasing = check_monotone_nonincreasing(S_values, tol=tol)
    return in_unit_interval and nonincreasing


# ---------------------------------------------------------------------------
# Indemnity admissibility
# ---------------------------------------------------------------------------


def check_admissible_indemnity(
    I_values: np.ndarray,
    x_grid: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """Check that an indemnity schedule is admissible.

    An indemnity :math:`I : [0, M] \\to [0, M]` is admissible when:

    * :math:`I(0) = 0`.
    * :math:`I` is non-decreasing.
    * The slope satisfies :math:`0 \\le I'(x) \\le 1` almost everywhere.

    The slope is estimated by finite differences on the supplied grid.

    Parameters
    ----------
    I_values : ndarray, shape (n,)
        Indemnity values at the points of *x_grid*.
    x_grid : ndarray, shape (n,)
        Strictly increasing loss grid.
    tol : float
        Numerical tolerance applied to all inequality checks.

    Returns
    -------
    bool
        ``True`` iff all admissibility conditions hold.
    """
    I_values = np.asarray(I_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    zero_at_origin = bool(abs(I_values[0]) <= tol)
    nondecreasing = check_monotone_nondecreasing(I_values, tol=tol)

    dx = np.diff(x_grid)
    dI = np.diff(I_values)
    slopes = dI / dx
    slope_lower = bool(np.all(slopes >= -tol))
    slope_upper = bool(np.all(slopes <= 1.0 + tol))

    return zero_at_origin and nondecreasing and slope_lower and slope_upper


# ---------------------------------------------------------------------------
# Optimality and feasibility checks
# ---------------------------------------------------------------------------


def check_bw_feasibility(
    bw_val: float,
    epsilon: float,
    tol: float = 1e-8,
) -> bool:
    """Check that a BW divergence value lies within the ambiguity ball.

    Parameters
    ----------
    bw_val : float
        Computed Bregman-Wasserstein divergence.
    epsilon : float
        Ambiguity radius.
    tol : float
        Numerical slack.

    Returns
    -------
    bool
        ``True`` iff :math:`D_{\\phi}(F \\| F_0) \\le \\epsilon + \\text{tol}`.
    """
    return bool(bw_val <= epsilon + tol)


def check_kkt_complementarity(
    lambda_val: float,
    constraint_slack: float,
    tol: float = 1e-6,
) -> bool:
    """Verify the KKT complementarity condition :math:`\\lambda \\cdot s = 0`.

    Parameters
    ----------
    lambda_val : float
        Lagrange multiplier (must be ≥ 0).
    constraint_slack : float
        Constraint slack :math:`s = \\epsilon - D_\\phi(F \\| F_0)` (must be ≥ 0).
    tol : float
        Tolerance for the product :math:`|\\lambda \\cdot s| \\le \\text{tol}`.

    Returns
    -------
    bool
        ``True`` iff :math:`\\lambda \\ge 0`, :math:`s \\ge 0`, and
        :math:`|\\lambda \\cdot s| \\le \\text{tol}`.
    """
    dual_feasible = bool(lambda_val >= -tol)
    primal_feasible = bool(constraint_slack >= -tol)
    complementarity = bool(abs(lambda_val * constraint_slack) <= tol)
    return dual_feasible and primal_feasible and complementarity


# ---------------------------------------------------------------------------
# Comprehensive F0 property validation
# ---------------------------------------------------------------------------


def validate_f0_properties(
    F0_func: Callable[[float], float],
    S0_func: Callable[[float], float],
    F0_inv_func: Callable[[float], float],
    alpha: float,
    M: float,
    n_points: int = 1_000,
) -> dict:
    """Validate mathematical properties of the base distribution triple.

    Tests performed:

    * Boundary values of CDF and survival function.
    * Monotonicity of CDF, survival function, and quantile function.
    * CDF + survival = 1 everywhere.
    * Round-trip consistency :math:`F_0(F_0^{-1}(t)) \\approx t`.
    * Quantile range contained in [0, M].
    * :math:`F_0^{-1}(\\alpha)` is a valid :math:`\\alpha`-quantile.

    Parameters
    ----------
    F0_func : callable
        CDF of the base distribution.
    S0_func : callable
        Survival function of the base distribution.
    F0_inv_func : callable
        Quantile (generalised inverse CDF) of the base distribution.
    alpha : float
        Confidence level, used to check the :math:`\\alpha`-quantile.
    M : float
        Upper bound of the loss support.
    n_points : int
        Number of evaluation points for grid-based checks.

    Returns
    -------
    dict
        Dictionary with the following boolean keys:

        * ``cdf_left_boundary``
        * ``cdf_right_boundary``
        * ``survival_left_boundary``
        * ``survival_right_boundary``
        * ``cdf_monotone``
        * ``survival_monotone``
        * ``quantile_monotone``
        * ``cdf_survival_sum``
        * ``inverse_roundtrip``
        * ``quantile_range``
        * ``alpha_quantile_consistency``
        * ``all_passed``
    """
    tol = 1e-8

    xs = np.linspace(0.0, M, n_points)
    ts = np.linspace(0.0, 1.0, n_points)

    cdf_vals = np.array([float(F0_func(x)) for x in xs], dtype=float)
    surv_vals = np.array([float(S0_func(x)) for x in xs], dtype=float)
    inv_vals = np.array([float(F0_inv_func(t)) for t in ts], dtype=float)

    q_alpha = float(F0_inv_func(alpha))
    cdf_at_q_alpha = float(F0_func(q_alpha))

    results: dict = {
        "cdf_left_boundary": bool(abs(float(F0_func(0.0))) < tol),
        "cdf_right_boundary": bool(abs(float(F0_func(M)) - 1.0) < tol),
        "survival_left_boundary": bool(abs(float(S0_func(0.0)) - 1.0) < tol),
        "survival_right_boundary": bool(abs(float(S0_func(M))) < tol),
        "cdf_monotone": check_monotone_nondecreasing(cdf_vals, tol=tol),
        "survival_monotone": check_monotone_nonincreasing(surv_vals, tol=tol),
        "quantile_monotone": check_monotone_nondecreasing(inv_vals, tol=tol),
        "cdf_survival_sum": bool(np.allclose(cdf_vals + surv_vals, 1.0, atol=tol)),
        "inverse_roundtrip": bool(
            np.allclose(
                np.array([float(F0_func(q)) for q in inv_vals], dtype=float),
                ts,
                atol=tol,
            )
        ),
        "quantile_range": bool(np.all(inv_vals >= -tol) and np.all(inv_vals <= M + tol)),
        "alpha_quantile_consistency": bool(cdf_at_q_alpha >= alpha - tol),
    }
    results["all_passed"] = all(results.values())
    return results
