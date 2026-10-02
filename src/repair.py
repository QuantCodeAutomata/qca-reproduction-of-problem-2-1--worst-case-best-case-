"""Flat-segment repair for monotonicity correction of candidate survival functions.

When the relaxed optimizer G*(x;beta) has an upward jump at overline(V_alpha),
this module repairs it by introducing a flat segment as described in Theorem 4.1.

Reference: "Distributionally Robust Insurance under Bregman-Wasserstein Divergence"
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def detect_upward_jump(
    G_star_values: np.ndarray,
    x_grid: np.ndarray,
    V_alpha_upper: float,
    tol: float = 1e-8,
) -> bool:
    """Check if G*(x;beta) has an upward jump at overline(V_alpha).

    Parameters
    ----------
    G_star_values : np.ndarray
        G*(x) evaluated on x_grid.
    x_grid : np.ndarray
        Grid of x values.
    V_alpha_upper : float
        The threshold overline(V_alpha).
    tol : float
        Tolerance for detecting a jump (upward increase).

    Returns
    -------
    bool
        True if an upward jump is detected at V_alpha_upper.
    """
    # Find index closest to V_alpha_upper
    idx = np.searchsorted(x_grid, V_alpha_upper)
    if idx <= 0 or idx >= len(x_grid):
        return False

    # Check jump: G*(V_alpha_upper+) > G*(V_alpha_upper-)
    left_val = G_star_values[max(0, idx - 1)]
    right_val = G_star_values[min(len(x_grid) - 1, idx)]
    return bool(right_val > left_val + tol)


def repair_survival_flat_segment(
    G_star_values: np.ndarray,
    V_alpha_upper: float,
    b: float,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Construct the repaired survival function S(x;b).

    S(x;b) = max(G*(x;beta), b)  for x in [0, V_alpha_upper]
    S(x;b) = min(G*(x;beta), b)  for x in (V_alpha_upper, M]

    This introduces a flat segment at level b around overline(V_alpha),
    ensuring monotonicity.

    Parameters
    ----------
    G_star_values : np.ndarray
        G*(x;beta) values on x_grid.
    V_alpha_upper : float
        The threshold overline(V_alpha).
    b : float
        Flat-segment level b in [0, 1].
    x_grid : np.ndarray
        Grid of x values.

    Returns
    -------
    np.ndarray
        Repaired S(x;b) values on x_grid.
    """
    S = np.empty_like(G_star_values)
    for i, x in enumerate(x_grid):
        if x <= V_alpha_upper:
            S[i] = max(G_star_values[i], b)
        else:
            S[i] = min(G_star_values[i], b)
    return S


def _admissible_b_interval(
    G_star_values: np.ndarray,
    x_grid: np.ndarray,
    V_alpha_upper: float,
) -> Tuple[float, float]:
    """Compute the admissible interval O_b(beta) for the flat-segment level b.

    The flat segment at level b is admissible when:
    - b <= G*(x) for all x < V_alpha_upper in the region where G* exceeds b (left side)
    - b >= G*(x) for all x > V_alpha_upper (right side)

    In practice, a safe choice is:
    b_lo = G*(V_alpha_upper+)  (just above the threshold)
    b_hi = G*(V_alpha_upper-)  (just below the threshold)

    Parameters
    ----------
    G_star_values : np.ndarray
        G*(x;beta) on x_grid.
    x_grid : np.ndarray
        Grid of x values.
    V_alpha_upper : float
        Threshold overline(V_alpha).

    Returns
    -------
    (b_lo, b_hi) : (float, float)
        Lower and upper bounds of admissible b.
    """
    idx = np.searchsorted(x_grid, V_alpha_upper)
    idx = np.clip(idx, 1, len(x_grid) - 1)

    # Left limit: G*(V_alpha_upper-)
    G_left = G_star_values[idx - 1]
    # Right limit: G*(V_alpha_upper+)
    G_right = G_star_values[min(idx, len(x_grid) - 1)]

    b_lo = min(G_left, G_right)
    b_hi = max(G_left, G_right)

    # Ensure a non-degenerate interval
    if b_hi - b_lo < 1e-12:
        b_lo = max(0.0, b_hi - 1e-8)

    return float(b_lo), float(b_hi)


def find_optimal_b(
    G_star_values: np.ndarray,
    V_alpha_upper: float,
    x_grid: np.ndarray,
    objective_func: Callable[[np.ndarray], float],
    tol: float = 1e-8,
) -> Tuple[float, np.ndarray]:
    """Find b* = argmax_{b in O_b(beta)} L(S(x;b);beta).

    Parameters
    ----------
    G_star_values : np.ndarray
        G*(x;beta) on x_grid.
    V_alpha_upper : float
        Threshold overline(V_alpha).
    x_grid : np.ndarray
        Grid of x values.
    objective_func : callable
        Objective functional L(S) -> float (to be maximized).
    tol : float
        Tolerance for the scalar optimizer.

    Returns
    -------
    (b_star, S_star_values) : (float, np.ndarray)
        Optimal b and corresponding repaired survival function.
    """
    b_lo, b_hi = _admissible_b_interval(G_star_values, x_grid, V_alpha_upper)

    if b_hi - b_lo < 1e-10:
        S_repaired = repair_survival_flat_segment(
            G_star_values, V_alpha_upper, b_lo, x_grid
        )
        return float(b_lo), S_repaired

    def neg_objective(b: float) -> float:
        S = repair_survival_flat_segment(G_star_values, V_alpha_upper, b, x_grid)
        return -objective_func(S)

    result = minimize_scalar(
        neg_objective,
        bounds=(b_lo, b_hi),
        method="bounded",
        options={"xatol": tol},
    )
    b_star = float(result.x)
    S_star = repair_survival_flat_segment(G_star_values, V_alpha_upper, b_star, x_grid)
    return b_star, S_star
