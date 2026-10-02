"""Bregman-Wasserstein constraint in survival-function form for Problem 2.2.

Implements the survival representation of the BW divergence and the associated
admissibility constraint Psi(S) <= zeta, used when optimizing over survival
functions S in the entropy-generator setting.

Reference: "Distributionally Robust Insurance under Bregman-Wasserstein Divergence"
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Constant term C0
# ---------------------------------------------------------------------------

def compute_C0(
    S0_func: Callable,
    phi_second_func: Callable,
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Compute the constant C0 = integral_0^M phi''(x)*x*S0(x) dx.

    This term is independent of S and separates from Psi(S) in the BW
    survival representation.

    Parameters
    ----------
    S0_func : callable
        Benchmark survival function S0(x).
    phi_second_func : callable
        Second derivative phi''(x) of the generator.
    M : float
        Support upper bound.
    quad_abs_tol, quad_rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    float
        C0 value.
    """
    def integrand(x: float) -> float:
        return float(phi_second_func(x)) * x * float(S0_func(x))

    val, _ = quad(integrand, 0.0, M, limit=200,
                  epsabs=quad_abs_tol, epsrel=quad_rel_tol)
    return float(val)


def compute_zeta(
    S0_func: Callable,
    phi_second_func: Callable,
    epsilon: float,
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Compute zeta = epsilon - C0.

    The BW constraint BW[F, F0] <= epsilon in survival form becomes
    Psi(S) <= zeta where zeta = epsilon - C0.

    Parameters
    ----------
    S0_func : callable
        Benchmark survival function.
    phi_second_func : callable
        Second derivative of the generator.
    epsilon : float
        BW radius.
    M : float
        Support upper bound.
    quad_abs_tol, quad_rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    float
        zeta value.
    """
    C0 = compute_C0(S0_func, phi_second_func, M, quad_abs_tol, quad_rel_tol)
    return float(epsilon - C0)


# ---------------------------------------------------------------------------
# Inner integral
# ---------------------------------------------------------------------------

def inner_integral_phi_second(
    x_val: float,
    S_val: float,
    S0_func: Callable,
    phi_second_func: Callable,
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Compute integral_0^M phi''(y) * min(S0(y), S_val) dy.

    Used in the double-integral term of Psi(S).

    Parameters
    ----------
    x_val : float
        The x-point (only used logically; S_val = S(x_val)).
    S_val : float
        Value of S at x_val.
    S0_func : callable
        Benchmark survival S0(y).
    phi_second_func : callable
        phi''(y).
    M : float
        Support upper bound.
    quad_abs_tol, quad_rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    float
        Value of the inner integral.
    """
    def integrand(y: float) -> float:
        return float(phi_second_func(y)) * min(float(S0_func(y)), S_val)

    val, _ = quad(integrand, 0.0, M, limit=200,
                  epsabs=quad_abs_tol, epsrel=quad_rel_tol)
    return float(val)


# ---------------------------------------------------------------------------
# Psi(S)
# ---------------------------------------------------------------------------

def compute_psi(
    S_func: Callable,
    S0_func: Callable,
    phi_prime_func: Callable,
    phi_prime_0: float,
    phi_second_func: Callable,
    M: float,
    x_grid: Optional[np.ndarray] = None,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Compute Psi(S) = integral_0^M [phi'(x)-phi'(0)] S(x) dx
                       - integral_0^M integral_0^M phi''(y) min(S0(y), S(x)) dy dx.

    Parameters
    ----------
    S_func : callable
        Candidate survival function S(x).
    S0_func : callable
        Benchmark survival S0(x).
    phi_prime_func : callable
        phi'(x).
    phi_prime_0 : float
        phi'(0).
    phi_second_func : callable
        phi''(x).
    M : float
        Support upper bound.
    x_grid : np.ndarray, optional
        If provided, discretize S_func on this grid for the double integral.
    quad_abs_tol, quad_rel_tol : float
        Quadrature tolerances.

    Returns
    -------
    float
        Psi(S) value.
    """
    # Term 1: integral_0^M [phi'(x)-phi'(0)] S(x) dx
    def term1_integrand(x: float) -> float:
        return (float(phi_prime_func(x)) - phi_prime_0) * float(S_func(x))

    term1, _ = quad(term1_integrand, 0.0, M, limit=200,
                    epsabs=quad_abs_tol, epsrel=quad_rel_tol)

    # Term 2 (double integral): integral_0^M integral_0^M phi''(y) min(S0(y), S(x)) dy dx
    if x_grid is not None:
        # Discretize for efficiency
        n = len(x_grid)
        S_vals = np.array([float(S_func(x)) for x in x_grid])
        inner_vals = np.array([
            inner_integral_phi_second(x_grid[i], S_vals[i], S0_func,
                                      phi_second_func, M,
                                      quad_abs_tol, quad_rel_tol)
            for i in range(n)
        ])
        term2 = float(np.trapz(inner_vals, x_grid))
    else:
        def outer_integrand(x: float) -> float:
            s_val = float(S_func(x))
            return inner_integral_phi_second(x, s_val, S0_func,
                                             phi_second_func, M,
                                             quad_abs_tol, quad_rel_tol)
        term2, _ = quad(outer_integrand, 0.0, M, limit=100,
                        epsabs=quad_abs_tol * 10, epsrel=quad_rel_tol * 10)

    return float(term1 - term2)


# ---------------------------------------------------------------------------
# BW feasibility check
# ---------------------------------------------------------------------------

def check_bw_survival_feasibility(
    S_func: Callable,
    S0_func: Callable,
    phi_prime_func: Callable,
    phi_prime_0: float,
    phi_second_func: Callable,
    epsilon: float,
    M: float,
    x_grid: Optional[np.ndarray] = None,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
    tol: float = 1e-6,
) -> dict:
    """Check whether a candidate survival S satisfies BW[F, F0] <= epsilon.

    Returns
    -------
    dict
        Keys: 'psi', 'C0', 'bw_value', 'zeta', 'feasible', 'margin'.
    """
    C0 = compute_C0(S0_func, phi_second_func, M, quad_abs_tol, quad_rel_tol)
    zeta = epsilon - C0
    psi = compute_psi(S_func, S0_func, phi_prime_func, phi_prime_0,
                      phi_second_func, M, x_grid, quad_abs_tol, quad_rel_tol)
    bw_value = psi + C0
    return {
        "psi": psi,
        "C0": C0,
        "bw_value": bw_value,
        "zeta": zeta,
        "feasible": bool(bw_value <= epsilon + tol),
        "margin": float(epsilon - bw_value),
    }
