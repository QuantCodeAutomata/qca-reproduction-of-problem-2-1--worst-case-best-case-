"""Problem 2.1: Worst-case and best-case VaR under Bregman-Wasserstein ambiguity.

Implements Proposition 3.1 (worst-case VaR), Proposition 3.2 (best-case VaR),
the marginal indemnification rule, and the full Example 3.1 pipeline.

Reference: "Distributionally Robust Insurance under Bregman-Wasserstein Divergence"
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

from src.distributions import F0, S0, F0_inv
from src.bregman import B_phi, phi, phi_prime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bw_integral_upper(D: float, F0_inv_fn: Callable, q_alpha: float,
                        alpha: float, k: float, M: float,
                        quad_abs_tol: float, quad_rel_tol: float) -> float:
    """Compute integral_{alpha}^{F0(D)} B_phi(D, F0_inv(t)) dt for Proposition 3.1."""
    upper_t = min(F0(D, M), 1.0 - 1e-12)
    if upper_t <= alpha + 1e-14:
        return 0.0

    def integrand(t: float) -> float:
        return float(B_phi(D, F0_inv_fn(t), q_alpha, k))

    val, _ = quad(integrand, alpha, upper_t,
                  limit=200, epsabs=quad_abs_tol, epsrel=quad_rel_tol)
    return float(val)


def _bw_integral_lower(D: float, F0_inv_fn: Callable, F0_fn: Callable,
                        q_alpha: float, alpha: float, k: float, M: float,
                        quad_abs_tol: float, quad_rel_tol: float) -> float:
    """Compute integral_{F0(D)}^{alpha} B_phi(D, F0_inv(t)) dt for Proposition 3.2."""
    lower_t = max(F0_fn(D, M), 1e-12)
    if lower_t >= alpha - 1e-14:
        return 0.0

    def integrand(t: float) -> float:
        return float(B_phi(D, F0_inv_fn(t), q_alpha, k))

    val, _ = quad(integrand, lower_t, alpha,
                  limit=200, epsabs=quad_abs_tol, epsrel=quad_rel_tol)
    return float(val)


# ---------------------------------------------------------------------------
# Worst-case VaR (Proposition 3.1)
# ---------------------------------------------------------------------------

def compute_worst_case_var(
    F0_inv_fn: Callable,
    q_alpha: float,
    alpha: float,
    k: float,
    epsilon: float,
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
    root_tol: float = 1e-10,
) -> float:
    """Compute overline(V_alpha) = sup VaR_alpha^F(X) over the BW ball.

    Implements Proposition 3.1:
    1. Compute T_M = integral_alpha^1 B_phi(M, F0_inv(t)) dt.
    2. If epsilon >= T_M: return M.
    3. Otherwise find smallest D in (q_alpha, M] s.t.
       integral_alpha^{F0(D)} B_phi(D, F0_inv(t)) dt >= epsilon.

    Parameters
    ----------
    F0_inv_fn : callable
        Quantile function F0^{-1}: (0,1) -> [0, M].
    q_alpha : float
        VaR_alpha under F0.
    alpha : float
        Confidence level.
    k : float
        Generator curvature.
    epsilon : float
        BW ambiguity radius.
    M : float
        Support upper bound.
    quad_abs_tol, quad_rel_tol, root_tol : float
        Numerical tolerances.

    Returns
    -------
    float
        Worst-case VaR (supremum, not attained if < M).
    """
    # T_M = integral_alpha^1 B_phi(M, F0_inv(t)) dt
    def tm_integrand(t: float) -> float:
        return float(B_phi(M, F0_inv_fn(t), q_alpha, k))

    T_M, _ = quad(tm_integrand, alpha, 1.0 - 1e-12,
                  limit=200, epsabs=quad_abs_tol, epsrel=quad_rel_tol)

    if epsilon >= T_M:
        return float(M)

    # target(D) = integral_alpha^{F0(D)} B_phi(D, F0_inv(t)) dt - epsilon
    def target(D: float) -> float:
        return _bw_integral_upper(D, F0_inv_fn, q_alpha, alpha, k, M,
                                  quad_abs_tol, quad_rel_tol) - epsilon

    # Search in (q_alpha, M)
    lo = q_alpha + 1e-6
    hi = M - 1e-6

    val_lo = target(lo)
    val_hi = target(hi)

    # If val_lo > 0, D is near q_alpha; if val_hi < 0, epsilon is large
    if val_hi < 0:
        return float(M)

    # Bracket: find crossing from negative to positive
    if val_lo >= 0:
        # Very small epsilon case — D is very close to q_alpha
        # Scan from lo towards hi to find where target first goes negative
        scan = np.linspace(lo, hi, 500)
        for d in scan:
            if target(d) >= 0:
                lo = max(q_alpha + 1e-9, d - (hi - lo) / 500)
                break
        val_lo = target(lo)

    D_star = brentq(target, lo, hi, xtol=root_tol, rtol=root_tol, maxiter=500)
    return float(D_star)


# ---------------------------------------------------------------------------
# Best-case VaR (Proposition 3.2)
# ---------------------------------------------------------------------------

def compute_best_case_var(
    F0_inv_fn: Callable,
    F0_fn: Callable,
    q_alpha: float,
    alpha: float,
    k: float,
    epsilon: float,
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
    root_tol: float = 1e-10,
) -> float:
    """Compute underline(V_alpha) = inf VaR_alpha^F(X) over the BW ball.

    Implements Proposition 3.2: infimum is attained.
    Find smallest D in [0, q_alpha) s.t.
    integral_{F0(D)}^alpha B_phi(D, F0_inv(t)) dt = epsilon.

    If the integral at D=0 is already <= epsilon, return 0.

    Parameters
    ----------
    F0_inv_fn : callable
        Quantile function F0^{-1}.
    F0_fn : callable
        CDF F0 as a 2-argument callable: F0_fn(x, M).
    q_alpha : float
        VaR_alpha under F0.
    alpha : float
        Confidence level.
    k : float
        Generator curvature.
    epsilon : float
        BW ambiguity radius.
    M : float
        Support upper bound.
    quad_abs_tol, quad_rel_tol, root_tol : float
        Numerical tolerances.

    Returns
    -------
    float
        Best-case VaR (infimum, attained).
    """
    def target(D: float) -> float:
        return _bw_integral_lower(D, F0_inv_fn, F0_fn, q_alpha, alpha, k, M,
                                  quad_abs_tol, quad_rel_tol) - epsilon

    val_at_0 = target(0.0)
    val_at_q = target(q_alpha - 1e-6)

    if val_at_0 <= 0.0:
        return 0.0

    if val_at_q >= 0.0:
        # Integral is always >= epsilon in [0, q_alpha); return q_alpha
        return float(q_alpha - 1e-6)

    D_star = brentq(target, 0.0, q_alpha - 1e-6,
                    xtol=root_tol, rtol=root_tol, maxiter=500)
    return float(D_star)


# ---------------------------------------------------------------------------
# Indemnity thresholds
# ---------------------------------------------------------------------------

def compute_indemnity_thresholds(
    S_Q: Callable,
    theta: float,
    kappa: float,
    V_alpha_lower: float,
    V_alpha_upper: float,
    M: float,
    root_tol: float = 1e-10,
) -> tuple[float, float]:
    """Compute the coverage thresholds d1* and d2*.

    d1* = inf{x in [0,M]: S_Q(x) <= 1/(1+theta)}
    d2* = inf{x in [0,M]: S_Q(x) <= kappa/(1+theta)}

    Under S_Q = S0 (truncated exponential), these have closed forms:
    S0(x) = c => x = -log(c*(1-exp(-M)) + exp(-M))

    Parameters
    ----------
    S_Q : callable
        Insurer survival function S_Q(x).
    theta : float
        Premium loading factor.
    kappa : float
        Ambiguity weight.
    V_alpha_lower, V_alpha_upper : float
        Best and worst-case VaR (used for bounding the search if needed).
    M : float
        Support upper bound.
    root_tol : float
        Root-finding tolerance.

    Returns
    -------
    (d1_star, d2_star) : (float, float)
    """
    # Analytical formula for truncated exponential: S0(x)=c => x=-log(c*(1-e^{-M})+e^{-M})
    nc = 1.0 - np.exp(-M)

    def invert_S0(c: float) -> float:
        arg = c * nc + np.exp(-M)
        arg = np.clip(arg, 1e-300, 1.0)
        return float(-np.log(arg))

    c1 = 1.0 / (1.0 + theta)
    c2 = kappa / (1.0 + theta)

    # Analytical inversion (valid for S_Q = S0)
    d1_star = invert_S0(c1)
    d2_star = invert_S0(c2)

    # Validate via numerical inversion as fallback
    def _check_or_solve(c: float, name: str) -> float:
        # Verify analytical answer
        S_at_d = S_Q(invert_S0(c))
        if abs(S_at_d - c) < 1e-6:
            return invert_S0(c)
        # Fall back to numerical root
        def f(x: float) -> float:
            return float(S_Q(x)) - c
        return float(brentq(f, 0.0, M - 1e-8, xtol=root_tol))

    d1_star = _check_or_solve(c1, "d1*")
    d2_star = _check_or_solve(c2, "d2*")

    return float(d1_star), float(d2_star)


# ---------------------------------------------------------------------------
# Net price function H
# ---------------------------------------------------------------------------

def compute_net_price_H(
    x: float,
    S_Q: Callable,
    V_alpha_lower: float,
    V_alpha_upper: float,
    kappa: float,
    theta: float,
) -> float:
    """Compute the net price function H(x).

    H(t) = (1+theta)*S_Q(t)
           - 1_{[0, underline(V_alpha)]}(t)
           - kappa * 1_{(underline(V_alpha), overline(V_alpha)]}(t)

    Equivalently:
    H(t) = (1+theta)*S_Q(t) - 1_{[0, V_lower]}(t) - kappa * 1_{(V_lower, V_upper]}(t)

    Parameters
    ----------
    x : float
        Evaluation point.
    S_Q : callable
        Insurer survival function.
    V_alpha_lower : float
        Best-case VaR (underline{V_alpha}).
    V_alpha_upper : float
        Worst-case VaR (overline{V_alpha}).
    kappa : float
        Ambiguity weight.
    theta : float
        Premium loading.

    Returns
    -------
    float
        H(x) value.
    """
    premium_term = (1.0 + theta) * float(S_Q(x))
    indicator_lower = 1.0 if x <= V_alpha_lower else 0.0
    indicator_upper = kappa if (x > V_alpha_lower and x <= V_alpha_upper) else 0.0
    return premium_term - indicator_lower - indicator_upper


# ---------------------------------------------------------------------------
# Optimal indemnity
# ---------------------------------------------------------------------------

def compute_optimal_indemnity(
    x_grid: np.ndarray,
    S_Q: Callable,
    V_alpha_lower: float,
    V_alpha_upper: float,
    kappa: float,
    theta: float,
    M: float,
) -> np.ndarray:
    """Construct the optimal indemnity I*(x) by the marginal indemnification rule.

    I*'(x) = 1 if H(x) < 0
    I*'(x) = 0 if H(x) > 0
    I*'(x) = 1 (eta convention) if H(x) = 0

    Integrate I*'(x) to get I*(x).

    Parameters
    ----------
    x_grid : np.ndarray
        Grid points in [0, M], starting at 0.
    S_Q : callable
        Insurer survival function.
    V_alpha_lower : float
        Best-case VaR.
    V_alpha_upper : float
        Worst-case VaR.
    kappa : float
        Ambiguity weight.
    theta : float
        Premium loading.
    M : float
        Support upper bound.

    Returns
    -------
    np.ndarray
        I*(x) values on x_grid, shape (len(x_grid),).
    """
    H_vals = np.array([
        compute_net_price_H(x, S_Q, V_alpha_lower, V_alpha_upper, kappa, theta)
        for x in x_grid
    ])
    I_prime = np.where(H_vals <= 0.0, 1.0, 0.0)

    # Integrate via trapezoid: I*(x) = integral_0^x I*'(t) dt
    dx = np.diff(x_grid)
    I_vals = np.zeros(len(x_grid))
    for i in range(1, len(x_grid)):
        I_vals[i] = I_vals[i - 1] + 0.5 * (I_prime[i - 1] + I_prime[i]) * dx[i - 1]

    return I_vals


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def compute_objective_21(
    I_values: np.ndarray,
    x_grid: np.ndarray,
    S_Q: Callable,
    V_alpha_lower: float,
    V_alpha_upper: float,
    kappa: float,
    theta: float,
) -> float:
    """Compute the alpha-maxmin VaR objective for Problem 2.1.

    Objective = kappa * (V_upper - I*(V_upper))
               + (1-kappa) * (V_lower - I*(V_lower))
               + pi(I(X))

    where pi(I(X)) = (1+theta) * integral_0^M S_Q(x) * I*'(x) dx.

    Parameters
    ----------
    I_values : np.ndarray
        I*(x) values on x_grid.
    x_grid : np.ndarray
        Grid points.
    S_Q : callable
        Insurer survival function.
    V_alpha_lower : float
        Best-case VaR.
    V_alpha_upper : float
        Worst-case VaR.
    kappa : float
        Ambiguity weight.
    theta : float
        Premium loading.

    Returns
    -------
    float
        Objective value.
    """
    # Interpolate I* at VaR levels
    I_at_upper = float(np.interp(V_alpha_upper, x_grid, I_values))
    I_at_lower = float(np.interp(V_alpha_lower, x_grid, I_values))

    # Compute derivative I*' via finite differences
    I_prime = np.gradient(I_values, x_grid)

    # Premium
    S_Q_vals = np.array([float(S_Q(x)) for x in x_grid])
    premium = (1.0 + theta) * float(np.trapz(S_Q_vals * I_prime, x_grid))

    obj = (kappa * (V_alpha_upper - I_at_upper)
           + (1.0 - kappa) * (V_alpha_lower - I_at_lower)
           + premium)
    return float(obj)


# ---------------------------------------------------------------------------
# Example 3.1 pipeline
# ---------------------------------------------------------------------------

def run_example31(
    k_values: List[float] = None,
    M: float = 100.0,
    alpha: float = 0.95,
    theta: float = 0.5,
    kappa: float = 0.9,
    epsilon: float = 0.5,
    n_grid: int = 2000,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
    root_tol: float = 1e-10,
) -> pd.DataFrame:
    """Run Example 3.1 for each k value and return a summary DataFrame.

    Parameters
    ----------
    k_values : list of float
        Curvature parameters (default [1, 2, 3, 4]).
    M, alpha, theta, kappa, epsilon : float
        Problem parameters.
    n_grid : int
        Number of grid points for indemnity integration.
    quad_abs_tol, quad_rel_tol, root_tol : float
        Numerical tolerances.

    Returns
    -------
    pd.DataFrame
        Columns: k, q_alpha, V_lower, V_upper, d1_star, d2_star, premium, objective.
    """
    if k_values is None:
        k_values = [1, 2, 3, 4]

    rows = []
    x_grid = np.linspace(0.0, M, n_grid)

    for k in k_values:
        q_alpha = float(F0_inv(alpha, M))

        V_upper = compute_worst_case_var(
            lambda t: float(F0_inv(t, M)), q_alpha, alpha, k, epsilon, M,
            quad_abs_tol, quad_rel_tol, root_tol
        )
        V_lower = compute_best_case_var(
            lambda t: float(F0_inv(t, M)), F0, q_alpha, alpha, k, epsilon, M,
            quad_abs_tol, quad_rel_tol, root_tol
        )

        d1_star, d2_star = compute_indemnity_thresholds(
            lambda x: float(S0(x, M)), theta, kappa, V_lower, V_upper, M, root_tol
        )

        I_vals = compute_optimal_indemnity(
            x_grid, lambda x: float(S0(x, M)), V_lower, V_upper, kappa, theta, M
        )

        # Premium
        I_prime_vals = np.gradient(I_vals, x_grid)
        S_Q_vals = np.array([float(S0(x, M)) for x in x_grid])
        prem = (1.0 + theta) * float(np.trapz(S_Q_vals * I_prime_vals, x_grid))

        obj = compute_objective_21(
            I_vals, x_grid, lambda x: float(S0(x, M)),
            V_lower, V_upper, kappa, theta
        )

        rows.append({
            "k": k,
            "q_alpha": q_alpha,
            "V_lower": V_lower,
            "V_upper": V_upper,
            "d1_star": d1_star,
            "d2_star": d2_star,
            "premium": prem,
            "objective": obj,
        })

    return pd.DataFrame(rows)
