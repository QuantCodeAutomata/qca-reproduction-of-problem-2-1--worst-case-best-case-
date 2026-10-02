"""TVaR-specific analytical results of Section 4.3.

Implements the three-case classification based on (1+lambda)*(1+theta),
the explicit derivative K'(t;x), closed-form hatG(x;beta), and G*(x;beta)
for the entropy generator phi(x)=(x+1)*log(x+1) and TVaR distortion.

Reference: "Distributionally Robust Insurance under Bregman-Wasserstein Divergence"
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import quad

from src.distributions import F0, S0, F0_inv
from src.bregman import phi_entropy_prime, phi_entropy_prime_inv
from src.distortion import tvar_distortion
from src.problem21 import compute_worst_case_var


# ---------------------------------------------------------------------------
# Case classification
# ---------------------------------------------------------------------------

def case_type(
    lambda_val: float,
    theta: float,
    alpha: float,
    S_Q_at_V_upper: float,
) -> int:
    """Classify the regime for the TVaR G* formula.

    Classification criteria based on (1+lambda)*(1+theta):
    - Case (i):  value < 1/(1-alpha)
    - Case (ii): 1/(1-alpha) <= value < 1/S_Q(overline(V_alpha))
    - Case (iii): value >= 1/S_Q(overline(V_alpha))

    Parameters
    ----------
    lambda_val : float
        Lagrange multiplier >= 0.
    theta : float
        Premium loading.
    alpha : float
        Confidence level.
    S_Q_at_V_upper : float
        S_Q evaluated at overline(V_alpha).

    Returns
    -------
    int
        1, 2, or 3.
    """
    value = (1.0 + lambda_val) * (1.0 + theta)
    boundary1 = 1.0 / (1.0 - alpha)  # = 20 for alpha=0.95

    if S_Q_at_V_upper <= 1e-15:
        boundary2 = np.inf
    else:
        boundary2 = 1.0 / S_Q_at_V_upper

    if value < boundary1:
        return 1
    elif value < boundary2:
        return 2
    else:
        return 3


# ---------------------------------------------------------------------------
# Threshold x1 and x2
# ---------------------------------------------------------------------------

def compute_x1(
    V_alpha_upper: float,
    theta: float,
    S_Q_func: Callable,
    M: float,
    root_tol: float = 1e-10,
) -> float:
    """Compute x1 = sup{x in [0, overline(V_alpha)): S_Q(x) >= 1/(1+theta)}.

    For S_Q = S0 truncated exponential, use analytical inversion.

    Parameters
    ----------
    V_alpha_upper : float
        Worst-case VaR threshold.
    theta : float
        Premium loading.
    S_Q_func : callable
        Insurer survival function.
    M : float
        Support upper bound.
    root_tol : float
        Root-finding tolerance.

    Returns
    -------
    float
        x1 value.
    """
    target_level = 1.0 / (1.0 + theta)
    # S_Q is decreasing, so x1 = S_Q^{-1}(target_level)
    # Check boundaries
    S_at_0 = float(S_Q_func(0.0))
    if target_level >= S_at_0:
        return 0.0
    S_at_upper = float(S_Q_func(V_alpha_upper))
    if target_level <= S_at_upper:
        return float(V_alpha_upper)

    def f(x: float) -> float:
        return float(S_Q_func(x)) - target_level

    x1 = brentq(f, 0.0, V_alpha_upper, xtol=root_tol)
    return float(x1)


def compute_x2(
    V_alpha_upper: float,
    lambda_val: float,
    theta: float,
    S_Q_func: Callable,
    M: float,
    root_tol: float = 1e-10,
) -> float:
    """Compute x2 = sup{x in [overline(V_alpha), M): S_Q(x) >= 1/((1+lambda)*(1+theta))}.

    Parameters
    ----------
    V_alpha_upper : float
        Worst-case VaR threshold.
    lambda_val : float
        Lagrange multiplier.
    theta : float
        Premium loading.
    S_Q_func : callable
        Insurer survival function.
    M : float
        Support upper bound.
    root_tol : float
        Root-finding tolerance.

    Returns
    -------
    float
        x2 value (V_alpha_upper if S_Q(V_alpha_upper) < target_level).
    """
    target_level = 1.0 / ((1.0 + lambda_val) * (1.0 + theta))
    S_at_upper = float(S_Q_func(V_alpha_upper))
    if target_level >= S_at_upper:
        return float(V_alpha_upper)

    S_at_M = float(S_Q_func(M))
    if target_level <= S_at_M:
        return float(M)

    def f(x: float) -> float:
        return float(S_Q_func(x)) - target_level

    x2 = brentq(f, V_alpha_upper, M - 1e-8, xtol=root_tol)
    return float(x2)


# ---------------------------------------------------------------------------
# hatG construction
# ---------------------------------------------------------------------------

def compute_hat_x(
    beta: float,
    alpha: float,
    hat_x_alpha_Q: float,
    M: float,
) -> float:
    """Compute hat_x = phi_prime_inv(phi_prime(hat_x_alpha_Q) + 1/(beta*(1-alpha))).

    For beta=0 or very small beta, return M (effectively infinity).

    Parameters
    ----------
    beta : float
        Dual variable for BW constraint (>= 0).
    alpha : float
        Confidence level.
    hat_x_alpha_Q : float
        VaR_alpha^{F_Q} = F_Q^{-1}(alpha).
    M : float
        Support upper bound.

    Returns
    -------
    float
        hat_x, clipped to [0, M].
    """
    if beta <= 1e-12:
        return float(M)

    increment = 1.0 / (beta * (1.0 - alpha))
    u = float(phi_entropy_prime(hat_x_alpha_Q)) + increment
    hat_x = float(phi_entropy_prime_inv(u)) - 1.0  # phi_prime_inv(u) = exp(u-1)-1
    # Clip
    hat_x = min(hat_x, M)
    hat_x = max(hat_x, hat_x_alpha_Q)
    return hat_x


def compute_hatS_Q(
    x_grid: np.ndarray,
    beta: float,
    alpha: float,
    S_Q_func: Callable,
    M: float,
) -> np.ndarray:
    """Compute hatS_Q(x) = S_Q(phi_prime_inv(phi_prime(x) - 1/(beta*(1-alpha)))).

    For the entropy generator: phi_prime(x)=log(x+1)+1, phi_prime_inv(u)=exp(u-1)-1.
    So argument = exp(phi_prime(x) - 1/(beta*(1-alpha)) - 1) - 1.

    Parameters
    ----------
    x_grid : np.ndarray
        Evaluation points.
    beta : float
        Dual variable (> 0).
    alpha : float
        Confidence level.
    S_Q_func : callable
        Insurer survival function.
    M : float
        Support upper bound.

    Returns
    -------
    np.ndarray
        hatS_Q values on x_grid.
    """
    if beta <= 1e-12:
        return np.array([float(S_Q_func(x)) for x in x_grid])

    decrement = 1.0 / (beta * (1.0 - alpha))
    result = np.empty(len(x_grid))
    for i, x in enumerate(x_grid):
        phi_prime_x = float(phi_entropy_prime(x))
        u = phi_prime_x - decrement
        arg = float(phi_entropy_prime_inv(u))  # = exp(u-1)-1
        arg = np.clip(arg, 0.0, M)
        result[i] = float(S_Q_func(arg))
    return result


def compute_hatG(
    x_grid: np.ndarray,
    beta: float,
    alpha: float,
    S_Q_func: Callable,
    F_Q_inv_func: Callable,
    M: float,
) -> np.ndarray:
    """Compute hatG(x;beta) on x_grid.

    hatG(x;beta):
    - S_Q(x)        for x in [0, hat_x_alpha_Q)
    - (1-alpha)     for x in [hat_x_alpha_Q, hat_x)
    - hatS_Q(x)     for x in [hat_x, M]

    For beta=0: hatG(x;0) = S_Q(x) (benchmark).

    Parameters
    ----------
    x_grid : np.ndarray
        Evaluation points.
    beta : float
        Dual variable.
    alpha : float
        Confidence level.
    S_Q_func : callable
        Insurer survival.
    F_Q_inv_func : callable
        Insurer quantile function.
    M : float
        Support upper bound.

    Returns
    -------
    np.ndarray
        hatG values on x_grid.
    """
    if beta <= 1e-12:
        return np.array([float(S_Q_func(x)) for x in x_grid])

    hat_x_alpha_Q = float(F_Q_inv_func(alpha))
    hat_x = compute_hat_x(beta, alpha, hat_x_alpha_Q, M)
    hatS_Q_vals = compute_hatS_Q(x_grid, beta, alpha, S_Q_func, M)

    result = np.empty(len(x_grid))
    for i, x in enumerate(x_grid):
        if x < hat_x_alpha_Q:
            result[i] = float(S_Q_func(x))
        elif x < hat_x:
            result[i] = 1.0 - alpha
        else:
            result[i] = hatS_Q_vals[i]

    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# K' derivative
# ---------------------------------------------------------------------------

def compute_K_prime(
    t: float,
    x: float,
    beta: float,
    alpha: float,
    F_Q_inv_func: Callable,
) -> float:
    """Compute K'(t;x) for TVaR distortion and entropy generator.

    K'(t;x) = 1/(1-alpha) - beta*[phi'(x) - phi'(F_Q^{-1}(1-t))]  for t < 1-alpha
    K'(t;x) = -beta*[phi'(x) - phi'(F_Q^{-1}(1-t))]                for t >= 1-alpha

    Parameters
    ----------
    t : float
        Integration variable (in [0,1]).
    x : float
        Spatial point.
    beta : float
        Dual variable.
    alpha : float
        Confidence level.
    F_Q_inv_func : callable
        Quantile function of F_Q.

    Returns
    -------
    float
        K'(t;x) value.
    """
    # Avoid boundary issues
    t_clipped = np.clip(t, 1e-12, 1.0 - 1e-12)
    q = float(F_Q_inv_func(1.0 - t_clipped))
    phi_diff = float(phi_entropy_prime(x)) - float(phi_entropy_prime(q))
    base = -beta * phi_diff
    if t < 1.0 - alpha:
        return 1.0 / (1.0 - alpha) + base
    else:
        return base


# ---------------------------------------------------------------------------
# G* construction (Section 4.3 three-case formula)
# ---------------------------------------------------------------------------

def compute_G_star_tvar(
    x_grid: np.ndarray,
    lambda_val: float,
    beta: float,
    alpha: float,
    theta: float,
    S_Q_func: Callable,
    F_Q_inv_func: Callable,
    V_alpha_upper: float,
    M: float,
) -> np.ndarray:
    """Compute G*(x;beta) for TVaR with entropy phi, implementing all cases.

    Implements the three-case classification from Section 4.3:
    - Case (i):   G*(x;beta) = S_Q(x) everywhere
    - Case (ii):  piecewise with threshold hat_x relative to V_alpha_upper
    - Case (iii): piecewise with thresholds x1, x2, and hat_x

    Parameters
    ----------
    x_grid : np.ndarray
        Evaluation points in [0, M].
    lambda_val : float
        Lagrange multiplier.
    beta : float
        Dual BW variable.
    alpha : float
        Confidence level.
    theta : float
        Premium loading.
    S_Q_func : callable
        Insurer survival function.
    F_Q_inv_func : callable
        Insurer quantile function.
    V_alpha_upper : float
        Worst-case VaR threshold.
    M : float
        Support upper bound.

    Returns
    -------
    np.ndarray
        G*(x;beta) values on x_grid.
    """
    # Determine case
    S_Q_at_V_upper = float(S_Q_func(V_alpha_upper))
    case = case_type(lambda_val, theta, alpha, S_Q_at_V_upper)

    # Precompute building blocks
    hat_x_alpha_Q = float(F_Q_inv_func(alpha))
    hat_x = compute_hat_x(beta, alpha, hat_x_alpha_Q, M)
    x1 = compute_x1(V_alpha_upper, theta, S_Q_func, M)
    x2 = compute_x2(V_alpha_upper, lambda_val, theta, S_Q_func, M)

    hatG_vals = compute_hatG(x_grid, beta, alpha, S_Q_func, F_Q_inv_func, M)
    S_Q_vals = np.array([float(S_Q_func(x)) for x in x_grid])

    if case == 1:
        # G*(x;beta) = S_Q(x) for all x
        return S_Q_vals.copy()

    elif case == 2:
        # (1+lambda)*(1+theta) >= 1/(1-alpha), < 1/S_Q(V_upper)
        # x2 is in [V_upper, M] but S_Q(V_upper) >= 1/((1+lambda)(1+theta))
        # so x2 = V_upper effectively (no region above V_upper in case ii)
        result = np.empty(len(x_grid))
        factor = (1.0 + lambda_val) * (1.0 + theta) * (1.0 - alpha)

        for i, x in enumerate(x_grid):
            if x < x1:
                # Below x1: S_Q(x) >= 1/(1+theta) so H >= 0 at S0 -> stay at S0
                result[i] = S_Q_vals[i]
            elif x < V_alpha_upper:
                # Between x1 and V_upper: use hatG capped by factor
                result[i] = min(hatG_vals[i], factor)
            else:
                # Above V_upper: in case ii, x2 ~ V_upper, so S_Q here
                result[i] = S_Q_vals[i]

        return np.clip(result, 0.0, 1.0)

    else:
        # Case (iii): (1+lambda)*(1+theta) >= 1/S_Q(V_upper)
        factor = (1.0 + lambda_val) * (1.0 + theta) * (1.0 - alpha)
        result = np.empty(len(x_grid))

        for i, x in enumerate(x_grid):
            if x < x1:
                result[i] = S_Q_vals[i]
            elif x < V_alpha_upper:
                result[i] = min(hatG_vals[i], factor)
            elif x < x2:
                # Between V_upper and x2: use modified formula with lambda
                factor2 = (1.0 + lambda_val) * (1.0 + theta) * (1.0 - alpha)
                result[i] = min(hatG_vals[i], factor2)
            else:
                # Above x2: S_Q(x) < threshold, use hatG
                result[i] = hatG_vals[i]

        return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Net price function H for TVaR
# ---------------------------------------------------------------------------

def compute_H_tvar(
    x_grid: np.ndarray,
    lambda_val: float,
    theta: float,
    alpha: float,
    S_Q_func: Callable,
    V_alpha_upper: float,
) -> np.ndarray:
    """Compute H(x; S_Q, lambda) for TVaR distortion.

    H(x; S_Q, lambda) = (1+lambda)*(1+theta)*S_Q(x)
                        - g(S_Q(x))
                        - lambda * 1_{[0, overline(V_alpha)]}(x)

    where g is TVaR distortion g(t) = min(t/(1-alpha), 1).

    Parameters
    ----------
    x_grid : np.ndarray
        Evaluation points.
    lambda_val : float
        Lagrange multiplier.
    theta : float
        Premium loading.
    alpha : float
        Confidence level.
    S_Q_func : callable
        Insurer survival function.
    V_alpha_upper : float
        Worst-case VaR threshold.

    Returns
    -------
    np.ndarray
        H values on x_grid.
    """
    result = np.empty(len(x_grid))
    for i, x in enumerate(x_grid):
        s = float(S_Q_func(x))
        g_s = float(tvar_distortion(s, alpha))
        indicator = 1.0 if x <= V_alpha_upper else 0.0
        result[i] = (1.0 + lambda_val) * (1.0 + theta) * s - g_s - lambda_val * indicator
    return result


# ---------------------------------------------------------------------------
# Full Section 4.3 analysis runner
# ---------------------------------------------------------------------------

def run_section43_analysis(
    lambda_values: Optional[List[float]] = None,
    beta_values: Optional[List[float]] = None,
    M: float = 100.0,
    alpha: float = 0.95,
    theta: float = 0.5,
    epsilon: float = 0.5,
    n_grid: int = 1000,
) -> dict:
    """Run the full Section 4.3 analysis across lambda and beta values.

    Computes G*(x;beta) and H for each (lambda, beta) pair, classifies cases,
    and returns a summary dictionary.

    Parameters
    ----------
    lambda_values : list of float
        Lagrange multiplier values to test.
    beta_values : list of float
        Dual BW variable values to test.
    M : float
        Support upper bound.
    alpha : float
        Confidence level.
    theta : float
        Premium loading.
    epsilon : float
        BW radius.
    n_grid : int
        Number of grid points.

    Returns
    -------
    dict
        Keys: 'x_grid', 'V_upper', 'G_star', 'case_map', 'case_table', 'H_map'.
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    if beta_values is None:
        beta_values = [0.0, 0.5, 1.0, 2.0]

    x_grid = np.linspace(0.0, min(M, 50.0), n_grid)

    # Use entropy generator (piecewise quadratic phi is for exp_1)
    # Compute V_upper self-sufficiently using entropy generator and Prop 3.1
    # For the entropy generator, we need B_phi_entropy in the BW formula
    from src.bregman import B_phi_entropy

    q_alpha = float(F0_inv(alpha, M))

    def _bw_upper_entropy(D: float) -> float:
        upper_t = min(float(F0(D, M)), 1.0 - 1e-12)
        if upper_t <= alpha + 1e-14:
            return 0.0
        def integrand(t: float) -> float:
            return float(B_phi_entropy(D, float(F0_inv(t, M))))
        val, _ = quad(integrand, alpha, upper_t, limit=200, epsabs=1e-10, epsrel=1e-10)
        return val

    # T_M for entropy generator
    def tm_integrand(t: float) -> float:
        return float(B_phi_entropy(M, float(F0_inv(t, M))))
    T_M, _ = quad(tm_integrand, alpha, 1.0 - 1e-12, limit=200, epsabs=1e-10, epsrel=1e-10)

    if epsilon >= T_M:
        V_upper = float(M)
    else:
        def target_V(D: float) -> float:
            return _bw_upper_entropy(D) - epsilon
        lo, hi = q_alpha + 1e-6, M - 1e-6
        val_hi = target_V(hi)
        if val_hi < 0:
            V_upper = float(M)
        else:
            try:
                V_upper = float(brentq(target_V, lo, hi, xtol=1e-10, maxiter=500))
            except Exception:
                V_upper = float(hi)

    S0_func = lambda x: float(S0(x, M))
    F0_inv_func = lambda t: float(F0_inv(t, M))
    S_Q_at_V_upper = S0_func(V_upper)

    G_star_dict: dict = {}
    case_map_dict: dict = {}
    H_map_dict: dict = {}
    case_rows = []

    for lam in lambda_values:
        case = case_type(lam, theta, alpha, S_Q_at_V_upper)
        for beta in beta_values:
            G = compute_G_star_tvar(
                x_grid, lam, beta, alpha, theta,
                S0_func, F0_inv_func, V_upper, M
            )
            G_star_dict[(lam, beta)] = G
            case_map_dict[(lam, beta)] = case

        H = compute_H_tvar(x_grid, lam, theta, alpha, S0_func, V_upper)
        H_map_dict[lam] = H

        # Compute thresholds for the case table
        x1 = compute_x1(V_upper, theta, S0_func, M)
        x2 = compute_x2(V_upper, lam, theta, S0_func, M)
        case_rows.append({
            "lambda": lam,
            "case": case,
            "(1+lam)(1+theta)": (1.0 + lam) * (1.0 + theta),
            "boundary_1_(1/(1-alpha))": 1.0 / (1.0 - alpha),
            "boundary_2_(1/SQ_Vupper)": 1.0 / max(S_Q_at_V_upper, 1e-15),
            "x1": x1,
            "x2": x2,
            "V_upper": V_upper,
        })

    return {
        "x_grid": x_grid,
        "V_upper": V_upper,
        "q_alpha": q_alpha,
        "G_star": G_star_dict,
        "case_map": case_map_dict,
        "H_map": H_map_dict,
        "case_table": case_rows,
    }
