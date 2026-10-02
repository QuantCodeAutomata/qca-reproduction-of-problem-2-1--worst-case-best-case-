"""Problem 2.2: Worst-case convex distortion risk with guaranteed VaR constraint.

Implements the Lagrangian reformulation, minimax reduction, G*(x;beta),
flat-segment repair, and KKT selection of lambda* and beta* for Problem 2.2
of "Distributionally Robust Insurance under Bregman-Wasserstein Divergence."

Uses entropy generator phi(x)=(x+1)*log(x+1) and TVaR distortion.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

from src.distributions import F0, S0, F0_inv
from src.bregman import (
    B_phi_entropy,
    phi_entropy_prime,
    phi_entropy_prime_inv,
    bw_divergence_quantile_entropy,
)
from src.distortion import tvar_distortion
from src.repair import (
    detect_upward_jump,
    repair_survival_flat_segment,
    find_optimal_b,
)
from src.bw_survival import compute_psi, compute_C0
from src.tvar_section43 import (
    case_type,
    compute_hat_x,
    compute_hatG,
    compute_hatS_Q,
    compute_x1,
    compute_x2,
    compute_H_tvar,
    compute_G_star_tvar,
)


class Problem22Solver:
    """Solver for Problem 2.2: robust distortion risk with guaranteed VaR.

    Implements the Lagrangian / minimax method from Section 4 using:
    - Entropy generator phi(x) = (x+1)*log(x+1)
    - TVaR distortion g(t) = min(t/(1-alpha), 1)
    - Truncated exponential benchmark S0 = S_Q

    Parameters
    ----------
    M : float
        Support upper bound (default 100).
    alpha : float
        Confidence level (default 0.95).
    theta : float
        Premium loading (default 0.5).
    epsilon : float
        BW ambiguity radius (default 0.5).
    quad_abs_tol, quad_rel_tol, root_tol : float
        Numerical tolerances.
    n_grid : int
        Number of grid points for survival-function objects.
    """

    def __init__(
        self,
        M: float = 100.0,
        alpha: float = 0.95,
        theta: float = 0.5,
        epsilon: float = 0.5,
        quad_abs_tol: float = 1e-10,
        quad_rel_tol: float = 1e-10,
        root_tol: float = 1e-10,
        n_grid: int = 500,
    ) -> None:
        self.M = M
        self.alpha = alpha
        self.theta = theta
        self.epsilon = epsilon
        self.quad_abs_tol = quad_abs_tol
        self.quad_rel_tol = quad_rel_tol
        self.root_tol = root_tol
        self.n_grid = n_grid

        # Populated in setup()
        self.q_alpha: float = 0.0
        self.V_upper: float = 0.0
        self._x_grid: np.ndarray = np.array([])
        self._S0_grid: np.ndarray = np.array([])
        self._C0: float = 0.0
        self._zeta: float = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize benchmark, compute overline(V_alpha), BW constant C0."""
        self.q_alpha = float(F0_inv(self.alpha, self.M))

        # Compute V_upper self-sufficiently using entropy generator Prop 3.1
        self.V_upper = self._compute_V_upper()

        # Grid
        self._x_grid = np.linspace(0.0, self.M, self.n_grid)
        self._S0_grid = np.array([float(S0(x, self.M)) for x in self._x_grid])

        # BW survival constant C0 = integral phi''(x)*x*S0(x) dx
        def c0_integrand(x: float) -> float:
            return (1.0 / (x + 1.0)) * x * float(S0(x, self.M))

        self._C0, _ = quad(c0_integrand, 0.0, self.M, limit=200,
                           epsabs=self.quad_abs_tol, epsrel=self.quad_rel_tol)
        self._zeta = self.epsilon - self._C0

    def _compute_V_upper(self) -> float:
        """Compute worst-case VaR with entropy generator using Proposition 3.1."""
        alpha, M = self.alpha, self.M

        def bw_upper(D: float) -> float:
            upper_t = min(float(F0(D, M)), 1.0 - 1e-12)
            if upper_t <= alpha + 1e-14:
                return 0.0
            def integrand(t: float) -> float:
                return float(B_phi_entropy(D, float(F0_inv(t, M))))
            val, _ = quad(integrand, alpha, upper_t, limit=200,
                          epsabs=self.quad_abs_tol, epsrel=self.quad_rel_tol)
            return val

        def tm_integrand(t: float) -> float:
            return float(B_phi_entropy(M, float(F0_inv(t, M))))

        T_M, _ = quad(tm_integrand, alpha, 1.0 - 1e-12, limit=200,
                      epsabs=self.quad_abs_tol, epsrel=self.quad_rel_tol)

        if self.epsilon >= T_M:
            return float(M)

        def target(D: float) -> float:
            return bw_upper(D) - self.epsilon

        lo, hi = self.q_alpha + 1e-6, M - 1e-6
        if target(hi) < 0:
            return float(M)
        try:
            return float(brentq(target, lo, hi, xtol=self.root_tol, maxiter=500))
        except Exception:
            return float(hi)

    # ------------------------------------------------------------------
    # BW Psi functional on grid
    # ------------------------------------------------------------------

    def _compute_psi_grid(self, S_vals: np.ndarray) -> float:
        """Compute Psi(S) on the stored grid using trapezoidal integration.

        Psi(S) = integral_0^M [phi'(x)-phi'(0)] S(x) dx
                 - integral_0^M integral_0^M phi''(y) min(S0(y), S(x)) dy dx
        """
        x_grid = self._x_grid
        M = self.M

        # phi'(x) = log(x+1)+1, phi'(0) = 0+1 = 1
        phi_prime_vals = np.log(x_grid + 1.0) + 1.0
        phi_prime_0 = 1.0

        term1 = float(np.trapz((phi_prime_vals - phi_prime_0) * S_vals, x_grid))

        # Inner integrals: for each x, integral_0^M (1/(y+1)) * min(S0(y), S(x)) dy
        inner_vals = np.empty(len(x_grid))
        for i, s_val in enumerate(S_vals):
            def inner_integrand(y: float) -> float:
                return (1.0 / (y + 1.0)) * min(float(S0(y, M)), s_val)
            val, _ = quad(inner_integrand, 0.0, M, limit=100,
                          epsabs=self.quad_abs_tol * 10, epsrel=self.quad_rel_tol * 10)
            inner_vals[i] = val

        term2 = float(np.trapz(inner_vals, x_grid))
        return float(term1 - term2)

    def _bw_from_S_vals(self, S_vals: np.ndarray) -> float:
        """BW[F, F0] from survival values on grid: psi + C0."""
        return self._compute_psi_grid(S_vals) + self._C0

    # ------------------------------------------------------------------
    # Net price function H
    # ------------------------------------------------------------------

    def compute_H(
        self,
        x_grid: np.ndarray,
        S_values: np.ndarray,
        lambda_val: float,
    ) -> np.ndarray:
        """Compute H(x; S, lambda) = (1+lambda)*(1+theta)*S_Q(x) - g(S(x)) - lambda*1_{[0,V_upper]}(x).

        Parameters
        ----------
        x_grid : np.ndarray
            Evaluation points.
        S_values : np.ndarray
            Survival function values S(x) on x_grid.
        lambda_val : float
            Lagrange multiplier.

        Returns
        -------
        np.ndarray
            H values.
        """
        M = self.M
        S_Q_vals = np.array([float(S0(x, M)) for x in x_grid])
        g_S = np.array([float(tvar_distortion(s, self.alpha)) for s in S_values])
        indicator = np.array([1.0 if x <= self.V_upper else 0.0 for x in x_grid])
        return (1.0 + lambda_val) * (1.0 + self.theta) * S_Q_vals - g_S - lambda_val * indicator

    # ------------------------------------------------------------------
    # Optimal indemnity for given S and lambda
    # ------------------------------------------------------------------

    def _compute_indemnity(
        self,
        S_vals: np.ndarray,
        lambda_val: float,
    ) -> Tuple[np.ndarray, float]:
        """Construct I*(x) from sign of H(x; S, lambda).

        Returns (I_star_values, premium).
        """
        x_grid = self._x_grid
        H_vals = self.compute_H(x_grid, S_vals, lambda_val)
        I_prime = np.where(H_vals <= 0.0, 1.0, 0.0)

        # Integrate
        dx = np.diff(x_grid)
        I_vals = np.zeros(len(x_grid))
        for i in range(1, len(x_grid)):
            I_vals[i] = I_vals[i - 1] + 0.5 * (I_prime[i - 1] + I_prime[i]) * dx[i - 1]

        # Premium: (1+theta) * integral S_Q(x) * I'(x) dx
        S_Q_vals = np.array([float(S0(x, self.M)) for x in x_grid])
        premium = (1.0 + self.theta) * float(np.trapz(S_Q_vals * I_prime, x_grid))
        return I_vals, premium

    # ------------------------------------------------------------------
    # Lagrangian objective for worst-case S
    # ------------------------------------------------------------------

    def _lagrangian_objective(
        self,
        S_vals: np.ndarray,
        lambda_val: float,
        A: float,
    ) -> float:
        """Compute the Lagrangian objective for given S, lambda, A.

        L(S; lambda) = integral_0^M min(g(S(x)), (1+lambda)*(1+theta)*S_Q(x) - lambda*1_{<= V_upper}) dx
                     - lambda*(A - V_upper)
        """
        x_grid = self._x_grid
        M = self.M
        S_Q_vals = np.array([float(S0(x, M)) for x in x_grid])
        g_S = np.array([float(tvar_distortion(s, self.alpha)) for s in S_vals])
        indicator = np.array([1.0 if x <= self.V_upper else 0.0 for x in x_grid])
        upper_bound = (1.0 + lambda_val) * (1.0 + self.theta) * S_Q_vals - lambda_val * indicator
        integrand_vals = np.minimum(g_S, upper_bound)
        obj = float(np.trapz(integrand_vals, x_grid)) - lambda_val * (A - self.V_upper)
        return obj

    # ------------------------------------------------------------------
    # G* construction and beta* selection
    # ------------------------------------------------------------------

    def _construct_G_star(
        self,
        lambda_val: float,
        beta: float,
    ) -> np.ndarray:
        """Construct G*(x;beta) using TVaR Section 4.3 formulas."""
        return compute_G_star_tvar(
            self._x_grid,
            lambda_val=lambda_val,
            beta=beta,
            alpha=self.alpha,
            theta=self.theta,
            S_Q_func=lambda x: float(S0(x, self.M)),
            F_Q_inv_func=lambda t: float(F0_inv(t, self.M)),
            V_alpha_upper=self.V_upper,
            M=self.M,
        )

    def _repair_G_star(
        self,
        G_star_vals: np.ndarray,
        lambda_val: float,
        A: float,
    ) -> Tuple[float, np.ndarray]:
        """Apply flat-segment repair to G* if needed.

        Returns (b_star, repaired_S_vals).
        """
        has_jump = detect_upward_jump(
            G_star_vals, self._x_grid, self.V_upper, tol=1e-6
        )
        if not has_jump:
            return float(np.interp(self.V_upper, self._x_grid, G_star_vals)), G_star_vals

        def obj_func(S: np.ndarray) -> float:
            return self._lagrangian_objective(S, lambda_val, A)

        b_star, S_repaired = find_optimal_b(
            G_star_vals, self.V_upper, self._x_grid, obj_func
        )
        return b_star, S_repaired

    def repair_and_solve_beta(
        self,
        lambda_val: float,
        A: float,
        beta_max: float = 10.0,
        beta_grid: int = 50,
    ) -> Tuple[float, np.ndarray]:
        """For fixed lambda, find beta* and the corresponding repaired S*.

        1. Test beta=0: if BW budget is feasible, set beta*=0.
        2. Otherwise solve for beta*>0 where BW constraint binds.

        Returns (beta_star, S_star_values).
        """
        # Test beta=0
        G0 = self._construct_G_star(lambda_val, beta=0.0)
        S0_vals = np.array([float(S0(x, self.M)) for x in self._x_grid])
        # Ensure S >= S0
        G0 = np.maximum(G0, S0_vals)
        _, S0_repaired = self._repair_G_star(G0, lambda_val, A)
        S0_repaired = np.maximum(S0_repaired, S0_vals)
        bw0 = self._bw_from_S_vals(S0_repaired)

        if bw0 <= self.epsilon + 1e-6:
            return 0.0, S0_repaired

        # Sweep beta to find where BW constraint binds
        def bw_at_beta(beta: float) -> float:
            G = self._construct_G_star(lambda_val, beta)
            G = np.maximum(G, S0_vals)
            _, S_rep = self._repair_G_star(G, lambda_val, A)
            S_rep = np.maximum(S_rep, S0_vals)
            return self._bw_from_S_vals(S_rep)

        # BW should decrease as beta increases (more penalization)
        bw_at_max = bw_at_beta(beta_max)
        if bw_at_max > self.epsilon:
            # Budget constraint can't be satisfied in [0, beta_max]
            # Return beta_max as best effort
            G = self._construct_G_star(lambda_val, beta_max)
            G = np.maximum(G, S0_vals)
            _, S_rep = self._repair_G_star(G, lambda_val, A)
            S_rep = np.maximum(S_rep, S0_vals)
            return beta_max, S_rep

        def target_beta(beta: float) -> float:
            return bw_at_beta(beta) - self.epsilon

        try:
            beta_star = float(brentq(target_beta, 0.0, beta_max,
                                     xtol=self.root_tol, maxiter=100))
        except Exception:
            beta_star = beta_max

        G_star = self._construct_G_star(lambda_val, beta_star)
        G_star = np.maximum(G_star, S0_vals)
        _, S_star = self._repair_G_star(G_star, lambda_val, A)
        S_star = np.maximum(S_star, S0_vals)
        return beta_star, S_star

    # ------------------------------------------------------------------
    # KKT outer search
    # ------------------------------------------------------------------

    def _kkt_tolerance(self) -> float:
        """Return numerical tolerance for the KKT constraint slack check.

        The tolerance accounts for trapezoidal-grid discretisation errors.
        We use 2/n_grid as a relative proxy for the grid error scale.
        """
        return max(2.0 / self.n_grid, 1e-3)

    def solve_kkt(
        self,
        A: float,
        lambda_max: float = 10.0,
        lambda_grid: int = 50,
    ) -> dict:
        """Find lambda* via KKT complementarity for guaranteed VaR ceiling A.

        KKT conditions:
        - lambda* >= 0
        - constraint_slack = pi(I*) - I*(V_upper) - A + V_upper <= 0
        - lambda* * constraint_slack = 0

        Parameters
        ----------
        A : float
            Guaranteed worst-case VaR ceiling.
        lambda_max : float
            Search upper bound for lambda.
        lambda_grid : int
            Number of lambda grid points for bracket search.

        Returns
        -------
        dict
            Keys: A, lambda_star, beta_star, b_star, S_star, I_star,
                  premium, bw_value, constraint_slack, kkt_residual, objective.
        """
        kkt_tol = self._kkt_tolerance()

        def evaluate_lambda(lambda_val: float) -> dict:
            beta_star, S_star = self.repair_and_solve_beta(lambda_val, A)
            I_star, premium = self._compute_indemnity(S_star, lambda_val)
            I_at_V_upper = float(np.interp(self.V_upper, self._x_grid, I_star))
            constraint_slack = premium - I_at_V_upper - A + self.V_upper
            bw_val = self._bw_from_S_vals(S_star)
            return {
                "lambda_val": lambda_val,
                "beta_star": beta_star,
                "S_star": S_star,
                "I_star": I_star,
                "premium": premium,
                "I_at_V_upper": I_at_V_upper,
                "constraint_slack": constraint_slack,
                "bw_value": bw_val,
            }

        # Evaluate at lambda=0 first
        res_0 = evaluate_lambda(0.0)
        # Use grid-aware tolerance so mild discretisation errors don't trigger lambda>0
        if res_0["constraint_slack"] <= kkt_tol:
            # Constraint is not binding at lambda=0
            r = res_0
            return {
                "A": A,
                "lambda_star": 0.0,
                "beta_star": r["beta_star"],
                "b_star": float(np.interp(self.V_upper, self._x_grid, r["S_star"])),
                "S_star": r["S_star"],
                "I_star": r["I_star"],
                "premium": r["premium"],
                "bw_value": r["bw_value"],
                "constraint_slack": r["constraint_slack"],
                "kkt_residual": abs(0.0 * r["constraint_slack"]),
                "objective": self._lagrangian_objective(r["S_star"], 0.0, A),
            }

        # Need to find lambda* > 0 that drives constraint_slack to 0
        # Sweep to find bracket
        res_max = evaluate_lambda(lambda_max)
        if res_max["constraint_slack"] >= -kkt_tol:
            # Constraint slack remains positive even at lambda_max
            r = res_max
            return {
                "A": A,
                "lambda_star": lambda_max,
                "beta_star": r["beta_star"],
                "b_star": float(np.interp(self.V_upper, self._x_grid, r["S_star"])),
                "S_star": r["S_star"],
                "I_star": r["I_star"],
                "premium": r["premium"],
                "bw_value": r["bw_value"],
                "constraint_slack": r["constraint_slack"],
                "kkt_residual": abs(lambda_max * r["constraint_slack"]),
                "objective": self._lagrangian_objective(r["S_star"], lambda_max, A),
            }

        # Bracket found: solve for zero-crossing
        def slack_func(lam: float) -> float:
            return evaluate_lambda(lam)["constraint_slack"]

        try:
            lambda_star = float(brentq(slack_func, 0.0, lambda_max,
                                       xtol=self.root_tol, maxiter=100))
        except Exception:
            lambda_star = lambda_max / 2.0

        r = evaluate_lambda(lambda_star)
        return {
            "A": A,
            "lambda_star": lambda_star,
            "beta_star": r["beta_star"],
            "b_star": float(np.interp(self.V_upper, self._x_grid, r["S_star"])),
            "S_star": r["S_star"],
            "I_star": r["I_star"],
            "premium": r["premium"],
            "bw_value": r["bw_value"],
            "constraint_slack": r["constraint_slack"],
            "kkt_residual": abs(lambda_star * r["constraint_slack"]),
            "objective": self._lagrangian_objective(r["S_star"], lambda_star, A),
        }

    # ------------------------------------------------------------------
    # Run over A values
    # ------------------------------------------------------------------

    def run(self, A_values: List[float] = None) -> pd.DataFrame:
        """Run the full procedure for each A value.

        Parameters
        ----------
        A_values : list of float
            Guaranteed VaR ceiling values to test.

        Returns
        -------
        pd.DataFrame
            Summary table.
        """
        if A_values is None:
            A_values = [1.406, 1.2, 0.9]

        rows = []
        for A in A_values:
            try:
                result = self.solve_kkt(A)
                rows.append({
                    "A": A,
                    "lambda_star": result["lambda_star"],
                    "beta_star": result["beta_star"],
                    "b_star": result["b_star"],
                    "bw_value": result["bw_value"],
                    "bw_margin": self.epsilon - result["bw_value"],
                    "constraint_slack": result["constraint_slack"],
                    "kkt_residual": result["kkt_residual"],
                    "premium": result["premium"],
                    "I_at_V_upper": float(np.interp(
                        self.V_upper, self._x_grid, result["I_star"]
                    )),
                    "objective": result["objective"],
                })
            except Exception as e:
                rows.append({
                    "A": A,
                    "lambda_star": float("nan"),
                    "beta_star": float("nan"),
                    "b_star": float("nan"),
                    "bw_value": float("nan"),
                    "bw_margin": float("nan"),
                    "constraint_slack": float("nan"),
                    "kkt_residual": float("nan"),
                    "premium": float("nan"),
                    "I_at_V_upper": float("nan"),
                    "objective": float("nan"),
                    "error": str(e),
                })

        return pd.DataFrame(rows)
