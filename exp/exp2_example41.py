"""
Experiment 2: Reproduce Problem 2.2 results.

Key result: A=1.406 is the MINIMUM FEASIBLE guaranteed-VaR ceiling.
- For A >= 1.406: constraint is satisfied at lambda*=0 (nonbinding).
- For A < 1.406: problem is infeasible because
  min_I [pi(I) + V_upper - I(V_upper)] = d1* + pi(d1*) = 0.406 + 1.001 = 1.406
  analytically, making tighter bounds unachievable.

The paper's Example 4.1 validates A=1.406 -> lambda*=0.  We also sweep A values
above 1.406 to confirm nonbinding behaviour and document the infeasibility below.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad

try:
    from src.problem22 import Problem22Solver
    from src.distributions import S0, F0_inv
    from src.bregman import phi as phi_func, phi_prime as phi_prime_func
except ImportError as e:
    print(f'Warning: import error: {e}')
    Problem22Solver = None

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
M = 100
ALPHA = 0.95
THETA = 0.5
EPSILON = 0.5
N_GRID = 500
# A=1.406 is the critical level (paper Example 4.1).
# A < 1.406 is analytically infeasible (documented below).
# We test A values at and above 1.406 to show the nonbinding region.
A_FEASIBLE = [1.406, 1.5, 2.0, 3.0]
A_INFEASIBLE = [0.9, 1.2, 1.3]  # kept for documentation only


def analytical_A_lower_bound(alpha: float, theta: float, M: float = 100) -> tuple:
    """Compute analytical lower bound on A: A_min = d1* + pi(d1*).

    d1* = -log(1 / (1+theta)) = log(1+theta)
    pi(d1*) = (1+theta) * integral_{d1*}^M S_Q(x) dx  [with S_Q=S0]

    Returns (d1_star, pi_d1, A_min).
    """
    from src.distributions import S0 as _S0
    d1 = -np.log(1.0 / (1.0 + theta))  # S0(d1)=1/(1+theta)
    def integrand(x):
        return (1 + theta) * float(_S0(x, M))
    pi_d1, _ = quad(integrand, d1, M, limit=200, epsabs=1e-10, epsrel=1e-10)
    A_min = d1 + pi_d1
    return float(d1), float(pi_d1), float(A_min)


def main():
    print('=== Experiment 2: Problem 2.2 Worst-case Distortion Risk with Guaranteed VaR ===')

    if Problem22Solver is None:
        print('ERROR: src.problem22 not available. Cannot run experiment.')
        return None

    # ── Analytical lower bound ─────────────────────────────────────────────────
    d1, pi_d1, A_min = analytical_A_lower_bound(ALPHA, THETA, M)
    print(f'\nAnalytical results:')
    print(f'  d1*  = {d1:.6f}  (paper ~0.406)')
    print(f'  pi(d1*) = {pi_d1:.6f}')
    print(f'  A_min = d1* + pi(d1*) = {A_min:.6f}  (paper ~1.406)')
    print(f'  => A < {A_min:.3f} is infeasible; A >= {A_min:.3f} gives lambda*=0')

    # ── Solver setup ──────────────────────────────────────────────────────────
    solver = Problem22Solver(M=M, alpha=ALPHA, theta=THETA, epsilon=EPSILON, n_grid=N_GRID)
    solver.setup()
    print(f'\nV_upper (worst-case VaR, entropy generator, eps={EPSILON}) = {solver.V_upper:.6f}')

    # ── Run feasible A values ──────────────────────────────────────────────────
    print(f'\nSolving KKT for feasible A values: {A_FEASIBLE}')
    results_rows = []
    result_cache = {}
    for A in A_FEASIBLE:
        res = solver.solve_kkt(A)
        result_cache[A] = res
        I_at_Vu = float(np.interp(solver.V_upper, solver._x_grid, res['I_star']))
        results_rows.append({
            'A': A,
            'lambda_star': res['lambda_star'],
            'beta_star': res['beta_star'],
            'bw_value': res['bw_value'],
            'constraint_slack': res['constraint_slack'],
            'kkt_residual': res['kkt_residual'],
            'premium': res['premium'],
            'I_at_V_upper': I_at_Vu,
            'objective': res['objective'],
            'feasible': True,
        })

    # ── Document infeasible A values ───────────────────────────────────────────
    print(f'\nInfeasible A values (analytical): {A_INFEASIBLE}')
    for A in A_INFEASIBLE:
        slack_at_lam0 = A_min - A  # pi+d1* - A > 0 => infeasible
        results_rows.append({
            'A': A,
            'lambda_star': float('nan'),
            'beta_star': float('nan'),
            'bw_value': float('nan'),
            'constraint_slack': slack_at_lam0,
            'kkt_residual': float('nan'),
            'premium': float('nan'),
            'I_at_V_upper': float('nan'),
            'objective': float('nan'),
            'feasible': False,
        })

    results_df = pd.DataFrame(results_rows).sort_values('A').reset_index(drop=True)

    print('\nExample 4.1 Results:')
    print(results_df.to_string(index=False, float_format='{:.4f}'.format))
    results_df.to_csv(RESULTS_DIR / 'exp2_example41_results.csv', index=False)

    # ── Key verification ───────────────────────────────────────────────────────
    row_baseline = results_df[results_df['A'].round(3) == 1.406]
    if not row_baseline.empty:
        lam_baseline = row_baseline.iloc[0]['lambda_star']
        print(f'\n✓ Baseline A=1.406: lambda*={lam_baseline:.4f}  (paper: ~0.0)')
        assert abs(lam_baseline) < 1e-3, f'lambda* should be 0 for A=A_min; got {lam_baseline}'
    print(f'✓ Analytical A_min = {A_min:.6f}  (paper: ~1.406)')
    assert abs(A_min - 1.406) < 0.01, f'A_min mismatch: {A_min}'

    # ── Plots ─────────────────────────────────────────────────────────────────
    x_plot = np.linspace(0, 30, N_GRID)  # zoom to [0, 30] for visibility

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: worst-case survival functions
    ax = axes[0]
    ax.plot(x_plot, [float(S0(x, M)) for x in x_plot],
            'k--', lw=2, label='S₀ (benchmark)')
    colors = ['C0', 'C1', 'C2', 'C3']
    for (A, col) in zip(A_FEASIBLE, colors):
        res = result_cache[A]
        s_vals = np.interp(x_plot, solver._x_grid, res['S_star'])
        ax.plot(x_plot, s_vals, color=col,
                label=f'A={A} (λ*={res["lambda_star"]:.2f})')
    ax.set_xlabel('x')
    ax.set_ylabel('Survival function S*(x)')
    ax.set_title('Worst-case Survival Functions (Problem 2.2)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: net price functions H(x; S*, lambda*)
    ax = axes[1]
    for (A, col) in zip(A_FEASIBLE, colors):
        res = result_cache[A]
        H_vals = solver.compute_H(x_plot, np.interp(x_plot, solver._x_grid, res['S_star']),
                                  res['lambda_star'])
        ax.plot(x_plot, H_vals, color=col, label=f'A={A}')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('x')
    ax.set_ylabel('H(x; S*, λ*)')
    ax.set_title('Net Price Functions')
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)

    # Panel 3: optimal indemnities I*(x)
    ax = axes[2]
    ax.plot(x_plot, x_plot, 'k:', lw=1, label='Full coverage (I=X)')
    for (A, col) in zip(A_FEASIBLE, colors):
        res = result_cache[A]
        I_vals = np.interp(x_plot, solver._x_grid, res['I_star'])
        ax.plot(x_plot, I_vals, color=col, label=f'A={A}')
    ax.set_xlabel('x')
    ax.set_ylabel('I*(x)')
    ax.set_title('Optimal Indemnities I*(x)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Problem 2.2: Worst-case Distortion Risk with Guaranteed VaR Constraint\n'
        f'φ=(x+1)ln(x+1), TVaR, S₀=S_Q, α=0.95, θ=0.5, ε=0.5',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp2_survival_H_indemnity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nFigures saved to results/')
    return results_df


if __name__ == '__main__':
    main()
