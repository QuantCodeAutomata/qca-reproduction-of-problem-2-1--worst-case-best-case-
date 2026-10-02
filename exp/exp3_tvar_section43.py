"""
Experiment 3: TVaR-specific Section 4.3 analytical reproduction.
Validates three-case classification and G*(x;beta) formulas.
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

try:
    from src.distributions import S0, F0_inv
except ImportError as e:
    print(f'Warning: could not import src.distributions: {e}')
    S0 = F0_inv = None

try:
    from src.tvar_section43 import (
        case_type, compute_hat_x, compute_G_star_tvar,
        run_section43_analysis,
    )
    _HAS_TVAR_BASE = True
except ImportError as e:
    print(f'Warning: could not import src.tvar_section43 base functions: {e}')
    case_type = compute_hat_x = compute_G_star_tvar = None
    run_section43_analysis = None
    _HAS_TVAR_BASE = False

# compute_H_tvar is a companion function not in the formal interface;
# import separately so the rest of the module still loads if it's absent.
try:
    from src.tvar_section43 import compute_H_tvar
    _HAS_COMPUTE_H_TVAR = True
except ImportError:
    compute_H_tvar = None
    _HAS_COMPUTE_H_TVAR = False

try:
    from src.distortion import tvar_distortion
except ImportError as e:
    print(f'Warning: could not import src.distortion: {e}')
    tvar_distortion = None

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print('=== Experiment 3: Section 4.3 TVaR-Specific Analysis ===')

    if not _HAS_TVAR_BASE:
        print('ERROR: src.tvar_section43 not available. Cannot run experiment.')
        return None

    if S0 is None or F0_inv is None:
        print('ERROR: src.distributions not available. Cannot run experiment.')
        return None

    M, alpha, theta, epsilon = 100, 0.95, 0.5, 0.5
    n_grid = 1000

    # Case (i)/(ii) boundary: (1+lambda)*(1+theta) = 1/(1-alpha) = 20
    #   => lambda_boundary1 = 20/1.5 - 1 = 12.33
    # Case (ii)/(iii) boundary: (1+lambda)*(1+theta) = 1/S_Q(V_upper)
    #   => V_upper large => S_Q(V_upper) ≈ 0 => boundary2 ≈ +inf (unreachable)
    # Strategy: include lambda values below (case i), crossing (case ii boundary), above (case ii)
    lambda_values = [0.0, 5.0, 12.0, 12.5, 13.0, 15.0, 20.0, 50.0]
    beta_values = [0.0, 0.5, 1.0, 2.0]

    results = run_section43_analysis(
        lambda_values=lambda_values,
        beta_values=beta_values,
        M=M, alpha=alpha, theta=theta, epsilon=epsilon,
        n_grid=n_grid
    )

    print('\nCase Classification Results:')
    print(pd.DataFrame(results['case_table']).to_string())

    # Save case table
    pd.DataFrame(results['case_table']).to_csv(
        RESULTS_DIR / 'exp3_case_classification.csv', index=False
    )

    x_grid = results['x_grid']
    V_upper = results['V_upper']

    # Plot G*(x;beta) for different lambda
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    beta_plot = 1.0  # Fixed beta for lambda sweep

    ax = axes[0]
    ax.plot(x_grid, np.array([S0(x, M) for x in x_grid]), 'k--', label='S0', linewidth=1.5)
    for lam in [0.0, 5.0, 13.0, 20.0]:
        if (lam, beta_plot) in results.get('G_star', {}):
            G = results['G_star'][(lam, beta_plot)]
            case = results['case_map'][(lam, beta_plot)]
            ax.plot(x_grid, G, label=f'$\\lambda$={lam} (case {case})')
    ax.axvline(
        x=V_upper, color='red', linestyle=':', alpha=0.7,
        label=f'$\\bar{{V}}_\\alpha$={V_upper:.2f}'
    )
    ax.set_xlabel('x')
    ax.set_ylabel('G*(x; beta)')
    ax.set_title(f'Candidate Worst-case Survival ($\\beta$={beta_plot})')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)

    # H(x; S_Q, lambda)
    ax = axes[1]
    if _HAS_COMPUTE_H_TVAR:
        for lam in [0.0, 5.0, 13.0, 20.0]:
            S0_func_local = lambda x, _M=M: float(S0(x, _M))
            H_vals = compute_H_tvar(
                x_grid, lam, theta, alpha, S0_func_local, V_upper
            )
            ax.plot(x_grid, H_vals, label=f'$\\lambda$={lam}')
    else:
        ax.text(0.5, 0.5, 'compute_H_tvar not available',
                transform=ax.transAxes, ha='center', va='center')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('H(x)')
    ax.set_title('Net Price Function H(x; S_Q, lambda)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)

    # G* for different beta at fixed lambda (case ii, lambda=13)
    ax = axes[2]
    lam_plot = 13.0
    ax.plot(x_grid, np.array([S0(x, M) for x in x_grid]), 'k--', label='S0', linewidth=1.5)
    for beta in beta_values:
        if (lam_plot, beta) in results.get('G_star', {}):
            G = results['G_star'][(lam_plot, beta)]
            ax.plot(x_grid, G, label=f'$\\beta$={beta}')
    ax.axvline(x=V_upper, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('G*(x; beta)')
    ax.set_title(f'G* for Different $\\beta$ ($\\lambda$={lam_plot}, Case ii)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)

    # Regime boundary visualization
    ax = axes[3]
    # boundary1: lambda where (1+lambda)*(1+theta) = 1/(1-alpha)
    boundary1 = 1.0 / ((1.0 - alpha) * (1.0 + theta)) - 1.0
    S_Q_at_V_upper = float(S0(V_upper, M))
    # boundary2 typically >> 50 (S_Q at V_upper is ~1e-8), mark as "inf"
    lambda_fine = np.linspace(0, 55, 500)
    cases_for_lambda = [case_type(lam, theta, alpha, S_Q_at_V_upper) for lam in lambda_fine]
    ax.axvline(
        boundary1, color='blue', linestyle='--',
        label=f'Case (i)/(ii) boundary: $\\lambda$≈{boundary1:.2f}'
    )
    ax.plot(lambda_fine, cases_for_lambda, 'g-', linewidth=2, label='Active case')
    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('Case')
    ax.set_yticks([1, 2, 3])
    ax.set_ylim(0.5, 3.5)
    ax.set_title('Case Classification vs. $\\lambda$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp3_tvar_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print('\nFigures saved to results/')
    return results


if __name__ == '__main__':
    main()
