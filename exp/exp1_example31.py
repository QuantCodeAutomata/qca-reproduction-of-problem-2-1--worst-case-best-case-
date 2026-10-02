"""
Experiment 1: Reproduce Problem 2.1 results.
Reproduces Example 3.1: d1*~0.406, d2*~0.511, V_lower~0.563, V_upper~6.596 for k=1.
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
    from src.distributions import F0, S0, F0_inv
except ImportError as e:
    print(f'Warning: could not import src.distributions: {e}')
    F0 = S0 = F0_inv = None

try:
    from src.bregman import phi, phi_prime, B_phi, bw_divergence_quantile
except ImportError as e:
    print(f'Warning: could not import src.bregman: {e}')
    phi = phi_prime = B_phi = bw_divergence_quantile = None

try:
    from src.problem21 import (
        compute_worst_case_var, compute_best_case_var,
        compute_indemnity_thresholds, compute_net_price_H,
        compute_optimal_indemnity, run_example31,
    )
except ImportError as e:
    print(f'Warning: could not import src.problem21: {e}')
    compute_worst_case_var = compute_best_case_var = None
    compute_indemnity_thresholds = compute_net_price_H = None
    compute_optimal_indemnity = run_example31 = None

try:
    from src.validation import check_admissible_indemnity, check_bw_feasibility
except ImportError as e:
    print(f'Warning: could not import src.validation: {e}')
    check_admissible_indemnity = check_bw_feasibility = None

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    """Run Example 3.1 and save results."""
    print('=== Experiment 1: Problem 2.1 Worst/Best-case VaR and Optimal Indemnity ===')

    if run_example31 is None:
        print('ERROR: src.problem21 not available. Cannot run experiment.')
        return None

    k_values = [1, 2, 3, 4]
    M, alpha, theta, kappa, epsilon = 100, 0.95, 0.5, 0.9, 0.5
    n_grid = 2000

    # Run full example
    results_df = run_example31(
        k_values=k_values, M=M, alpha=alpha, theta=theta,
        kappa=kappa, epsilon=epsilon, n_grid=n_grid
    )

    print('\nExample 3.1 Results:')
    print(results_df.to_string(index=False, float_format='{:.4f}'.format))

    # Save table
    results_df.to_csv(RESULTS_DIR / 'exp1_example31_results.csv', index=False)

    # Verification against paper values
    k1_row = results_df[results_df['k'] == 1].iloc[0]
    print(f'\nVerification for k=1:')
    print(f'  d1*  = {k1_row["d1_star"]:.4f} (paper: ~0.406)')
    print(f'  d2*  = {k1_row["d2_star"]:.4f} (paper: ~0.511)')
    print(f'  V_lower = {k1_row["V_lower"]:.4f} (paper: ~0.563)')
    print(f'  V_upper = {k1_row["V_upper"]:.4f} (paper: ~6.596)')

    k4_row = results_df[results_df['k'] == 4].iloc[0]
    print(f'\nVerification for k=4:')
    print(f'  V_upper = {k4_row["V_upper"]:.4f} (paper: ~4.894)')

    # Generate figures
    x_grid = np.linspace(0, 20, 1000)  # Focus on [0,20] for clarity

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: phi(x;k)
    ax = axes[0]
    if phi is not None and F0_inv is not None:
        for k in k_values:
            q_alpha = F0_inv(alpha, M)
            y = phi(x_grid, q_alpha, k)
            ax.plot(x_grid, y, label=f'k={k}')
        ax.axvline(x=F0_inv(alpha, M), color='gray', linestyle='--', alpha=0.5, label=r'$q_\alpha$')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\phi(x;k)$')
    ax.set_title('Bregman Generator')
    ax.legend()
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)

    # Right panel: optimal indemnity I*(x)
    ax = axes[1]
    x_fine = np.linspace(0, 20, 2000)
    if compute_optimal_indemnity is not None and S0 is not None:
        for k in k_values:
            row = results_df[results_df['k'] == k].iloc[0]
            I_vals = compute_optimal_indemnity(
                x_fine, S0, row['V_lower'], row['V_upper'], kappa, theta, M
            )
            ax.plot(x_fine, I_vals, label=f'k={k}')
    ax.plot(x_fine, x_fine, 'k--', alpha=0.3, label='full coverage')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$I^*(x)$')
    ax.set_title('Optimal Indemnity')
    ax.legend()
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.1, 15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp1_phi_and_indemnity.png', dpi=150, bbox_inches='tight')
    plt.close()

    # H(x) sign map
    fig, ax = plt.subplots(figsize=(10, 5))
    x_plot = np.linspace(0, 15, 1000)
    if compute_net_price_H is not None and S0 is not None:
        for k in k_values:
            row = results_df[results_df['k'] == k].iloc[0]
            H_vals = np.array([
                compute_net_price_H(xi, S0, row['V_lower'], row['V_upper'], kappa, theta)
                for xi in x_plot
            ])
            ax.plot(x_plot, H_vals, label=f'k={k}')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('H(x)')
    ax.set_title('Net Price Function H(x) for Different k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp1_net_price_H.png', dpi=150, bbox_inches='tight')
    plt.close()

    print('\nFigures saved to results/')
    return results_df


if __name__ == '__main__':
    main()
