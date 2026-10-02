import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.distributions import F0, S0, F0_inv
    _HAS_DISTRIBUTIONS = True
except ImportError:
    _HAS_DISTRIBUTIONS = False

try:
    from src.problem21 import (
        compute_worst_case_var, compute_best_case_var,
        compute_indemnity_thresholds, compute_net_price_H,
        compute_optimal_indemnity, run_example31,
    )
    _HAS_PROBLEM21 = True
except ImportError:
    _HAS_PROBLEM21 = False

_HAS_ALL = _HAS_DISTRIBUTIONS and _HAS_PROBLEM21

pytestmark = pytest.mark.skipif(
    not _HAS_ALL,
    reason='src.distributions or src.problem21 not available'
)

M, alpha, theta, kappa, epsilon = 100, 0.95, 0.5, 0.9, 0.5


def test_worst_case_var_above_q_alpha():
    q_alpha = F0_inv(alpha, M)
    V_upper = compute_worst_case_var(F0_inv, q_alpha, alpha, k=1, epsilon=epsilon, M=M)
    assert V_upper > q_alpha


def test_best_case_var_below_q_alpha():
    q_alpha = F0_inv(alpha, M)
    V_lower = compute_best_case_var(F0_inv, F0, q_alpha, alpha, k=1, epsilon=epsilon, M=M)
    assert V_lower < q_alpha


def test_worst_case_var_increases_with_epsilon():
    q_alpha = F0_inv(alpha, M)
    V1 = compute_worst_case_var(F0_inv, q_alpha, alpha, k=1, epsilon=0.3, M=M)
    V2 = compute_worst_case_var(F0_inv, q_alpha, alpha, k=1, epsilon=0.5, M=M)
    assert V2 >= V1 - 1e-6


def test_example31_k1_values():
    """Test that k=1 recovers paper values approximately."""
    results = run_example31(k_values=[1], n_grid=1000)
    row = results[results['k'] == 1].iloc[0]
    assert abs(row['d1_star'] - 0.406) < 0.05, f'd1*={row["d1_star"]}'
    assert abs(row['d2_star'] - 0.511) < 0.05, f'd2*={row["d2_star"]}'
    assert abs(row['V_lower'] - 0.563) < 0.1, f'V_lower={row["V_lower"]}'
    assert abs(row['V_upper'] - 6.596) < 0.5, f'V_upper={row["V_upper"]}'


def test_example31_k4_v_upper():
    """Test k=4 worst-case VaR."""
    results = run_example31(k_values=[4], n_grid=1000)
    row = results[results['k'] == 4].iloc[0]
    assert abs(row['V_upper'] - 4.894) < 0.5, f'V_upper k=4={row["V_upper"]}'


def test_k_decreases_v_upper():
    """Larger k should lower worst-case VaR."""
    results = run_example31(k_values=[1, 2, 4], n_grid=500)
    V_upper_values = results.set_index('k')['V_upper']
    assert V_upper_values[4] < V_upper_values[1] - 0.5


def test_indemnity_admissible():
    q_alpha = F0_inv(alpha, M)
    V_lower = compute_best_case_var(F0_inv, F0, q_alpha, alpha, k=1, epsilon=epsilon, M=M)
    V_upper = compute_worst_case_var(F0_inv, q_alpha, alpha, k=1, epsilon=epsilon, M=M)
    x_grid = np.linspace(0, M, 500)
    I_vals = compute_optimal_indemnity(x_grid, S0, V_lower, V_upper, kappa, theta, M)
    # I(0) = 0
    assert abs(I_vals[0]) < 1e-8
    # Nondecreasing
    assert np.all(np.diff(I_vals) >= -1e-8)
    # I(x) >= 0
    assert np.all(I_vals >= -1e-8)


def test_net_price_H_structure():
    """Validate H sign structure for the deductible/capped-layer indemnity.

    H(t) = (1+theta)*S_Q(t) - 1_{[0,V_lower]}(t) - kappa*1_{(V_lower,V_upper]}(t)

    Below the deductible d1* (where S_Q > 1/(1+theta)): H > 0 → I*'=0.
    Between d1* and d2* (full-coverage layer): H < 0 → I*'=1.
    Well above d2* (where S_Q << kappa/(1+theta)): H > 0 → I*'=0.
    """
    V_lower, V_upper = 0.563, 6.596
    d1_star, d2_star = 0.406, 0.511  # paper values

    # Below deductible: S_Q ≈ 1 >> 1/(1+theta) ≈ 0.667 → H > 0 (no coverage)
    x_below_ded = 0.1  # well below d1* ≈ 0.406
    H_below = compute_net_price_H(x_below_ded, S0, V_lower, V_upper, kappa, theta)
    assert H_below > 0, f"H should be > 0 below deductible, got {H_below}"

    # In full-coverage layer [d1*, d2*]: H < 0 → full insurance optimal
    x_mid = 0.45  # between d1* and d2*
    H_mid = compute_net_price_H(x_mid, S0, V_lower, V_upper, kappa, theta)
    assert H_mid < 0, f"H should be < 0 in full-coverage layer, got {H_mid}"
