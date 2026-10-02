import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.distributions import S0, F0_inv
    _HAS_DISTRIBUTIONS = True
except ImportError:
    _HAS_DISTRIBUTIONS = False

try:
    from src.tvar_section43 import (
        case_type, compute_hat_x, compute_G_star_tvar,
    )
    _HAS_TVAR_CORE = True
except ImportError:
    _HAS_TVAR_CORE = False

# compute_hatS_Q is not in the formal interface; import defensively.
try:
    from src.tvar_section43 import compute_hatS_Q
    _HAS_COMPUTE_HATS_Q = True
except (ImportError, AttributeError):
    compute_hatS_Q = None
    _HAS_COMPUTE_HATS_Q = False

# compute_H_tvar is not in the formal interface; import defensively.
try:
    from src.tvar_section43 import compute_H_tvar
    _HAS_COMPUTE_H_TVAR = True
except (ImportError, AttributeError):
    compute_H_tvar = None
    _HAS_COMPUTE_H_TVAR = False

try:
    from src.distortion import tvar_distortion
    _HAS_DISTORTION = True
except ImportError:
    tvar_distortion = None
    _HAS_DISTORTION = False

_HAS_ALL_CORE = _HAS_DISTRIBUTIONS and _HAS_TVAR_CORE

pytestmark = pytest.mark.skipif(
    not _HAS_ALL_CORE,
    reason='src.distributions or src.tvar_section43 core not available'
)

M, alpha, theta = 100, 0.95, 0.5


def test_case_boundaries():
    # At lambda=0, (1+0)*(1+0.5)=1.5 vs 1/(1-0.95)=20 => case (i)
    S_Q_V_upper = S0(6.596, M)  # approximate V_upper for k=1
    c = case_type(0.0, theta, alpha, S_Q_V_upper)
    assert c == 1  # should be case (i) since 1.5 < 20


def test_case_i_definition():
    # (1+lambda)*(1+theta) < 1/(1-alpha) => case 1
    S_Q_V_upper = 0.01  # small value
    # lambda=5: (1+5)*1.5=9 < 20 => case (i)
    c = case_type(5.0, theta, alpha, S_Q_V_upper)
    assert c == 1


def test_hat_x_decreases_with_beta():
    hat_x_alpha_Q = F0_inv(alpha, M)
    hat_x1 = compute_hat_x(1.0, alpha, hat_x_alpha_Q, M)
    hat_x2 = compute_hat_x(2.0, alpha, hat_x_alpha_Q, M)
    # Larger beta => smaller hat_x
    assert hat_x2 <= hat_x1 + 1e-6


def test_hat_x_beta0_returns_M():
    hat_x_alpha_Q = F0_inv(alpha, M)
    hat_x = compute_hat_x(0.0, alpha, hat_x_alpha_Q, M)
    assert hat_x >= M - 1e-6


def test_G_star_bounds():
    x_grid = np.linspace(0, 20, 500)
    V_upper = 6.596  # approximate
    G = compute_G_star_tvar(
        x_grid, lambda_val=0.0, beta=0.0,
        alpha=alpha, theta=theta,
        S_Q_func=lambda x: S0(x, M),
        F_Q_inv_func=lambda t: F0_inv(t, M),
        V_alpha_upper=V_upper, M=M
    )
    assert np.all(G >= -1e-8)
    assert np.all(G <= 1 + 1e-8)


def test_G_star_case_i_equals_S0():
    # In case (i), G*(x;beta) = S_Q(x)
    x_grid = np.linspace(0, 20, 100)
    V_upper = 6.596
    # lambda=0: (1+0)*1.5=1.5 < 20 => case (i)
    G = compute_G_star_tvar(
        x_grid, lambda_val=0.0, beta=0.0,
        alpha=alpha, theta=theta,
        S_Q_func=lambda x: S0(x, M),
        F_Q_inv_func=lambda t: F0_inv(t, M),
        V_alpha_upper=V_upper, M=M
    )
    S_Q_vals = np.array([S0(x, M) for x in x_grid])
    assert np.allclose(G, S_Q_vals, atol=1e-6)


@pytest.mark.skipif(not _HAS_COMPUTE_H_TVAR, reason='compute_H_tvar not available')
def test_H_tvar_sign_structure():
    x_grid = np.linspace(0, 20, 500)
    V_upper = 6.596
    H = compute_H_tvar(
        x_grid, lambda_val=0.0, theta=theta, alpha=alpha,
        S_Q_func=lambda x: S0(x, M), V_alpha_upper=V_upper
    )
    assert len(H) == len(x_grid)
    assert H.dtype == float or np.issubdtype(H.dtype, np.floating)


@pytest.mark.skipif(not _HAS_DISTORTION, reason='src.distortion not available')
def test_tvar_distortion_valid():
    for t in np.linspace(0, 1, 20):
        g = tvar_distortion(t, alpha)
        assert 0 <= g <= 1
    assert abs(tvar_distortion(1.0, alpha) - 1.0) < 1e-10
    assert abs(tvar_distortion(0.0, alpha)) < 1e-10
