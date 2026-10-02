import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Core interface imports ---
try:
    from src.distributions import F0_inv, F0
    _HAS_DISTRIBUTIONS = True
except ImportError:
    _HAS_DISTRIBUTIONS = False

try:
    from src.bregman import (
        phi, phi_prime, B_phi,
        bw_divergence_quantile,
        phi_entropy, phi_entropy_prime,
        phi_entropy_prime_inv, B_phi_entropy,
    )
    _HAS_BREGMAN_CORE = True
except ImportError:
    _HAS_BREGMAN_CORE = False

# --- Extended functions not in the formal interface; import separately ---
try:
    from src.bregman import phi_second
    _HAS_PHI_SECOND = True
except (ImportError, AttributeError):
    phi_second = None
    _HAS_PHI_SECOND = False

try:
    from src.bregman import phi_entropy_second
    _HAS_PHI_ENTROPY_SECOND = True
except (ImportError, AttributeError):
    phi_entropy_second = None
    _HAS_PHI_ENTROPY_SECOND = False

try:
    from src.bregman import bw_divergence_quantile_entropy
    _HAS_BW_ENTROPY = True
except (ImportError, AttributeError):
    bw_divergence_quantile_entropy = None
    _HAS_BW_ENTROPY = False

_HAS_ALL_CORE = _HAS_DISTRIBUTIONS and _HAS_BREGMAN_CORE

pytestmark = pytest.mark.skipif(
    not _HAS_ALL_CORE,
    reason='src.distributions or src.bregman core not available'
)

M, alpha = 100, 0.95


@pytest.fixture(scope='module')
def q_alpha():
    return F0_inv(alpha, M)


def test_phi_continuity_at_q_alpha(q_alpha):
    for k in [1, 2, 4]:
        left = phi(q_alpha - 1e-9, q_alpha, k)
        right = phi(q_alpha + 1e-9, q_alpha, k)
        assert abs(left - right) < 1e-6


def test_phi_prime_continuity_at_q_alpha(q_alpha):
    for k in [1, 2, 4]:
        left = phi_prime(q_alpha - 1e-9, q_alpha, k)
        right = phi_prime(q_alpha + 1e-9, q_alpha, k)
        assert abs(left - right) < 1e-5


def test_b_phi_nonneg(q_alpha):
    for k in [1, 2, 4]:
        for x in [0.1, 1.0, 3.0, 8.0]:
            for y in [0.5, 2.0, 5.0]:
                val = B_phi(x, y, q_alpha, k)
                assert val >= -1e-10, f'B_phi({x},{y},k={k})={val} < 0'


def test_b_phi_zero_diagonal(q_alpha):
    for k in [1, 2, 4]:
        for x in [0.5, 1.0, q_alpha, q_alpha + 1.0]:
            val = B_phi(x, x, q_alpha, k)
            assert abs(val) < 1e-10


def test_bw_divergence_identity(q_alpha):
    for k in [1, 2]:
        val = bw_divergence_quantile(F0_inv, F0_inv, q_alpha, k)
        assert abs(val) < 1e-6


def test_phi_entropy_nonneg():
    xs = np.linspace(0, 10, 50)
    for x in xs:
        assert phi_entropy(x) >= -1e-12


def test_phi_entropy_prime_inv_roundtrip():
    xs = np.linspace(0, 10, 50)
    for x in xs:
        u = phi_entropy_prime(x)
        x_back = phi_entropy_prime_inv(u)
        assert abs(x_back - x) < 1e-8


def test_b_phi_entropy_nonneg():
    for x in [0.1, 1.0, 5.0, 10.0]:
        for y in [0.5, 2.0, 7.0]:
            val = B_phi_entropy(x, y)
            assert val >= -1e-10


@pytest.mark.skipif(not _HAS_BW_ENTROPY, reason='bw_divergence_quantile_entropy not available')
def test_bw_entropy_identity():
    val = bw_divergence_quantile_entropy(F0_inv, F0_inv, M)
    assert abs(val) < 1e-5
