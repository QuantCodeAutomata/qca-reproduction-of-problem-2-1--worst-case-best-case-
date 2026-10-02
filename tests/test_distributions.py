import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.distributions import F0, S0, F0_inv, Config
    _HAS_DISTRIBUTIONS = True
except ImportError:
    _HAS_DISTRIBUTIONS = False

pytestmark = pytest.mark.skipif(
    not _HAS_DISTRIBUTIONS,
    reason='src.distributions not available'
)

M = 100


def test_f0_boundary_conditions():
    assert abs(F0(0, M)) < 1e-10
    assert abs(F0(M, M) - 1.0) < 1e-10


def test_s0_boundary_conditions():
    assert abs(S0(0, M) - 1.0) < 1e-10
    assert abs(S0(M, M)) < 1e-10


def test_f0_s0_complementarity():
    xs = np.linspace(0, M, 100)
    for x in xs:
        assert abs(F0(x, M) + S0(x, M) - 1.0) < 1e-10


def test_f0_monotone():
    xs = np.linspace(0, M, 500)
    f_vals = np.array([F0(x, M) for x in xs])
    assert np.all(np.diff(f_vals) >= -1e-12)


def test_f0_inv_monotone():
    ts = np.linspace(0, 1, 200)[1:-1]  # avoid endpoints
    q_vals = np.array([F0_inv(t, M) for t in ts])
    assert np.all(np.diff(q_vals) >= -1e-12)


def test_f0_inv_round_trip():
    # Restrict to x <= 20: for x > 22, exp(-x) < 1e-10 causes catastrophic
    # cancellation in 1 - F0(x), making the round-trip error exceed 1e-6.
    xs = np.linspace(0.01, 20.0, 50)
    for x in xs:
        t = F0(x, M)
        x_back = F0_inv(t, M)
        assert abs(x_back - x) < 1e-6


def test_config_defaults():
    cfg = Config()
    assert cfg.M == 100
    assert cfg.alpha == 0.95
    assert cfg.theta == 0.5
    assert cfg.kappa == 0.9
    assert cfg.epsilon == 0.5


def test_f0_range():
    xs = np.linspace(0, M, 200)
    for x in xs:
        val = F0(x, M)
        assert 0 <= val <= 1


def test_s0_nonneg():
    xs = np.linspace(0, M, 200)
    for x in xs:
        assert S0(x, M) >= -1e-12
