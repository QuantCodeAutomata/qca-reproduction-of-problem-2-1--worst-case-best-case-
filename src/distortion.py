"""Distortion risk measures and insurance premium computations.

All integration uses ``scipy.integrate.quad`` (or the trapezoidal rule on
pre-computed grids) to maintain numerical consistency with the rest of the
project.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.integrate import quad

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# TVaR distortion and its generalised inverse
# ---------------------------------------------------------------------------


def tvar_distortion(t: ArrayLike, alpha: float) -> ArrayLike:
    """TVaR distortion function :math:`g(t) = \\min\\!\\left(\\frac{t}{1-\\alpha}, 1\\right)`.

    Parameters
    ----------
    t : float or ndarray
        Probability argument(s) in [0, 1].
    alpha : float
        Confidence level in (0, 1).

    Returns
    -------
    float or ndarray
        Distorted probability value(s) in [0, 1].
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    result = np.minimum(t / (1.0 - alpha), 1.0)
    return float(result[0]) if scalar else result


def tvar_distortion_inv(u: ArrayLike, alpha: float) -> ArrayLike:
    """Generalised inverse of the TVaR distortion.

    The generalised inverse is defined as
    :math:`g^{-1}(u) = \\inf\\{t \\in [0,1] : g(t) \\ge u\\}`.

    For the TVaR distortion this evaluates to:

    .. math::
        g^{-1}(u) = (1 - \\alpha)\\,u, \\quad u \\in [0, 1].

    Note that :math:`g^{-1}(1) = 1 - \\alpha` (not 1), which reflects the
    flat portion of *g* near 1.

    Parameters
    ----------
    u : float or ndarray
        Value(s) in [0, 1].
    alpha : float
        Confidence level in (0, 1).

    Returns
    -------
    float or ndarray
        Pre-image value(s) in :math:`[0, 1 - \\alpha]`.
    """
    u = np.asarray(u, dtype=float)
    scalar = u.ndim == 0
    u = np.atleast_1d(u)
    result = (1.0 - alpha) * u
    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Generic distortion risk measure
# ---------------------------------------------------------------------------


def distortion_risk(
    S: Callable[[float], float],
    g: Callable[[float], float],
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Distortion risk measure :math:`\\rho_g(X) = \\int_0^M g(S(x))\\,\\mathrm{d}x`.

    Parameters
    ----------
    S : callable
        Survival function of the loss *X* on [0, M].
    g : callable
        Distortion function; maps [0, 1] to [0, 1].
    M : float
        Upper bound of the loss support.
    quad_abs_tol : float
        Absolute tolerance for ``scipy.integrate.quad``.
    quad_rel_tol : float
        Relative tolerance for ``scipy.integrate.quad``.

    Returns
    -------
    float
        Risk measure value.
    """

    def integrand(x: float) -> float:
        return float(g(S(x)))

    val, _ = quad(
        integrand,
        0.0,
        M,
        limit=200,
        epsabs=quad_abs_tol,
        epsrel=quad_rel_tol,
    )
    return float(val)


# ---------------------------------------------------------------------------
# Insurance premium
# ---------------------------------------------------------------------------


def premium_from_indemnity(
    I_values: np.ndarray,
    x_grid: np.ndarray,
    S_Q_func: Callable[[float], float],
    theta: float,
) -> float:
    """Loaded premium :math:`\\pi(I) = (1+\\theta)\\int_0^M S_Q(x)\\,I'(x)\\,\\mathrm{d}x`.

    The integral is evaluated via the trapezoidal rule using finite-difference
    approximations of *I'* on the supplied grid.

    Parameters
    ----------
    I_values : ndarray, shape (n,)
        Indemnity function evaluated at the points in *x_grid*.  Must satisfy
        *I_values[0] = 0* and be non-decreasing with slope in [0, 1].
    x_grid : ndarray, shape (n,)
        Strictly increasing grid of loss values covering [0, M].
    S_Q_func : callable
        Survival function of the pricing measure *Q* on [0, M].
    theta : float
        Safety loading factor (≥ 0).

    Returns
    -------
    float
        Loaded premium.

    Notes
    -----
    The integration-by-parts identity

    .. math::
        E_Q[I(X)] = \\int_0^M S_Q(x)\\,I'(x)\\,\\mathrm{d}x

    holds when *I(0) = 0* and :math:`I(M)\\,S_Q(M) = 0`.
    """
    I_values = np.asarray(I_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    dx = np.diff(x_grid)
    dI = np.diff(I_values)
    I_prime = dI / dx  # finite-difference approximation of I' at midpoints

    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    S_Q_mid = np.array([S_Q_func(xi) for xi in x_mid], dtype=float)

    integral = float(np.sum(S_Q_mid * I_prime * dx))
    return (1.0 + theta) * integral


def premium_scalar(
    I_prime_func: Callable[[float], float],
    S_Q: Callable[[float], float],
    theta: float,
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Loaded premium using ``scipy.integrate.quad`` for high accuracy.

    Computes

    .. math::
        \\pi(I) = (1 + \\theta)\\int_0^M S_Q(x)\\,I'(x)\\,\\mathrm{d}x.

    Parameters
    ----------
    I_prime_func : callable
        Derivative *I'* of the indemnity function; maps *x* in [0, M] to
        a value in [0, 1].
    S_Q : callable
        Survival function of the pricing measure *Q* on [0, M].
    theta : float
        Safety loading factor (≥ 0).
    M : float
        Upper bound of the loss support.
    quad_abs_tol : float
        Absolute tolerance for ``scipy.integrate.quad``.
    quad_rel_tol : float
        Relative tolerance for ``scipy.integrate.quad``.

    Returns
    -------
    float
        Loaded premium.
    """

    def integrand(x: float) -> float:
        return float(S_Q(x)) * float(I_prime_func(x))

    val, _ = quad(
        integrand,
        0.0,
        M,
        limit=200,
        epsabs=quad_abs_tol,
        epsrel=quad_rel_tol,
    )
    return (1.0 + theta) * float(val)
