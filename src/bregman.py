"""Bregman divergences and Bregman-Wasserstein (BW) divergences.

Two generator families are implemented:

1. **Piecewise quadratic generator** (used in Example 3.1 and related results):

   .. math::
       \\phi(x; k) = \\begin{cases}
           x^2 & x \\in [0, q_\\alpha) \\\\
           q_\\alpha^2 + 2q_\\alpha(x - q_\\alpha) + k(x - q_\\alpha)^2
               & x \\in [q_\\alpha, M]
       \\end{cases}

   For *k = 1* this globally reduces to :math:`\\phi(x) = x^2`.

2. **Entropy-like generator** (used in Example 4.1):

   .. math::
       \\phi_{\\mathrm{ent}}(x) = (x+1)\\log(x+1).
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.integrate import quad, dblquad

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Piecewise quadratic generator
# ---------------------------------------------------------------------------


def phi(x: ArrayLike, q_alpha: float, k: float) -> ArrayLike:
    """Piecewise quadratic Bregman generator.

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s) in [0, M].
    q_alpha : float
        Quantile threshold :math:`q_\\alpha = F_0^{-1}(\\alpha)`.
    k : float
        Curvature multiplier on :math:`[q_\\alpha, M]`.  *k = 1* recovers the
        globally quadratic :math:`\\phi(x) = x^2`.

    Returns
    -------
    float or ndarray
        Generator value(s).
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.where(
        x < q_alpha,
        x ** 2,
        q_alpha ** 2 + 2.0 * q_alpha * (x - q_alpha) + k * (x - q_alpha) ** 2,
    )
    return float(result[0]) if scalar else result


def phi_prime(x: ArrayLike, q_alpha: float, k: float) -> ArrayLike:
    """First derivative of the piecewise quadratic generator.

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s) in [0, M].
    q_alpha : float
        Quantile threshold.
    k : float
        Curvature multiplier on :math:`[q_\\alpha, M]`.

    Returns
    -------
    float or ndarray
        :math:`\\phi'(x; k)`.

    Notes
    -----
    .. math::
        \\phi'(x; k) = \\begin{cases}
            2x & x \\in [0, q_\\alpha) \\\\
            2q_\\alpha + 2k(x - q_\\alpha) & x \\in [q_\\alpha, M]
        \\end{cases}
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.where(
        x < q_alpha,
        2.0 * x,
        2.0 * q_alpha + 2.0 * k * (x - q_alpha),
    )
    return float(result[0]) if scalar else result


def phi_second(x: ArrayLike, q_alpha: float, k: float) -> ArrayLike:
    """Second derivative of the piecewise quadratic generator.

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s) in [0, M].
    q_alpha : float
        Quantile threshold.
    k : float
        Curvature multiplier on :math:`[q_\\alpha, M]`.

    Returns
    -------
    float or ndarray
        :math:`\\phi''(x; k)`.

    Notes
    -----
    .. math::
        \\phi''(x; k) = \\begin{cases}
            2 & x \\in [0, q_\\alpha) \\\\
            2k & x \\in [q_\\alpha, M]
        \\end{cases}
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.where(x < q_alpha, 2.0, 2.0 * k)
    return float(result[0]) if scalar else result


def B_phi(
    x: ArrayLike,
    y: ArrayLike,
    q_alpha: float,
    k: float,
) -> ArrayLike:
    """Bregman divergence induced by the piecewise quadratic generator.

    .. math::
        B_\\phi(x, y) = \\phi(x) - \\phi(y) - \\phi'(y)\\,(x - y).

    Parameters
    ----------
    x : float or ndarray
        "True" point(s).
    y : float or ndarray
        "Reference" point(s).
    q_alpha : float
        Quantile threshold.
    k : float
        Curvature multiplier.

    Returns
    -------
    float or ndarray
        Non-negative divergence value(s).
    """
    return phi(x, q_alpha, k) - phi(y, q_alpha, k) - phi_prime(y, q_alpha, k) * (x - y)


# ---------------------------------------------------------------------------
# BW divergence — quantile form
# ---------------------------------------------------------------------------


def bw_divergence_quantile(
    F1_inv: Callable[[float], float],
    F2_inv: Callable[[float], float],
    q_alpha: float,
    k: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Bregman-Wasserstein divergence via the quantile (copula) representation.

    .. math::
        D_{\\phi}(F_1 \\| F_2)
        = \\int_0^1 B_\\phi\\!\\bigl(F_1^{-1}(t),\\, F_2^{-1}(t)\\bigr)\\,\\mathrm{d}t.

    Parameters
    ----------
    F1_inv : callable
        Quantile function of the first distribution; maps *t* in (0, 1) to
        a loss value.
    F2_inv : callable
        Quantile function of the reference distribution.
    q_alpha : float
        Quantile threshold for the generator.
    k : float
        Generator curvature multiplier.
    quad_abs_tol : float
        Absolute quadrature tolerance.
    quad_rel_tol : float
        Relative quadrature tolerance.

    Returns
    -------
    float
        BW divergence value (≥ 0).
    """

    def integrand(t: float) -> float:
        return float(B_phi(F1_inv(t), F2_inv(t), q_alpha, k))

    val, _ = quad(
        integrand,
        0.0,
        1.0,
        limit=200,
        epsabs=quad_abs_tol,
        epsrel=quad_rel_tol,
    )
    return float(val)


# ---------------------------------------------------------------------------
# BW divergence — survival form (Lemma 2.1)
# ---------------------------------------------------------------------------


def bw_divergence_survival(
    S1: Callable[[float], float],
    S2: Callable[[float], float],
    phi_prime_func: Callable[[float], float],
    phi_second_func: Callable[[float], float],
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """Bregman-Wasserstein divergence via the survival-function form (Lemma 2.1).

    The formula is:

    .. math::
        D_\\phi(F_1 \\| F_2) =
        \\int_0^M [\\phi'(x) - \\phi'(0)]\\,S_1(x)\\,\\mathrm{d}x
        + \\int_0^M \\phi''(x)\\,x\\,S_2(x)\\,\\mathrm{d}x
        - \\int_0^M\\!\\int_0^M
              \\phi''(y)\\,\\min(S_2(y),\\,S_1(x))\\,\\mathrm{d}y\\,\\mathrm{d}x.

    Parameters
    ----------
    S1 : callable
        Survival function of the first distribution on [0, M].
    S2 : callable
        Survival function of the reference distribution on [0, M].
    phi_prime_func : callable
        First derivative :math:`\\phi'` of the generator.
    phi_second_func : callable
        Second derivative :math:`\\phi''` of the generator.
    M : float
        Upper bound of the support.
    quad_abs_tol : float
        Absolute quadrature tolerance (passed to ``quad`` and ``dblquad``).
    quad_rel_tol : float
        Relative quadrature tolerance.

    Returns
    -------
    float
        BW divergence value.
    """
    phi_prime_0 = phi_prime_func(0.0)

    def term1_integrand(x: float) -> float:
        return (phi_prime_func(x) - phi_prime_0) * S1(x)

    def term2_integrand(x: float) -> float:
        return phi_second_func(x) * x * S2(x)

    def double_integrand(y: float, x: float) -> float:
        return phi_second_func(y) * min(S2(y), S1(x))

    term1, _ = quad(
        term1_integrand, 0.0, M,
        limit=200, epsabs=quad_abs_tol, epsrel=quad_rel_tol,
    )
    term2, _ = quad(
        term2_integrand, 0.0, M,
        limit=200, epsabs=quad_abs_tol, epsrel=quad_rel_tol,
    )
    term3, _ = dblquad(
        double_integrand,
        0.0, M,
        0.0, M,
        epsabs=quad_abs_tol,
        epsrel=quad_rel_tol,
    )
    return float(term1 + term2 - term3)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def validate_bw_consistency(
    F1_inv: Callable[[float], float],
    S1: Callable[[float], float],
    F2_inv: Callable[[float], float],
    S2: Callable[[float], float],
    q_alpha: float,
    k: float,
    M: float,
    tol: float = 1e-4,
) -> dict:
    """Check that the quantile-form and survival-form BW divergences agree.

    Parameters
    ----------
    F1_inv : callable
        Quantile function of F1.
    S1 : callable
        Survival function of F1.
    F2_inv : callable
        Quantile function of F2 (reference).
    S2 : callable
        Survival function of F2 (reference).
    q_alpha : float
        Quantile threshold for the generator.
    k : float
        Generator curvature multiplier.
    M : float
        Upper bound of the support.
    tol : float
        Absolute tolerance for declaring the two values consistent.

    Returns
    -------
    dict
        Keys:

        * ``quantile_form`` – value from :func:`bw_divergence_quantile`
        * ``survival_form`` – value from :func:`bw_divergence_survival`
        * ``absolute_difference`` – |quantile_form - survival_form|
        * ``consistent`` – bool, True if difference ≤ *tol*
    """

    def phi_prime_func(x: float) -> float:
        return float(phi_prime(x, q_alpha, k))

    def phi_second_func(x: float) -> float:
        return float(phi_second(x, q_alpha, k))

    qf = bw_divergence_quantile(F1_inv, F2_inv, q_alpha, k)
    sf = bw_divergence_survival(S1, S2, phi_prime_func, phi_second_func, M)
    diff = abs(qf - sf)
    return {
        "quantile_form": qf,
        "survival_form": sf,
        "absolute_difference": diff,
        "consistent": diff <= tol,
    }


# ---------------------------------------------------------------------------
# Entropy-like generator
# ---------------------------------------------------------------------------


def phi_entropy(x: ArrayLike) -> ArrayLike:
    """Entropy-like Bregman generator: :math:`\\phi_{\\rm ent}(x) = (x+1)\\log(x+1)`.

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).  Must satisfy *x > -1*.

    Returns
    -------
    float or ndarray
        Generator value(s).
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = (x + 1.0) * np.log(x + 1.0)
    return float(result[0]) if scalar else result


def phi_entropy_prime(x: ArrayLike) -> ArrayLike:
    """First derivative of the entropy-like generator.

    .. math::
        \\phi_{\\rm ent}'(x) = \\log(x+1) + 1.

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).  Must satisfy *x > -1*.

    Returns
    -------
    float or ndarray
        Derivative value(s).
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.log(x + 1.0) + 1.0
    return float(result[0]) if scalar else result


def phi_entropy_second(x: ArrayLike) -> ArrayLike:
    """Second derivative of the entropy-like generator.

    .. math::
        \\phi_{\\rm ent}''(x) = \\frac{1}{x+1}.

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).  Must satisfy *x > -1*.

    Returns
    -------
    float or ndarray
        Second derivative value(s).
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = 1.0 / (x + 1.0)
    return float(result[0]) if scalar else result


def phi_entropy_prime_inv(u: ArrayLike) -> ArrayLike:
    """Inverse of :math:`\\phi_{\\rm ent}'`.

    Solves :math:`\\phi_{\\rm ent}'(x) = u`, giving:

    .. math::
        (\\phi_{\\rm ent}')^{-1}(u) = e^{u-1} - 1.

    Parameters
    ----------
    u : float or ndarray
        Value(s) in the range of :math:`\\phi_{\\rm ent}'`.

    Returns
    -------
    float or ndarray
        Pre-image value(s).
    """
    u = np.asarray(u, dtype=float)
    scalar = u.ndim == 0
    u = np.atleast_1d(u)
    result = np.exp(u - 1.0) - 1.0
    return float(result[0]) if scalar else result


def B_phi_entropy(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Bregman divergence induced by the entropy-like generator.

    .. math::
        B_{\\phi_{\\rm ent}}(x, y)
        = \\phi_{\\rm ent}(x) - \\phi_{\\rm ent}(y)
          - \\phi_{\\rm ent}'(y)\\,(x - y).

    Parameters
    ----------
    x : float or ndarray
        "True" point(s).  Must satisfy *x > -1*.
    y : float or ndarray
        "Reference" point(s).  Must satisfy *y > -1*.

    Returns
    -------
    float or ndarray
        Non-negative divergence value(s).
    """
    return phi_entropy(x) - phi_entropy(y) - phi_entropy_prime(y) * (x - y)


def bw_divergence_quantile_entropy(
    F1_inv: Callable[[float], float],
    F2_inv: Callable[[float], float],
    M: float,
    quad_abs_tol: float = 1e-10,
    quad_rel_tol: float = 1e-10,
) -> float:
    """BW divergence with the entropy-like generator via the quantile form.

    .. math::
        D_{\\phi_{\\rm ent}}(F_1 \\| F_2)
        = \\int_0^1
          B_{\\phi_{\\rm ent}}\\!\\bigl(F_1^{-1}(t),\\, F_2^{-1}(t)\\bigr)
          \\,\\mathrm{d}t.

    Parameters
    ----------
    F1_inv : callable
        Quantile function of the first distribution.
    F2_inv : callable
        Quantile function of the reference distribution.
    M : float
        Upper bound of the support (used only as documentation context; the
        integral is over :math:`t \\in (0, 1)`).
    quad_abs_tol : float
        Absolute quadrature tolerance.
    quad_rel_tol : float
        Relative quadrature tolerance.

    Returns
    -------
    float
        BW divergence value (≥ 0).
    """

    def integrand(t: float) -> float:
        x_val = F1_inv(t)
        y_val = F2_inv(t)
        return float(B_phi_entropy(x_val, y_val))

    val, _ = quad(
        integrand,
        0.0,
        1.0,
        limit=200,
        epsabs=quad_abs_tol,
        epsrel=quad_rel_tol,
    )
    return float(val)
