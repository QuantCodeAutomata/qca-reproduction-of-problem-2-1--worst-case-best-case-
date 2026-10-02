# Repository: Distributionally Robust Insurance Paper Reproduction

## Project Layout

```
src/              Python package with core mathematical modules
configs/          YAML experiment configuration files
exp/              Experiment runner scripts
data/             Raw / preprocessed data
results/          Output tables and figures
tests/            Unit / integration tests
```

## Source Modules (`src/`)

| File | Purpose |
|---|---|
| `__init__.py` | Empty package marker |
| `distributions.py` | Truncated Exp(1) on [0, M]: `F0`, `S0`, `F0_inv`, `validate_f0`; `Config` dataclass |
| `bregman.py` | Piecewise-quadratic Bregman generator (`phi`, `phi_prime`, `phi_second`, `B_phi`); BW divergence in quantile form (`bw_divergence_quantile`) and survival form (`bw_divergence_survival`); cross-validation helper; entropy generator family (`phi_entropy*`, `B_phi_entropy`, `bw_divergence_quantile_entropy`) |
| `distortion.py` | TVaR distortion `tvar_distortion` / its generalised inverse; `distortion_risk`; premium via trapezoidal rule (`premium_from_indemnity`) and `scipy.integrate.quad` (`premium_scalar`) |
| `validation.py` | Monotonicity checkers, survival function validator, admissible-indemnity checker, BW feasibility, KKT complementarity, `validate_f0_properties` |

## Config Files (`configs/`)

| File | Experiment |
|---|---|
| `example31.yaml` | Piecewise-quadratic generator, `k_values=[1,2,3,4]` |
| `example41.yaml` | Entropy generator, `A_values=[1.406, 1.2, 0.9]` |

## Key Mathematical Conventions

- **Base distribution**: Truncated Exp(1) on [0, M=100]; normalising constant `1 - exp(-M)`.
- **Piecewise generator**: quadratic below `q_alpha = F0_inv(alpha)`, quadratic with curvature `k` above.  Setting `k=1` recovers the globally quadratic `phi(x) = x^2`.
- **BW divergence**: two equivalent forms — quantile form (1-D `quad` over `t ∈ (0,1)`) and survival form (two 1-D `quad` + one 2-D `dblquad`).
- **Premium**: integration-by-parts identity `E_Q[I(X)] = ∫ S_Q(x) I'(x) dx` exploited in both the trapezoidal and `quad` versions.
- **TVaR generalised inverse**: `g^{-1}(u) = (1-alpha)*u` for all `u ∈ [0,1]`.

## Dependencies

numpy, scipy, standard library only (no pandas or matplotlib imported in `src/`).

## Running Smoke Tests

```bash
python -m py_compile src/__init__.py src/distributions.py src/bregman.py src/distortion.py src/validation.py
```

## Experiment Scripts (`exp/`)

| File | Purpose |
|---|---|
| `exp1_example31.py` | Reproduces Example 3.1 (Problem 2.1): worst/best-case VaR and optimal indemnity for k=1..4. Paper targets for k=1: d1*≈0.406, d2*≈0.511, V_lower≈0.563, V_upper≈6.596. |
| `exp2_example41.py` | Reproduces Example 4.1 (Problem 2.2): worst-case distortion risk with guaranteed VaR constraint. Baseline A=1.406 ↔ λ*≈0. |
| `exp3_tvar_section43.py` | Section 4.3 TVaR three-case analysis: case classification, G*(x;β) formulas, regime boundaries. |

All experiment scripts use `matplotlib.use('Agg')` before other matplotlib imports.
Each wraps all `src.*` imports in `try/except`, printing a warning and returning
`None` from `main()` when a required module is absent.

## Test Suite (`tests/`)

| File | Covers |
|---|---|
| `test_distributions.py` | F0/S0/F0_inv boundary, monotonicity, complementarity, Config defaults |
| `test_bregman.py` | phi continuity, B_phi non-negativity/zero-diagonal, BW identity, entropy round-trips |
| `test_problem21.py` | VaR ordering, Example 3.1 paper values, admissibility, H sign structure |
| `test_tvar_section43.py` | Case boundaries, hat_x monotonicity, G* bounds, H_tvar type check |

### Import-Safety Pattern
Every test module wraps all `src.*` imports in `try/except` and sets a
`_HAS_*` flag. `pytestmark = pytest.mark.skipif(not _HAS_ALL, ...)` at module
level skips all tests when a core dependency is missing. Functions not in the
formal interface (e.g., `phi_second`, `compute_H_tvar`, `compute_hatS_Q`,
`bw_divergence_quantile_entropy`) get separate `_HAS_*` flags with per-test
`@pytest.mark.skipif` decorators.

### Numerical Precision Note — F0_inv Round-Trip
`F0_inv(F0(x))` incurs catastrophic cancellation for large x because
`1 - F0(x) = exp(-x)` is tiny while stored `t ≈ 1.0`.  The absolute error
grows as `∼ machine_epsilon / exp(-x)`.  Empirically:
- x=20: error ≈ 2e-8  (< 1e-6 ✓)
- x=25: error ≈ 4e-6  (> 1e-6 ✗)
`test_f0_inv_round_trip` therefore uses `xs = np.linspace(0.01, 20, 50)`.

### Running Tests
```bash
pytest tests/ -v
```
Expected while src modules are partial:
- test_distributions: all PASS
- test_bregman: all PASS (bw_divergence_quantile_entropy may SKIP)
- test_problem21: all SKIP (until src.problem21 exists)
- test_tvar_section43: all SKIP (until src.tvar_section43 exists)
