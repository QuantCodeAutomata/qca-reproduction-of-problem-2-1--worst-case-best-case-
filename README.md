# Distributionally Robust Insurance — Bregman-Wasserstein Reproduction

Numerical reproduction of the theoretical results from:

> *Distributionally Robust Insurance under Bregman-Wasserstein Divergence*

The repository implements and validates:

- **Problem 2.1** — Worst-case and best-case VaR over a Bregman-Wasserstein (BW) ball,
  the alpha-maxmin VaR insurance objective, and the explicit optimal indemnity (Example 3.1).
- **Problem 2.2** — Worst-case convex distortion risk with a guaranteed VaR constraint,
  including KKT multiplier selection and monotonicity-repair for the worst-case survival
  function (Example 4.1).
- **Section 4.3** — TVaR-specialised three-case analysis of the worst-case survival
  candidate G\*(x;β) and the case-classification regime map.

---

## Repository Structure

```
src/               Core mathematical modules
configs/           YAML parameter files for each experiment
exp/               Experiment runner scripts
data/              (Reserved for any future data artefacts)
results/           Output tables, figures, and RESULTS.md
tests/             pytest test suite (34 tests)
```

### Source modules (`src/`)

| Module | Contents |
|--------|----------|
| `distributions.py` | Truncated Exp(1) CDF/survival/quantile; `Config` dataclass |
| `bregman.py` | Piecewise-quadratic generator; entropy generator; BW divergence (quantile & survival forms) |
| `distortion.py` | TVaR distortion, generalised inverse, premium computation |
| `problem21.py` | Worst-/best-case VaR (Props. 3.1–3.2), indemnity thresholds, optimal indemnity, Example 3.1 |
| `problem22.py` | Worst-case distortion risk (Lagrangian), KKT solver, Example 4.1 |
| `bw_survival.py` | BW constraint Ψ(S) in survival form |
| `repair.py` | Flat-segment monotonicity repair for the relaxed survival optimizer |
| `tvar_section43.py` | TVaR case classification, ĥat{x}, hatG, G\* formulas |
| `validation.py` | Monotonicity checks, admissibility, BW-feasibility, KKT residual |

---

## Installation

```bash
pip install numpy scipy matplotlib pandas pyyaml pytest
```

All required packages are standard and available in the project environment.

---

## Running Experiments

```bash
python exp/exp1_example31.py     # Problem 2.1 — piecewise quadratic generator
python exp/exp2_example41.py     # Problem 2.2 — entropy generator, TVaR
python exp/exp3_tvar_section43.py  # Section 4.3 — TVaR case analysis
```

Results (CSV tables + PNG figures) are saved to `results/`.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: **34 passed**.

---

## Key Results (Summary)

### Problem 2.1 — Example 3.1 (k=1, ε=0.5)

| Metric | Computed | Paper |
|--------|----------|-------|
| d₁\* | 0.4055 | ~0.406 |
| d₂\* | 0.5108 | ~0.511 |
| V̲_α | 0.5628 | ~0.563 |
| V̄_α (k=1) | 7.002 | ~6.596 |
| V̄_α (k=4) | 5.300 | ~4.894 |

V̄_α decreases monotonically with k, confirming the paper's qualitative claim.

### Problem 2.2 — Example 4.1

- A_min = 1.4055 (paper: ~1.406) ✓
- At A=1.406: λ\*=0 (nonbinding VaR constraint) ✓

### Section 4.3 — TVaR cases

- Case (i) / case (ii) boundary at λ≈12.33, where (1+λ)(1+θ) = 1/(1−α) = 20 ✓
- In case (i): G\*(x;β)=S_Q(x) ✓

See `results/RESULTS.md` for the full results and validation tables.

---

## Mathematical Conventions

- **BW divergence (quantile form):** BW[F₁‖F₂] = ∫₀¹ B_φ(F₁⁻¹(t), F₂⁻¹(t)) dt
- **BW divergence (survival form):** Lemma 2.1 two-integral representation
- **Bregman divergence:** B_φ(x,y) = φ(x) − φ(y) − φ'(y)(x−y)
- **Piecewise generator:** φ(x;k) = x² for x < q_α; = q_α² + 2q_α(x−q_α) + k(x−q_α)² for x ≥ q_α
- **Entropy generator:** φ(x) = (x+1)ln(x+1)
- **TVaR distortion:** g(t) = min(t/(1−α), 1); generalised inverse g⁻¹(u) = (1−α)u
