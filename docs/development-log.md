# Development Log

Chronological record of significant development milestones in the
computational-finance project.

## 2026-07-25 (Saturday) — Repository initialized

- Set up project structure: `src/`, `tests/`, `notebooks/`, `docs/`
- Wrote README with 3-phase project scope
- Configured MIT License, `.gitignore`, `requirements.txt`
- Created Python virtual environment (Python 3.14.4)
- Installed core scientific stack: NumPy 2.5, Pandas 3.0, Matplotlib 3.11,
  SciPy 1.18, pytest 9.1, Jupyter 1.1
- First sanity check tests passing (`tests/test_imports.py`, 4 tests)
- Repository initialized on GitHub as public portfolio artifact
- Migrated `day1.py` scratch work from personal workspace (excluded via `.gitignore`)

## Next Milestones

- **Phase 1 (Numerical Linear Algebra Core):** Matrix class launch — Week 3 of Phase A (~Aug 10, 2026)
- **Phase 2 (Statistical Foundations):** launch after 18.06 progress reaches SVD material
- **Phase 3 (Monte Carlo Methods):** launch in Phase C of gap year (Jan 2027)

## Notes

Development log is updated only for significant milestones (new phase launch,
major feature completion, structural changes). Day-to-day work is tracked in
git commit history.


## 2026-08-27 (Thursday) | Phase 1 Session 1 complete

- Implemented dot_product and euclidean_norm from scratch
- 9 tests passing (basic, edge cases, NumPy verification)
- First real Flagship implementation committed


## 2026-09-13 (Sunday) | Phase 1 Session 2 complete

- Implemented four matrix norms from scratch: Frobenius, spectral, infinity, one
- All functions include input validation (raises ValueError on heterogeneous matrices)
- Spectral norm optimized to compute smaller of A@A^T or A^T@A based on matrix dimensions
- 24 tests passing


## 2026-10-20 (Monday) | Phase 1 Session 3 complete

- Implemented LU decomposition using Doolittle method (unit diagonal on L)
- Function validates square input (raises ValueError on non-square matrices)
- Zero pivot detection raises ZeroDivisionError when pivoting would be required
- Returns tuple (L, U) with L lower triangular and U upper triangular such that A = LU
- No pivoting, no system solving (deferred to Session 4)
- 14 tests passing


## 2026-11-06 (Friday) | Phase 1 Session 4 complete

- Implemented linear system solver using LU decomposition from Session 3
- forward_substitution(L, B) solves LY = B for lower triangular L
- backwards_substitution(U, Y) solves UX = Y for upper triangular U
- linear_system_solver(A, B) orchestrates full solve, calls lu_decomposition once
- Takes L and U directly to enable reuse in Sessions 5-10 and avoid duplicate computation
- Input validation for dimensions and heterogeneous rows in both substitution functions
- No pivoting (deferred to Session 5)
- 17 tests passing 

## 2026-11-12 (Thursday) | Phase 1 Session 5 complete

- Implemented PLU decomposition with partial pivoting
- plu_decomposition(A) returns (P, L, U) such that PA = LU
- plu_solve(A, b) solves Ax = b by calling plu_decomposition then forward and backwards substitution from Session 4
- Partial pivoting selects largest absolute value in column as pivot
- Row swaps tracked in P and applied to L for columns already computed
- Input validation for non-square, heterogeneous, non-numeric entries, singular matrices, zero pivots, and incompatible A/b dimensions
- Single element matrix handling with explicit zero check
- Scipy benchmark tests included for both decomposition and solve
- 16 tests passing