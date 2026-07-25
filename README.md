# Computational Finance

A library of numerical methods and computational tools for quantitative finance, implemented from scratch in Python for pedagogical depth. The project builds foundational infrastructure that supports downstream applied research in equity markets, statistical arbitrage, and volatility modeling.

## Project Scope

The library is developed in three phases, each targeting a distinct layer of the computational finance stack.

### Phase 1: Numerical Linear Algebra Core

From-scratch Python implementations of numerical linear algebra algorithms, with verification against NumPy on random matrices.

- Matrix class with fundamental operations (add, multiply, transpose, determinant)
- LU decomposition (with and without partial pivoting)
- QR decomposition (Classical Gram-Schmidt, Modified Gram-Schmidt, Householder)
- Cholesky decomposition for symmetric positive definite matrices
- Eigenvalue methods (power iteration, inverse iteration, QR algorithm)
- Basic Singular Value Decomposition

### Phase 2: Statistical Foundations for Equity Research

Statistical infrastructure for equity research applications.

- Sample and shrinkage covariance estimation
- Principal Component Analysis on equity returns
- Numerical optimization (gradient descent, Newton's method, conjugate gradient)
- Minimum variance portfolio construction
- Black-Scholes options pricing with numerical Greeks

### Phase 3: Monte Carlo Methods for Equity Research

Simulation-based methods for empirical finance research.

- Monte Carlo foundations with convergence analysis
- Bootstrap methods for equity return statistics and Sharpe ratio inference
- Multi-asset correlated return simulation via Cholesky decomposition
- Options pricing via Monte Carlo with variance reduction (antithetic, control variates)

## Repository Structure

```
computational-finance/
├── src/           # Library source code
├── tests/         # Unit tests (pytest, target 80%+ coverage)
├── notebooks/     # Demonstration and analysis notebooks
├── docs/          # Mathematical background and API documentation
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone git@github.com:elsayedmunzir/computational-finance.git
cd computational-finance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
pytest
```

## Author

Munzir Elsayed. Built as part of a structured gap year preparing for EUR IBEOR (Econometrics and Operations Research), 2026-2027.

## License

MIT License. See LICENSE file for details.