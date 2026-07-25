"""Sanity check: verify core scientific Python stack imports and works."""

import numpy as np
import pandas as pd
import matplotlib
import scipy


def test_numpy_available():
    """NumPy should be importable and basic operations should work."""
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    assert np.__version__.startswith("2.")


def test_pandas_available():
    """Pandas should be importable and create a DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.shape == (3, 2)
    assert df["a"].sum() == 6


def test_matplotlib_available():
    """Matplotlib should import (no plot rendering needed for the sanity check)."""
    assert matplotlib.__version__ is not None


def test_scipy_available():
    """SciPy should import and expose expected submodules."""
    from scipy import stats, linalg
    assert stats is not None
    assert linalg is not None