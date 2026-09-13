from src.matrix_ops import the_frobenius_norm
from src.matrix_ops import the_spectral_norm
from src.matrix_ops import the_infinity_norm
from src.matrix_ops import the_one_norm
import numpy as np
from numpy import linalg as LA
import math 
import pytest


## Frobenius norm tests 
def test_frobenius_norm_basic():
    result = the_frobenius_norm([[1, 2, 3], [4, 5, 6]])
    assert result == math.sqrt(91)

def test_frobenius_norm_zeros():
    result = the_frobenius_norm([[0, 0, 0], [0, 0, 0]])
    assert result == 0

def test_frobenius_norm_negative_numbers():
    result = the_frobenius_norm([[-1, -2, -3], [-1, -2, -3]])
    assert result == math.sqrt(28)

def test_frobenius_norm_single_element():
    result = the_frobenius_norm([[5], [2]])
    assert result == math.sqrt(29)

def test_frobenius_norm_matches_numpy():
    matrix = [[8.3, 21 , 1], 
              [2.8, 13, 4]]
    assert math.isclose(the_frobenius_norm(matrix), np.linalg.norm(matrix))

def test_frobenius_raises_on_heterogeneous():
    with pytest.raises(ValueError):
        the_frobenius_norm([[1, 2], [3]])


## Spectral norm tests 
def test_spectral_norm_basic():
    result = the_spectral_norm([[1, 2, 3], [4, 5, 6]])
    assert math.isclose(result, math.sqrt((182 + (2 * math.sqrt(8065))) / 4))

def test_spectral_norm_zeros():
    result = the_spectral_norm([[0, 0, 0], [0, 0, 0]])
    assert result == 0

def test_spectral_norm_negative_numbers():
    result = the_spectral_norm([[-1, -2, -3], [-1, -2, -3]])
    assert math.isclose(result, math.sqrt(28))

def test_spectral_norm_single_element():
    result = the_spectral_norm([[5], [2]])
    assert result == math.sqrt(29)

def test_spectral_norm_matches_numpy():
    matrix = [[20, 21 , 1], 
              [3, 13, 4]]
    assert math.isclose(the_spectral_norm(matrix), np.linalg.norm(matrix, ord=2))

def test_spectral_raises_on_heterogeneous():
    with pytest.raises(ValueError):
        the_spectral_norm([[1, 2], [3]])


## Infinity norm tests 
def test_infinity_norm_basic():
    result = the_infinity_norm([[1, 2, 3], [4, 5, 6]])
    assert result == 15 

def test_infinity_norm_zeros():
    result = the_infinity_norm([[0, 0, 0], [0, 0, 0]])
    assert result == 0

def test_infinity_norm_negative_numbers():
    result = the_infinity_norm([[-1, -2, -3], [-1, -2, -3]])
    assert result == 6 

def test_infinity_norm_single_element():
    result = the_infinity_norm([[5], [2]])
    assert result == 5

def test_infinity_norm_matches_numpy():
    matrix = [[20, 21 , 1], 
              [3, 13, 4]]
    assert math.isclose(the_infinity_norm(matrix), np.linalg.norm(matrix, ord=np.inf))

def test_infinity_raises_on_heterogeneous():
    with pytest.raises(ValueError):
        the_infinity_norm([[1, 2], [3]])


## One norm tests 
def test_one_norm_basic():
    result = the_one_norm([[1, 2, 3], [4, 5, 6]])
    assert result == 9

def test_one_norm_zeros():
    result = the_one_norm([[0, 0, 0], [0, 0, 0]])
    assert result == 0

def test_one_norm_negative_numbers():
    result = the_one_norm([[-1, -2, -3], [-1, -2, -3]])
    assert result == 6 

def test_one_norm_single_element():
    result = the_one_norm([[5], [2]])
    assert result == 7

def test_one_norm_matches_numpy():
    matrix = [[20, 21 , 1], 
              [3, 13, 4]]
    assert math.isclose(the_one_norm(matrix), np.linalg.norm(matrix, ord=1))

def test_one_raises_on_heterogeneous():
    with pytest.raises(ValueError):
        the_one_norm([[1, 2], [3]])