import math
import pytest
import numpy as np 
from scipy.linalg import lu 
from src.lu_decomposition import lu_decomposition


def test_lu_decomposition_basic():
    result = lu_decomposition([[2, 4, 2], [4, 10, 6], [2, 6, 5]])
    np.testing.assert_allclose(result[0], np.array([[1, 0, 0],[2, 1, 0], [1, 1, 1]]))
    np.testing.assert_allclose(result[1], np.array([[2, 4, 2], [0, 2, 2], [0, 0, 1]]))

def test_lu_decomposition_negatives():
    result = lu_decomposition([[-2, -2, -4], [-4, -6, -10], [-6, -10, -19]])
    np.testing.assert_allclose(result[0], np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]]))
    np.testing.assert_allclose(result[1], np.array([[-2, -2, -4], [0, -2, -2], [0, 0, -3]]))

def test_lu_decomposition_decimals():
    result = lu_decomposition([[2.5, 3, 1.5], [5, 8, 4], [2.5, 4, 3]])
    np.testing.assert_allclose(result[0], np.array([[1, 0, 0], [2, 1, 0], [1, 0.5, 1]]))
    np.testing.assert_allclose(result[1], np.array([[2.5, 3, 1.5], [0, 2, 1], [0, 0, 1]]))

def test_lu_decomposition_identitiy_matrix():
    result = lu_decomposition([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(result[0], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    np.testing.assert_allclose(result[1], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

def test_lu_decomposition_A_equals_LU(): 
    result = lu_decomposition([[3, 6, 3], [6, 15, 9], [3, 9, 10]])
    np.testing.assert_allclose(result[0] @ result[1], np.array([[3, 6, 3], [6, 15, 9], [3, 9, 10]]))

def test_lu_decomposition_diagonal_matrix():
    result = lu_decomposition([[2, 0, 0, 0], [0, -3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 4]])
    np.testing.assert_allclose(result[0], np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    np.testing.assert_allclose(result[1], np.array([[2, 0, 0, 0], [0, -3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 4]]))

def test_lu_decomposition_LU_dimesnions(): 
    result = lu_decomposition([[2, 3, 1], [4, 8, 6], [6, 9, 10]])
    l_shape = result[0].shape
    u_shape = result[1].shape 
    assert l_shape == (3, 3)
    assert u_shape == (3, 3)

def test_lu_decomposition_L_diagonal_ones(): 
    result = lu_decomposition([[3, 2, 1], [6, 5, 4], [9, 8, 10]])
    n = len(result[0])
    for i in range(n): 
        assert result[0][i, i] == 1 

def test_lu_decomposition_U_is_upper_triangular():
    U = lu_decomposition([[4,  2,  3], [8,  7,  5], [12, 10, 11]])[1]
    n = len(U)
    for i in range(1, n):
        for j in range(0, i-1):
            assert U[i, j] == 0 

def test_lu_decomposition_L_is_lower_triangular(): 
    L = lu_decomposition([[3, 2, 1], [6, 7, 4], [9, 5, 8]])[0]
    n = len(L)
    for i in range(n-1): 
        for j in range(i+1, n):
            assert L[i, j] == 0

def test_lu_decomposition_pivoting_required(): 
    with pytest.raises(ZeroDivisionError):
        lu_decomposition([[0, 2, 1], [4, 6, 2], [2, 3, 5]])

def test_lu_decomposition_zeros():
    with pytest.raises(ZeroDivisionError):
        lu_decomposition([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    
def test_lu_decomposition_non_square(): 
    with pytest.raises(ValueError): 
        lu_decomposition([[2, 3, 1], [4, 7, 5]])

def test_lu_decomposition_matches_scipy():
    matrix = np.array([[23, 3, 1], [4, 8, 5], [6, 7, 9]])
    p, l, u = lu(matrix)
    lu_result = lu_decomposition(matrix)
    np.testing.assert_allclose((p @ l) - lu_result[0], np.zeros((3, 3)))
    np.testing.assert_allclose(u - lu_result[1], np.zeros((3, 3)))







