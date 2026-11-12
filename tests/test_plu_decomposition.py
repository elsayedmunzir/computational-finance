import pytest
import numpy as np 
from scipy.linalg import lu 
from scipy.linalg import solve
from src.plu_decomposition import plu_solve
from src.plu_decomposition import plu_decomposition
from src.linear_system_solver import forward_substitution
from src.linear_system_solver import backwards_substitution


def test_plu_decomposition_basic(): 
    P, L, U = plu_decomposition([[2, 1, 1],[4, -6, 0],[-2, 7, 2]])
    np.testing.assert_allclose(P, np.array([[0, 1, 0],[1, 0, 0],[0, 0, 1]]))
    np.testing.assert_allclose(L, np.array([[1, 0, 0], [1/2, 1, 0], [-1/2, 1, 1]]))
    np.testing.assert_allclose(U, np.array([[4, -6, 0], [0, 4, 1], [0, 0, 1]]))

def test_plu_decomposition_negatives(): 
    P, L, U = plu_decomposition([[-2, -1, -3], [-4, -6, -8], [-1, -3, -5]])
    np.testing.assert_allclose(P - np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), np.zeros((3, 3)))
    np.testing.assert_allclose(L - np.array([[1, 0, 0], [1/2, 1, 0], [1/4, -3/4, 1]]), np.zeros((3 ,3)))
    np.testing.assert_allclose(U - np.array([[-4, -6, -8], [0, 2, 1], [0, 0, -9/4]]), np.zeros((3, 3)))

def test_plu_decomposition_decimals(): 
    P, L, U = plu_decomposition([[0.5, 1.2, 0.3], [1.5, 0.4, 2.0], [0.2, 0.8, 1.1]])
    np.testing.assert_allclose(P, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    np.testing.assert_allclose(L, np.array([[1, 0, 0], [1/3, 1, 0], [2/15, 0.7, 1]]))
    np.testing.assert_allclose(U, np.array([[1.5, 0.4, 2.0], [0, 16/15, -11/30], [0, 0, 1.09]]))

def test_plu_decomposition_identity(): 
    P, L, U = plu_decomposition([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(P, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    np.testing.assert_allclose(L, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    np.testing.assert_allclose(U, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

def test_plu_decomposition_PA_equals_LU(): 
    A = np.array([[3, -1, 2], [6, -4, 0], [-3, 7, 8]])
    P, L, U = plu_decomposition(A)
    np.testing.assert_allclose(P @ A, L @ U)

def test_plu_decomposition_pivoting_required(): 
    with pytest.raises(ZeroDivisionError):
        plu_decomposition([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

def test_plu_decomposition_zeros(): 
    with pytest.raises(ZeroDivisionError):
        plu_decomposition([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

def test_plu_decomposition_non_square():
    with pytest.raises(ValueError): 
        plu_decomposition([[1, 2, 3], [4, 5, 6]])

def test_plu_decomposition_heterogeneous():
    with pytest.raises(ValueError):
        plu_decomposition([[1, 2, 3], [4, 5], [6, 7, 8, 9]])

def test_plu_decomposition_single_element(): 
    p, l, u = plu_decomposition([[1]])
    np.testing.assert_allclose(p,  np.array([[1]]))
    np.testing.assert_allclose(l,  np.array([[1]]))
    np.testing.assert_allclose(u,  np.array([[1]]))

def test_plu_decomposition_single_element_zero(): 
    with pytest.raises(ValueError): 
        plu_decomposition([[0]])


def test_plu_decomposition_non_numeric():
    with pytest.raises(TypeError): 
        plu_decomposition([[1, 2], ["a", "b"]])

def test_plu_solve_Ax_equals_b(): 
    A = [[2, 1, -1], [3, 2, 1], [1, -1, 2]]
    x = plu_solve(A, [1, 10, 5])
    np.testing.assert_allclose(A @ x, np.array([1, 10, 5]))

def test_plu_solve_incompatible_A_and_b(): 
    with pytest.raises(ValueError):
        plu_solve([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2])

def test_plu_solve_matches_scipy():
    np.testing.assert_allclose(plu_solve([[4, -2, 1], [1, 5, -1], [2, 1, 3]], [3, 8, 13]), solve([[4, -2, 1], [1, 5, -1], [2, 1, 3]], [3, 8, 13]))

def test_plu_decomposition_matches_scipy(): 
    A = [[3, 1, 2], [6, 3, 4], [3, 1, 5]]
    P, L, U = lu(A)
    plu_result = plu_decomposition(A)
    np.testing.assert_allclose(P - plu_result[0], np.zeros((3, 3)))
    np.testing.assert_allclose(L - plu_result[1], np.zeros((3, 3)))
    np.testing.assert_allclose(U - plu_result[2], np.zeros((3, 3)))
