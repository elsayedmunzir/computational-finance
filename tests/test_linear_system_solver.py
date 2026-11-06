import pytest
import numpy as np 
from scipy.linalg import solve 
from src.lu_decomposition import lu_decomposition
from src.linear_system_solver import forward_substitution
from src.linear_system_solver import linear_system_solver
from src.linear_system_solver import backwards_substitution


def test_linear_system_solver_basic():
    result = linear_system_solver([[3, 1],[1, 2]], [5, 5])
    np.testing.assert_allclose(result, np.array([1, 2]))

def test_linear_system_solver_negatives(): 
    result = linear_system_solver([[-2,  1, -3],[ 4, -1,  2],[ 1, -3,  5]], [-9, 8, 10])
    np.testing.assert_allclose(result, np.array([1, 2, 3]))

def test_linear_system_solver_decimals():
    result = linear_system_solver([[2.0,  1.0, -0.5,  1.0],[1.0,  3.0,  2.0, -1.0],[0.5, -1.0,  4.0,  2.0],[1.0,  2.0,  1.0,  3.0]], [3.75, -6.5, 10.75, 7.0])
    np.testing.assert_allclose(result, np.array([1.5, -2.0, 0.5, 3.0]))

def test_linear_system_solver_identitiy_matrix():
    result = linear_system_solver([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3])
    np.testing.assert_allclose(result, np.array([1, 2, 3]))

def test_linear_system_solver_dimension(): 
    result = linear_system_solver([[1, 2, 3],[2, 5, 3],[1, 0, 8]], [6, 10, 9])
    assert len(result) == 3 

def test_linear_system_solver_AX_equals_B(): 
    A = [[1, 2, 3],[2, 5, 3],[1, 0, 8]]
    B = [6, 10, 9]
    X = linear_system_solver(A, B)
    np.testing.assert_allclose(A @ X, np.array(B))

def test_forward_substitution_Ly_equals_B():
    Y = forward_substitution([[1, 0, 0],[2, 1, 0],[-1, 3, 1]], [1, 4, 8])
    np.testing.assert_allclose(Y, np.array([1, 2, 3]))

def test_backwards_substitution_Ux_equals_Y(): 
    X = backwards_substitution([[2, 1, 3],[0, 3, 6],[0, 0, 4]], [11, 15, 8])
    np.testing.assert_allclose(X, np.array([2, 1, 2]))
    

def test_linear_system_solver_pivoting_required(): 
    with pytest.raises(ZeroDivisionError):
        linear_system_solver([[0, 2, 3],[1, 1, 1],[2, 3, 1]], [7, 4, 9])

def test_linear_system_solver_zeros():
    with pytest.raises(ZeroDivisionError): 
        linear_system_solver([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]], [1, 1, 1, 1])

def test_linear_system_solver_non_square(): 
    with pytest.raises(ValueError):
        linear_system_solver([[1, 2, 1],[2, 1, 3]], [6, 10])

def test_linear_system_solver_Heterogeneous():
    with pytest.raises(ValueError):
        linear_system_solver([[1, 2, 3],[4, 5],[6, 7, 8, 9]], [1, 2, 3])

def test_linear_system_solver_single_element(): 
    with pytest.raises(TypeError):
        linear_system_solver([5], [10])

def test_forward_substitution_incompatabile_A_B_dimensions(): 
    with pytest.raises(ValueError):
        forward_substitution([[2, 3],[4, 1],[5, 6]], [13])

def test_linear_system_solver_empty_matrix(): 
    with pytest.raises(ValueError): 
        linear_system_solver([[]], [])

def test_linear_system_solver_non_numeric():
    with pytest.raises(NameError): 
        linear_system_solver([[1, 2], [a, b]], [1, 2])

def test_linear_system_solver_matches_numpy(): 
    np.testing.assert_allclose(linear_system_solver([[5, 2, 1],[2, 6, 3],[1, 3, 7]], [12, 23, 28]), solve([[5, 2, 1],[2, 6, 3],[1, 3, 7]], [12, 23, 28]))