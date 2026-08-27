from src.vector_ops import dot_product
from src.vector_ops import euclidean_norm 
import numpy as np
import math 

def test_dot_product_basic():
    result = dot_product([1, 2, 3], [4, 5, 6])
    assert result == 32

def test_dot_product_with_zeros():
    result = dot_product([0, 0, 0], [1, 2, 3])
    assert result == 0 

def test_dot_product_negative_numbers():
    result = dot_product([-1, -2, -3], [1, 2, 3])
    assert result == -14

def test_dot_product_single_element():
    result = dot_product([5], [2])
    assert result == 10 

def test_dot_product_matches_numpy():
    a = [4.5, 3.2 , 7.8]
    b = [8.36, 10.14, 2]
    assert math.isclose(dot_product(a, b), np.dot(a, b))

def test_dot_product_mismatched_lengths(): 
    result = dot_product([2, 3], [1, 2, 3])
    assert isinstance(result, str)

def test_euclidean_norm_basic():
    result = euclidean_norm([3, 4])
    assert result == 5

def test_euclidean_norm_with_zeros():
    result = euclidean_norm([0, 0, 0])
    assert result == 0.0

def test_euclidean_norm_matches_numpy():
    v = [1, 3 , 5]
    assert math.isclose(euclidean_norm(v), np.linalg.norm(v))