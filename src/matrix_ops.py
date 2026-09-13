import math
import numpy as np
from numpy import linalg as LA


def the_frobenius_norm(matrix): 
    m = len(matrix) 
    n = len(matrix[0]) 
    total = 0 
    for i in range(m):
        if len(matrix[i]) != n:
            raise ValueError("The matrix's row's are of different sizes")
        else: 
            for j in range(n): 
                total += (matrix[i][j]) ** 2 
    return math.sqrt(total)
        

def the_spectral_norm(matrix): 
    matrix = np.array(matrix)
    m = len(matrix)
    n = len(matrix[0])
    matrix_transpose = matrix.T 
    for i in range(m): 
        if len(matrix[i]) != n:
            raise ValueError("The matrix's row's are of different sizes")
        else: 
            if m >= n:
                B = matrix @ matrix_transpose
            else: 
                B = matrix_transpose @ matrix 

    eigenvalues = np.linalg.eigvalsh(B)
    highest_eigenval = max(eigenvalues)
    return math.sqrt(highest_eigenval)


def the_infinity_norm(matrix): 
    m = len(matrix) 
    n = len(matrix[0]) 
    array_of_totals = []
    for i in range(m):
        if len(matrix[i]) != n: 
            raise ValueError("The matrix's row's are of different sizes")
        else:
            total = 0 
            for j in range(n):
                total += abs(matrix[i][j])
            array_of_totals.append(total)
    return max(array_of_totals)
                
  
def the_one_norm(matrix): 
    m = len(matrix) 
    n = len(matrix[0]) 
    array_of_totals = [0 * i for i in range(n)]
    for i in range(m): 
        if len(matrix[i]) != n: 
            raise ValueError("The matrix's row's are of different sizes")
        else: 
            for j in range(n): 
                array_of_totals[j] += abs(matrix[i][j])
    return max(array_of_totals) 




        
    




    