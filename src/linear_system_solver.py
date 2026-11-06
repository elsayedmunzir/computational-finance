import numpy as np 
from src.lu_decomposition import lu_decomposition

def forward_substitution(L, B):
    B = np.array(B)                            ##Initialize the vector B
    L = np.array(L)
    Y = np.zeros(len(L), dtype=float)          ##Initialize the vector Y

    if len(L[0]) != len(B): 
        raise ValueError("dimensions of L and B are incompatible")    ##Mismatched dimensions verification

    for i in range(len(L)):                                      ##Heterogeneous matrix verification
        if len(L[i]) != len(L[0]): 
            raise ValueError("heterogeneous matrix")

    for i in range(len(Y)):                                           ##forward substitution
        Sum_of_constants = 0 
        j = i - 1 
        while j > -1: 
            Sum_of_constants += L[i, j] * Y[j]
            j -= 1 
        Y[i] = (B[i] - Sum_of_constants) / L[i, i]
    
    return Y 

    
def backwards_substitution(U, Y):
    U = np.array(U)
    m = len(U)    
    X = np.zeros(m, dtype=float)                                      ##Initialize the vector X

    if len(U[0]) != len(Y): 
        raise ValueError("dimensions of U and Y are incompatible")    ##Mismatched dimensions verification

    for i in range(len(U)):                                      ##Heterogeneous matrix verification
        if len(U[i]) != len(U[0]): 
            raise ValueError("heterogeneous matrix")

    X[m-1] = Y[m-1] / U[m-1, m-1]                                     ##Last element solved directly
    for i in range(m - 2, -1, -1):                                    ##Backwards substitution
        Sum_of_constants2 = 0
        for j in range(m-1, i, -1): 
            Sum_of_constants2 += X[j] * U[i, j]
        X[i] = (Y[i] - Sum_of_constants2) / U[i, i]

    return X 

def linear_system_solver(matrix, B): 
    L, U = lu_decomposition(matrix) 
    Y = forward_substitution(L, B)
    X = backwards_substitution(U, Y)
    return X 
