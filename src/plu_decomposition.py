import numpy as np 
from src.linear_system_solver import forward_substitution
from src.linear_system_solver import backwards_substitution

def plu_decomposition(A): 
    """Factorize a matrix A into LU such that PA = LU where P is a permutation matrix"""                
    m = len(A)
    n = len(A[0])

    if m != n: 
        raise ValueError("Dimensions mismatch")            ##Verify non-square matrix 

    for i in range(m):                                     ##Verify heterogeneous matrix
        if len(A[i]) != n: 
            raise ValueError("Heterogeneous matrix")
        for j in range(n):                                 ##Verify non-numeric entries 
            invalid = False 
            try: 
                entry = float(A[i][j])
            except ValueError: 
                invalid = True 
            if invalid: 
                raise TypeError("Only numeric entries are acceptable")
    
    U = np.array(A, dtype=float)                           ##Initialize U
    L = np.zeros((m, n), dtype=float)                      ##Initialize L
    P = np.zeros((m, n))                                   ##Initialize P 
    for i in range(m):                                     ##1's along the diagonal
        L[i, i] = 1 
        P[i, i] = 1 

    if m == 1 and n == 1:                                  ##Single element matrices 
        if A[i][i] != 0: 
            return (P, L, U )
        else: 
            raise ValueError("Singular matrix")

    for i in range(n): 
        pivot = U[i, i]                       
        row_of_highest_value = i                           ##Pivot row
        for j in range(i + 1, m): 
            if abs(U[j, i]) > abs(pivot):                  ##Find the highest abs value in the column
                pivot = U[j, i]
                row_of_highest_value = j 
        
        if i != row_of_highest_value:                      ##If pivot in a different row 
            P[[i, row_of_highest_value]] = P[[row_of_highest_value, i]]       ##Swap rows of P
            U[[i, row_of_highest_value]] = U[[row_of_highest_value, i]]       ##Swap rows of U
            if i > 0:                                                         ##Starting from the second pivot
                L[[i, row_of_highest_value], :i] = L[[row_of_highest_value, i], :i]       ##Swap the rows entries in the previous columns 

        for j in range(i+1, m):                                                ##Row operations 
            if pivot == 0:                                  
                raise ZeroDivisionError("Zero pivot cannot be avoided")                                
            else: 
                multiplier = U[j, i] / pivot 
                L[j, i] = multiplier 
                U[j, :] = U[j, : ] - (multiplier * U[i, :]) 
    return (P, L, U) 

def plu_solve(A, b):
    """Solve Ax = b using PA = LU, forward, and backward substitution"""
    if len(b) != len(A):
        raise ValueError("Incompatible dimensions")
    P, L, U = plu_decomposition(A) 
    y = forward_substitution(L, P @ b)
    x = backwards_substitution(U, y)
    return x
    


