import numpy as np 

def lu_decomposition(matrix): 
    m = len(matrix)
    n = len(matrix[0])
    if m != n:                                                    ##Square matrix validation
        raise ValueError("The matrix must be a square matrix")
    matrix = np.array(matrix)
    L = np.zeros((m, n))                                          ##Lower triangular initialization
    for i in range(m): 
        L[i, i] = 1   
    U = np.array(matrix, dtype=float)                             ##Upper triangular initialization 
    for i in range(n-1):                                      ##For each diagonal entry, eliminate entries below                   
        for j in range(i+1, m):  
            if U[i, i] == 0:                                  ##Zero in the pivot position
                raise ZeroDivisionError("Pivoting required")                                
            else: 
                multiplier = U[j,i] / U[i, i]                 ##Gaussian elimination
                L[j, i] = multiplier 
                U[j, :] = U[j, : ] - (multiplier * U[i, :])    
    return (L, U)






         
    


