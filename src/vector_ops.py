def dot_product(a, b):
    sum = 0 
    if len(a) == len(b):
        for i in range(len(a)): 
            sum += a[i] * b[i]
        return sum 
    else: 
        return "Invalid, please ensure that the dimensions match"


import math 
def euclidean_norm(v):
    return math.sqrt(dot_product(v, v))


