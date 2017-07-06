import numpy as np

def mySweep(B, m):
    A = np.copy(B)
    n, c = A.shape
    for k in range(m):
        for i in range(n):
            for j in range(n):
                if i != k and j != k:
                   A[i,j] = A[i,j]- A[i,k]*A[k,j]/A[k,k]
        for i in range(n):
            if i != k:
               A[i,k] = A[i,k]/A[k,k]

        for j in range(n):
            if j != k:
               A[k,j] = A[k,j]/A[k,k]

        A[k,k] = -1/A[k,k]

return A