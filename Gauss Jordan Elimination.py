def myGaussJordan(A, m):
    n = A.shape[0]
    B = np.hstack((A, np.identity(n)))
    for k in range(m):
        a = B[k, k]
        for j in range(n*2):
            B[k, j] = B[k, j] / a
        for i in range(n):
            if i != k:
               a = B[i, k]
               for j in range(n*2):
                   B[i, j] = B[i, j] - B[k, j]*a;

     return B

def myGaussJordanVec(A, m):
    n = A.shape[0]
    B = np.hstack((A, np.identity(n)))
    for k in range(m):
        B[k, :] = B[k, ] / B[k, k]
        for i in range(n):
            if i != k:
               B[i, ] = B[i, ] - B[k, ]*B[i, k];
    return B