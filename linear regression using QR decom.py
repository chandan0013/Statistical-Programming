/////use def qr(Z): function

n = 100

p = 5

X = np.random.random_sample((n, p))

beta = np.array(range(1, p+1))

Y = np.dot(X, beta) + np.random.standard_normal(n)

Z = np.hstack((np.ones(n).reshape((n, 1)), X, Y.reshape((n, 1))))

_, R = qr(Z)

R1 = R[:p+1, :p+1]

Y1 = R[:p+1, p+1]

beta = np.linalg.solve(R1, Y1)

print beta