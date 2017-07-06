///// Using sweep operator
n = 100
p = 5
X = matrix(rnorm(n*p), nrow=n)
beta = matrix(1:p, nrow = p)
Y = X %*% beta + rnorm(n)
lm(Y~X)
Z = cbind(rep(1, n), X, Y)
A = t(Z) %*% Z
S = mySweep(A, p+1)
beta = S[1:(p+1), p+2]


///// using in-built operator
n = 100
X1 = rnorm(n)
X2 = rnorm(n)
Y = X1 + 2*X2 + rnorm(n)
lm(Y~X1+X2)
A = data.frame(x1 = X1, x2 = X2, y = Y)
lm(y ~ x1 + x2, data = A)
data(trees)
lt = log(trees)
m <- lm(Volume~Height+Girth, data=lt)
summary(m)