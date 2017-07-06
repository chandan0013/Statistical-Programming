myRidge <- function(X, Y, lambda)
{n = dim(X)[1]
 p = dim(X)[2]
 Z = cbind(rep(1, n), X, Y)
 A = t(Z) %*% Z
 D = diag(rep(lambda, p+2))
 D[1, 1] = 0
 D[p+2, p+2] = 0
 A = A + D
 S = mySweep(A, p+1)
 beta = S[1:(p+1), p+2]
 return(beta)
}


//////Spline regression
n = 20
p = 500
sigma = .1
lambda = 1.
x = runif(n)
x = sort(x)
Y = x^2 + rnorm(n)*sigma
X = matrix(x, nrow=n)
for (k in (1:(p-1))/p)
   X = cbind(X, (x>k)*(x-k))

beta = myRidge(X, Y, lambda)
Yhat = cbind(rep(1, n), X)%*%beta
plot(x, Y, ylim = c(-.2, 1.2), col = "red")
par(new = TRUE)
plot(x, Yhat, ylim = c(-.2, 1.2), type = ’l’, col = "green")