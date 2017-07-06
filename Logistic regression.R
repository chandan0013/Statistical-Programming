myLogistic <- function(X_train, Y_train, X_test, Y_test,
num_iterations = 500, learning_rate = 1e-1)
{ n <- dim(X_train)[1]

p <- dim(X_train)[2]+1

ntest <- dim(X_test)[1]

X_train1 <- cbind(rep(1, n), X_train)

X_test1 <- cbind(rep(1, ntest), X_test)

sigma <- .1

beta <- matrix(rnorm(p)*sigma, nrow=p)

acc_train <- rep(0, num_iterations)

acc_test <- rep(0, num_iterations)

for(it in 1:num_iterations)

{

pr <- 1/(1 + exp(-X_train1 %*% beta))

dbeta <- matrix(rep(1, n), nrow = 1) %*%((matrix(Y_train - pr, n, p)*X_train1))/n

beta <- beta + learning_rate * t(dbeta)

prtest <- 1/(1 + exp(-X_test1 %*% beta))

acc_train[it] <- accuracy(pr, Y_train)

acc_test[it] <- accuracy(prtest, Y_test)
print(c(it, acc_train[it], acc_test[it]))

}

output <- list(beta = beta, acc_train = acc_train, acc_test = acc_test)

output

}