my_SVM <- function(X_train, Y_train, X_test, Y_test, lambda = 0.01,

num_iterations = 1000, learning_rate = 0.1)

{

n <- dim(X_train)[1]

p <- dim(X_train)[2] + 1

X_train1 <- cbind(rep(1, n), X_train)

Y_train <- 2 * Y_train - 1

beta <- matrix(rep(0, p), nrow = p)

ntest <- nrow(X_test)

X_test1 <- cbind(rep(1, ntest), X_test)

Y_test <- 2 * Y_test - 1

acc_train <- rep(0, num_iterations)

acc_test <- rep(0, num_iterations)

for(it in 1:num_iterations)

{

s <- X_train1 %*% beta

db <- s * Y_train < 1


beta <- beta + learning_rate * t(dbeta)

beta[2:p] <- beta[2:p] - lambda * beta[2:p]

acc_train[it] <- mean(sign(s * Y_train))

acc_test[it] <- mean(sign(X_test1 %*% beta * Y_test))

}

model <- list(beta = beta, acc_train = acc_train, acc_test = acc_test)

model

}