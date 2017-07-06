myAdaboost <- function(X_train, Y_train, X_test, Y_test,

num_iterations = 200)

{

n <- dim(X_train)[1]

p <- dim(X_train)[2]

threshold <- 0.8

X_train1 <- 2 * (X_train > threshold) - 1

Y_train <- 2 * Y_train - 1

X_test1 <- 2 * (X_test > threshold) - 1

Y_test <- 2 * Y_test - 1

beta <- matrix(rep(0,p), nrow = p)

w <- matrix(rep(1/n, n), nrow = n)

weak_results <- Y_train * X_train1 > 0

acc_train <- rep(0, num_iterations)

acc_test <- rep(0, num_iterations)

for(it in 1:num_iterations)

{

w <- w / sum(w)

weighted_weak_results <- w[,1] * weak_results

weighted_accuracy <- colSums(weighted_weak_results)

e <- 1 - weighted_accuracy

j <- which.min(e)


beta[j] <- beta[j] + dbeta
acc_train[it] <- mean(sign(X_train1 %*% beta) == Y_train)

acc_test[it] <- mean(sign(X_test1 %*% beta) == Y_test)

}

output <- list(beta = beta, acc_train = acc_train, acc_test = acc_test)

output

}