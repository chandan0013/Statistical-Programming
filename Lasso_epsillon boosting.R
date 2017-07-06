T = 3000

epsilon = .0001

beta = matrix(rep(0, p), nrow = p)

db = matrix(rep(0, p), nrow = p)

beta_all = matrix(rep(0, p*T), nrow = p)

R = Y

for (t in 1:T)

{

for (j in 1:p)

db[j] = sum(R*X[, j])
j = which.max(abs(db))

beta[j] = beta[j]+db[j]*epsilon

R = R - X[, j]*db[j]*epsilon

beta_all[, t] = beta

}

matplot(t(matrix(rep(1, p), nrow = 1)%*%abs(beta_all)), t(beta_all), type = ’l’)