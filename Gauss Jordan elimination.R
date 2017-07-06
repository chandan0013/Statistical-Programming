myGaussJordan <- function(A, m)
{n <- dim(A)[1]
 B <- cbind(A, diag(rep(1, n))
 for (k in 1:m)
  {a <- B[k, k]
   for (j in 1:(n*2))
      B[k, j] <- B[k, j]/a
   for (i in 1:n)
      if (i != k)
      {a <- B[i, k]
       for (j in 1:(n*2))
          B[i, j] <- B[i, j] - B[k, j]*a;
      }
   }
return(B)
}



myGaussJordanVec <- function(A, m)
{B <- cbind(A, diag(rep(1, n)))
 for (k in 1:m)
   {B[k, ] <- B[k, ]/B[k, k]
    for (i in 1:n)
       if (i != k)
         B[i, ] <- B[i, ] - B[k, ]*B[i, k];
   }
 return(B)
}

