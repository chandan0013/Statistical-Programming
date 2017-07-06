mySweep <- function(A, m)
{ n <- dim(A)[1]
  for (k in 1:m)
  { for (i in 1:n)
      for (j in 1:n)
         if (i!=k & j!=k)
            A[i,j] <- A[i,j] - A[i,k]*A[k,j]/A[k,k]
        
    for (i in 1:n)
       if (i!=k)
         A[i,k] <- A[i,k]/A[k,k]

    for (j in 1:n)
      if (j!=k)
         A[k,j] <- A[k,j]/A[k,k]

    A[k,k] <- - 1/A[k,k]
  }

return(A)
}