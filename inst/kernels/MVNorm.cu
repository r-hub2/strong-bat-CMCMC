/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

// TODO: Improve numerical stability of this by using backsolve method as per
//       mvdnorm in mvtnorm package?

/*
 Example usage in R.  Mean in mu, cov in Sigma

 L <- t(chol(Sigma))
 Linv <- solve(L)
 X <- c(mu, t(Linv)%*%Linv)

 OR

 X <- c(mu, solve(Sigma))
 */

#include <math.h>

// Computes -0.5 * (theta-mu) %*% SigmaInv %*% (theta-mu)
__device__ float logdens(const float *theta, const float *Xf, const int *Xi) {
  // Setup parameters of multivariate normal
  const float *mu = Xf;
  const float *SigmaInv = Xf + DIM;

  float res = 0.0f, tmp = 0.0f;
  for(int j=0; j<DIM; j++) {
    tmp = 0.0f;
    for(int i=0; i<DIM; i++) {
      tmp += (theta[i]-mu[i]) * *(SigmaInv++);
    }
    res += tmp * (theta[j]-mu[j]);
  }

  return(res * (-0.5));
}
