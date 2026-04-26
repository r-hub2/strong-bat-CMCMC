/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#include "check.h"
#include "generate_mvn.h"

// Generate multivariate normal samples, strided (ie res_d column major)
//   res_d: col major matrix on GPU (n x dim): stores the sampled values one per row
//   n:     number of samples to generate
//   U_d:   col major lower triangular matrix on GPU (dim x dim): Cholesky decomposition of covariance matrix
//   dim:   dimension of the MVN
int generateMVN_gpu(curandGenerator_t *curng, cublasHandle_t *cublas, float *res_d, size_t n, const float *U_d, size_t dim) {
  // Fill standard Normal variates up
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandGenerateNormal(*curng, res_d, n*dim, 0.0, 1.0) );

  const float alpha=1.0;
  // Transform to multivariate Normal
  // cov(matrix(rnorm(100000),ncol=2)%*%chol(Sig))
  // For L lower-triangular with Sig = L %*% t(L), we need: res_d %*% t(L).
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasStrmm(*cublas,
                                                  CUBLAS_SIDE_RIGHT,
                                                  CUBLAS_FILL_MODE_LOWER,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_DIAG_NON_UNIT,
                                                  (int)n, (int)dim,
                                                  &alpha,
                                                  U_d, (int)dim,
                                                  res_d, (int)n,
                                                  res_d, (int)n) );

  return(0);
}
