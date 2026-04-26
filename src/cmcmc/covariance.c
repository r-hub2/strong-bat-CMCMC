/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#include "check.h"
#include "covariance.h"

// Compute covariance of the matrix data and place in cov
//   data_d:    col major matrix (obs x dim)
//   cov_d:     symmetric covariance returned here (dim x dim)
//   obs:       number of observations
//   dim:       number of variables
//   onesV_d:   vector (obs) with every element equal to 1.0f
//   ws_d:      vector (dim) of workspace on GPU
//   scaleCov:  scaling factor for the covariance
int cov_gpu(cublasHandle_t *cublas, float *data_d, float *cov_d,
            size_t obs, size_t dim,
            float *onesV_d, float *ws_d, float scaleCov) {
  // Match CUDA/src/common/covariance.cu (population covariance):
  // cov = (scaleCov/obs) * X^T X - scaleCov * mu mu^T, where mu=(1/obs)X^T 1.
  return cov_gpu_mean_cov(cublas, data_d, cov_d, obs, dim, onesV_d, ws_d, scaleCov);
}

int cov_gpu_mean_cov(cublasHandle_t *cublas, const float *data_d, float *cov_d,
                     size_t obs, size_t dim,
                     const float *onesV_d, float *ws_d, float scaleCov) {
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;

  // cov_d <- (scaleCov/obs) * data_d^T %*% data_d
  const float alpha_cov = (obs > 0) ? (scaleCov / (float)obs) : 0.0f;
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS ==
                cublasSsyrk(*cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                           (int)dim, (int)obs, &alpha_cov,
                           data_d, (int)obs, &beta0, cov_d, (int)dim) );

  // ws_d <- (1/obs) * data_d^T %*% onesV_d  (mean)
  const float alpha_mu = (obs > 0) ? (1.0f / (float)obs) : 0.0f;
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS ==
                cublasSgemv(*cublas, CUBLAS_OP_T,
                           (int)obs, (int)dim, &alpha_mu,
                           data_d, (int)obs,
                           onesV_d, 1,
                           &beta0,
                           ws_d, 1) );

  // cov_d <- cov_d - scaleCov * ws_d %*% ws_d^T
  const float alpha2 = -scaleCov;
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS ==
                cublasSsyrk(*cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           (int)dim, 1, &alpha2,
                           ws_d, (int)dim,
                           &beta1,
                           cov_d, (int)dim) );

  return 0;
}
