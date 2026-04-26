/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#ifndef COVARIANCE_H
#define COVARIANCE_H

#include <cublas_v2.h>

// Compute covariance of the matrix data and place in cov
//   data_d:    col major matrix (obs x dim)
//   cov_d:     symmetric covariance returned here (dim x dim)
//   obs:       number of observations
//   dim:       number of variables
//   onesV_d:   vector (obs) with every element equal to 1.0f
//   ws_d:      vector (dim) of workspace on GPU
//   scaleCov:  scaling factor for the covariance
//
// Returns 0 on success
int cov_gpu(cublasHandle_t *cublas, float *data_d, float *cov_d,
            size_t obs, size_t dim,
            float *onesV_d, float *ws_d, float scaleCov);

// Compute mean (ws_d) and covariance (cov_d) of column-major data (obs x dim).
// Mean is always unscaled (1/obs); covariance is scaled by scaleCov.
// cov_d is filled in column-major order (dim x dim) with the upper triangle valid.
int cov_gpu_mean_cov(cublasHandle_t *cublas, const float *data_d, float *cov_d,
                     size_t obs, size_t dim,
                     const float *onesV_d, float *ws_d, float scaleCov);

#endif
