/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#ifndef GENERATE_MVN_H
#define GENERATE_MVN_H

#include <stddef.h>
#include <curand.h>
#include <cublas_v2.h>

// Generate multivariate normal samples, strided (ie res_d column major)
//   res_d: col major matrix on GPU (n x dim): stores the sampled values one per row
//   n:     number of samples to generate
//   U_d:   col major upper triangular matrix on GPU (dim x dim): Cholesky decomposition of covariance matrix
//   dim:   dimension of the MVN
int generateMVN_gpu(curandGenerator_t *curng, cublasHandle_t *cublas, float *res_d, size_t n, const float *U_d, size_t dim);

#endif
