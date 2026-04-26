/*
 Simple logistic-regression posterior kernel for CMCMC.
 
 Expects:
   - DIM is the parameter dimension (compile-time).
   - Xi[0] = n (number of observations)
   - Xf layout:
       X   : n*DIM floats, row-major (i from 0..n-1, j from 0..DIM-1)
       y   : n floats, values in {0,1}
       var : 1 float, prior variance for each coefficient (default: 100.0 if <= 0)
 
 logdens(theta) = sum_i log p(y_i | X_i, theta) + log p(theta)
 with y_i ~ Bernoulli(sigmoid(X_i theta)), and theta_j ~ N(0, var).
 */

#include <math.h>

static __device__ __forceinline__ float log_sigmoid(float x) {
  // log(sigmoid(x)) = -log(1 + exp(-x))
  return -log1pf(expf(-x));
}

static __device__ __forceinline__ float log1m_sigmoid(float x) {
  // log(1 - sigmoid(x)) = -log(1 + exp(x))
  return -log1pf(expf(x));
}

__device__ float logdens(const float *theta, const float *Xf, const int *Xi) {
  const int n = (Xi ? Xi[0] : 0);
  if (n <= 0) return -INFINITY;

  const float *X = Xf;
  const float *y = Xf + (size_t)n * (size_t)DIM;
  const float prior_var_in = Xf[(size_t)n * (size_t)DIM + (size_t)n];
  const float prior_var = (prior_var_in > 0.0f ? prior_var_in : 100.0f);
  const float inv_prior_var = 1.0f / prior_var;

  float ll = 0.0f;
  for (int i = 0; i < n; i++) {
    float eta = 0.0f;
    const float *Xi_row = X + (size_t)i * (size_t)DIM;
    for (int j = 0; j < DIM; j++) {
      eta += Xi_row[j] * theta[j];
    }
    const float yi = y[i];
    // yi assumed 0/1; treat anything > 0.5 as 1.
    ll += (yi > 0.5f) ? log_sigmoid(eta) : log1m_sigmoid(eta);
  }

  // Independent N(0, prior_var) prior on theta.
  float lp = 0.0f;
  for (int j = 0; j < DIM; j++) {
    const float tj = theta[j];
    lp += -0.5f * tj * tj * inv_prior_var;
  }

  return ll + lp;
}
