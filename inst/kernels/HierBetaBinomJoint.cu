/*
 Hierarchical Beta–Binomial (joint) posterior kernel for CMCMC.

 State space (DIM must equal 2 + J, where J is Xi[0]):
   theta[0]     = log(alpha)
   theta[1]     = log(beta)
   theta[2 + j] = z_j = logit(theta_j)   for j = 0..J-1
 with alpha>0, beta>0, and theta_j = sigmoid(z_j) in (0,1).

 Data input:
   - Xi[0] = J (number of experiments)
   - Xi[1 + j] = n_j  for j = 0..J-1
   - Xi[1 + J + j] = y_j for j = 0..J-1

 Target (up to an additive constant):
   log p(log alpha, log beta, z | y)
 using:
   y_j | theta_j ~ Binomial(n_j, theta_j)
   theta_j | alpha,beta ~ Beta(alpha, beta)
   p(alpha,beta) ∝ (alpha+beta)^(-5/2)

 Notes:
   - Binomial coefficients choose(n_j, y_j) are constant and omitted.
   - The state uses unconstrained variables (log alpha, log beta, z=logit(theta)).
     The corresponding Jacobians are included:
       d(alpha)/d(log alpha) = alpha, d(beta)/d(log beta) = beta,
       d(theta)/d(z) = theta(1-theta).
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
  (void)Xf;
  if (!Xi) return -INFINITY;

  const int J = Xi[0];
  if (J <= 0) return -INFINITY;

  // DIM must match the runtime J for this joint kernel.
  if (DIM != (2 + J)) return -INFINITY;

  const float log_alpha = theta[0];
  const float log_beta = theta[1];
  const float alpha = expf(log_alpha);
  const float beta = expf(log_beta);
  if (!isfinite(alpha) || !isfinite(beta) || alpha <= 0.0f || beta <= 0.0f) return -INFINITY;

  const float s = alpha + beta;
  if (!isfinite(s) || s <= 0.0f) return -INFINITY;

  // Hyperprior p(alpha,beta) ∝ (alpha+beta)^(-5/2).
  float lp = -2.5f * logf(s);
  // Jacobian for (log alpha, log beta) parameterisation: +log(alpha) + log(beta).
  lp += log_alpha + log_beta;

  const float logB_ab = lgammaf(alpha) + lgammaf(beta) - lgammaf(s);

  float ll = 0.0f;
  float lp_theta = 0.0f;

  for (int j = 0; j < J; j++) {
    const int nj = Xi[1 + j];
    const int yj = Xi[1 + J + j];
    if (nj < 0 || yj < 0 || yj > nj) return -INFINITY;

    const float z = theta[2 + j];
    const float ltheta = log_sigmoid(z);
    const float l1mtheta = log1m_sigmoid(z);

    ll += ((float)yj) * ltheta + ((float)(nj - yj)) * l1mtheta;
    lp_theta += (alpha - 1.0f) * ltheta + (beta - 1.0f) * l1mtheta;

    // Jacobian for z = logit(theta): +log(theta) + log(1-theta).
    lp_theta += ltheta + l1mtheta;
  }

  // Beta prior normalizing constants: -J * log B(alpha,beta).
  lp_theta -= (float)J * logB_ab;

  return lp + ll + lp_theta;
}
