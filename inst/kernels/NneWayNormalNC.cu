/*
 One-way Normal random-effects model (full joint, non-centered) target kernel.

 Non-centered parameterization:
   z_j ~ Normal(0,1)
   theta_j = mu + tau * z_j
   ybar_j | mu,tau,z_j ~ Normal(mu + tau z_j, sigma^2 / n_j)

 Hyperprior:
   p(mu, log tau) ∝ 1  (flat in (mu, log tau))

 State space (DIM must equal 2 + J, where J is Xi[0]):
   theta[0]     = mu
   theta[1]     = log(tau)
   theta[2 + j] = z_j     for j = 0..J-1
 with tau = exp(log(tau)) > 0.

 Data input (recommended layout):
   - Xi[0]       = J
   - Xi[1 + j]   = n_j    for j = 0..J-1
   - Xf[0]       = sigma2 (common within-group variance sigma^2 > 0)
   - Xf[1 + j]   = ybar_j for j = 0..J-1

 Target on unconstrained state (mu, log tau, z):
   log p(mu, log tau, z | y)
   => up to constants:
        -0.5 * sum_j (ybar_j - (mu + tau z_j))^2 / (sigma2 / n_j)
        -0.5 * sum_j z_j^2
 */

#include <math.h>

__device__ float logdens(const float *theta, const float *Xf, const int *Xi) {
  if (!Xi || !Xf) return -INFINITY;

  const int J = Xi[0];
  if (J <= 0) return -INFINITY;
  if (DIM != (2 + J)) return -INFINITY;

  const float mu = theta[0];
  const float log_tau = theta[1];
  if (!isfinite(mu) || !isfinite(log_tau)) return -INFINITY;

  const float tau = expf(log_tau);
  if (!isfinite(tau) || tau <= 0.0f) return -INFINITY;

  const float sigma2 = Xf[0];
  if (!isfinite(sigma2) || sigma2 <= 0.0f) return -INFINITY;

  float ll = 0.0f;
  float lpz = 0.0f;

  for (int j = 0; j < J; j++) {
    const int nj = Xi[1 + j];
    if (nj <= 0) return -INFINITY;

    const float ybar = Xf[1 + j];
    const float zj = theta[2 + j];
    if (!isfinite(ybar) || !isfinite(zj)) return -INFINITY;

    // Likelihood: ybar_j | mu,tau,z_j ~ N(mu + tau*z_j, sigma2 / n_j)
    const float mean_j = mu + tau * zj;
    const float inv_sigj2 = ((float)nj) / sigma2; // 1 / (sigma2 / n_j)
    const float dy = ybar - mean_j;
    ll += -0.5f * dy * dy * inv_sigj2;

    // Prior on z_j: N(0,1)
    lpz += -0.5f * zj * zj;
  }

  return ll + lpz;
}
