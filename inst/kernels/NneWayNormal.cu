/*
 One-way Normal random-effects model (full joint) target kernel for CMCMC.

 State space (DIM must equal 2 + J, where J is Xi[0]):
   theta[0]     = mu
   theta[1]     = log(tau)
   theta[2 + j] = theta_j   for j = 0..J-1
 with tau = exp(log(tau)) > 0.

 Data input (recommended layout):
   - Xi[0]       = J
   - Xi[1 + j]   = n_j    for j = 0..J-1
   - Xf[0]       = sigma2 (common within-group variance sigma^2 > 0)
   - Xf[1 + j]   = ybar_j for j = 0..J-1

 Likelihood (using sufficient statistics):
   ybar_j | theta_j ~ Normal(theta_j, sigma^2 / n_j)

 Prior (random effects):
   theta_j | mu, tau ~ Normal(mu, tau^2)

 Hyperprior:
   p(mu, log tau) ∝ 1  (flat in (mu, log tau))

 Target on unconstrained state (mu, log tau, theta):
   log p(mu, log tau, theta | y)
   => up to constants:
        -0.5 * sum_j (ybar_j - theta_j)^2 / (sigma2 / n_j)
        -0.5 * sum_j (theta_j - mu)^2 / tau^2
        -J * log(tau)
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

  const float inv_tau2 = expf(-2.0f * log_tau); // 1 / tau^2
  if (!isfinite(inv_tau2) || inv_tau2 <= 0.0f) return -INFINITY;

  const float sigma2 = Xf[0];
  if (!isfinite(sigma2) || sigma2 <= 0.0f) return -INFINITY;

  float ll = 0.0f;
  float lp = 0.0f;

  for (int j = 0; j < J; j++) {
    const int nj = Xi[1 + j];
    if (nj <= 0) return -INFINITY;

    const float ybar = Xf[1 + j];
    const float thetaj = theta[2 + j];
    if (!isfinite(ybar) || !isfinite(thetaj)) return -INFINITY;

    // Likelihood: ybar_j | theta_j ~ N(theta_j, sigma2 / n_j)
    const float inv_sigj2 = ((float)nj) / sigma2; // 1 / (sigma2 / n_j)
    const float dy = ybar - thetaj;
    ll += -0.5f * dy * dy * inv_sigj2;

    // Prior: theta_j | mu,tau ~ N(mu, tau^2)
    const float dt = thetaj - mu;
    lp += -0.5f * dt * dt * inv_tau2 - log_tau;
  }

  return ll + lp;
}
