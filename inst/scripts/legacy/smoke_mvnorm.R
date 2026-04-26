# Legacy smoke test.
# For a minimal kernel example with trace plots, see inst/scripts/mvnorm_example.R.
# Smoke test for CMCMC on a CUDA-capable machine.
#
# Expected: compiles the MVNorm kernel via nvcc, runs a tiny sampler call, and returns a matrix.
#
# In RStudio: edit the parameters below, then select the code and press Run.

library(CMCMC)

# CMCMC tries to detect CUDA paths when the package is loaded.

set.seed(1)
d <- 2
K <- 100 # must be even for Louis engine; larger K helps covariance be PD
it <- 2
covit <- 1

init <- matrix(rnorm(K * d), nrow = K, ncol = d)

# MVNorm kernel expects X = c(mu, SigmaInv) where mu is length d and SigmaInv is d*d.
mu <- rep(0, d)
Sigma <- diag(d)
SigmaInv <- as.vector(solve(Sigma))
X <- c(mu, SigmaInv)

samples <- CMCMC::cmcmc(
  init = init,
  GPUkernel = "MVNorm",
  it = it,
  covit = covit,
  p = K,
  X = X,
  Xi = integer(0),
  saved_iterations = 1,
  seed = 123,
  device = 0
)

stopifnot(is.matrix(samples))
print(dim(samples))
print(head(samples))
