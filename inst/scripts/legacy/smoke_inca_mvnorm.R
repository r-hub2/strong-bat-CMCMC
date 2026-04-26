# Legacy smoke test.
# For a minimal kernel example with trace plots, see inst/scripts/mvnorm_example.R.
# Smoke test for INCA (global cumulative covariance) with MVNorm target.
#
# In RStudio: edit the parameters below, then select the code and press Run.

library(CMCMC)


set.seed(1)
d <- 2
K <- 200
it <- 20
covit <- 5

init <- matrix(rnorm(K * d), nrow = K, ncol = d)

# MVNorm kernel expects X = c(mu, SigmaInv) where mu is length d and SigmaInv is d*d.
mu <- rep(0, d)
Sigma <- diag(d)
SigmaInv <- as.vector(solve(Sigma))
X <- c(mu, SigmaInv)

samples <- CMCMC::inca(
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
