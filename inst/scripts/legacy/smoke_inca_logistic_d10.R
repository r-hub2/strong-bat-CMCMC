# Legacy smoke test.
# For a minimal kernel example with trace plots, see inst/scripts/logistic_example.R.
# Smoke test for INCA (global covariance) on a CUDA-capable machine.
#
# In RStudio: edit the parameters below, then select the code and press Run.

library(CMCMC)


set.seed(1)
d <- 10
K <- 200
it <- 20
covit <- 1

# Simple synthetic logistic regression data (n observations, d covariates)
n <- 200
X <- matrix(rnorm(n * d), nrow = n, ncol = d)
beta_true <- rnorm(d, sd = 0.2)
lin <- as.vector(X %*% beta_true)
p_true <- 1 / (1 + exp(-lin))
y <- rbinom(n, size = 1, prob = p_true)

init <- matrix(rnorm(K * d, sd = 0.1), nrow = K, ncol = d)

# Logistic kernel convention:
#   Xi[0] = n
#   Xf = [X (row-major length n*d), y (length n), prior_var (length 1)]
prior_var <- 10
Xf <- c(as.vector(t(X)), as.numeric(y), as.numeric(prior_var))
Xi <- as.integer(n)

samples <- CMCMC::inca(
  init = init,
  GPUkernel = "Logistic",
  it = it,
  covit = covit,
  p = K,
  X = Xf,
  Xi = Xi,
  saved_iterations = 1,
  seed = 123,
  device = 0
)

stopifnot(is.matrix(samples))
print(dim(samples))
print(head(samples))
