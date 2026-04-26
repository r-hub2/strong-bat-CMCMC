# Legacy smoke test.
# For a minimal kernel example with trace plots, see inst/scripts/nne_way_normal_nc_example.R.
# Smoke test: one-way Normal random-effects (full joint, non-centered) target using CMCMC.
#
# Kernel: `NneWayNormalNC`
# State (DIM = 2 + J): (mu, log_tau, z_1..z_J) with theta_j = mu + tau*z_j.
# Data:
#   Xi = c(J, n_1..n_J)
#   Xf = c(sigma2, ybar_1..ybar_J)
#
# In RStudio: edit the parameters below, then select the code and press Run.

library(CMCMC)


set.seed(1)

J <- 40
n <- sample(5:50, J, replace = TRUE)

mu_true <- 1.0
tau_true <- 0.7
sigma2 <- 1.5^2

theta_true <- rnorm(J, mean = mu_true, sd = tau_true)
ybar <- rnorm(J, mean = theta_true, sd = sqrt(sigma2 / n))

Xi <- as.integer(c(J, n))
Xf <- c(sigma2, ybar)

K <- 200 # even
it <- 50
covit <- 5

mu0 <- mean(ybar)
tau0 <- sd(ybar)
if (!is.finite(tau0) || tau0 <= 0) tau0 <- 1

# Initialize z near (ybar - mu)/tau.
z0 <- (ybar - mu0) / tau0

init_center <- c(mu0, log(tau0), z0)
dim <- length(init_center)
stopifnot(dim == (2 + J))

init <- matrix(rnorm(K * dim, mean = init_center, sd = 0.2),
               nrow = K, ncol = dim, byrow = TRUE)

samples <- CMCMC::inca(
  init = init,
  GPUkernel = "NneWayNormalNC",
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

param <- samples[, 3:(2 + dim), drop = FALSE]
mu <- param[, 1]
tau <- exp(param[, 2])
z <- param[, 3:ncol(param), drop = FALSE]

# Recover theta draws.
theta <- sweep(z, 1, tau, `*`) + mu

print(summary(mu))
print(summary(tau))
print(dim(theta))


theta_mean <- colMeans(theta)
theta_ci <- apply(theta, 2, quantile, c(0.05, 0.95))
cbind(theta_true = theta_true, ybar = ybar, post_mean = theta_mean,
      lo = theta_ci[1,], hi = theta_ci[2,])
