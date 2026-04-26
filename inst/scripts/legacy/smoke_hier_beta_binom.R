# Legacy smoke test.
# For a minimal kernel example with trace plots, see inst/scripts/hier_beta_binom_joint_example.R.
# Smoke test: hierarchical Beta–Binomial (joint) target using CMCMC.
#
# Kernel: `HierBetaBinomJoint`
# State (DIM = 2 + J): (log_alpha, log_beta, z_1..z_J) with theta_j = sigmoid(z_j).
# Data: Xi = c(J, n_1..n_J, y_1..y_J)
#
# In RStudio: edit the parameters below, then select the code and press Run.

library(CMCMC)


set.seed(1)

J <- 5
n <- sample(20:200, J, replace = TRUE)

alpha_true <- 2
beta_true <- 8
theta_true <- rbeta(J, alpha_true, beta_true)
y <- rbinom(J, size = n, prob = theta_true)

Xi <- as.integer(c(J, n, y))
Xf <- numeric() # unused by this kernel

K <- 200 # even
it <- 50
covit <- 5

# Initialize near a reasonable point.
alpha0 <- 1
beta0 <- 1
theta0 <- (y + 0.5) / (n + 1) # in (0,1)
z0 <- qlogis(theta0)          # unconstrained

init_center <- c(log(alpha0), log(beta0), z0)
dim <- length(init_center)
stopifnot(dim == (2 + J))

init <- matrix(rnorm(K * dim, mean = init_center, sd = 0.2),
               nrow = K, ncol = dim, byrow = TRUE)

samples <- CMCMC::inca(
  init = init,
  GPUkernel = "HierBetaBinomJoint",
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

# Quick sanity: convert final population to alpha/beta.
param <- samples[, 3:(2 + dim), drop = FALSE]
log_alpha <- param[, 1]
log_beta <- param[, 2]

alpha <- exp(log_alpha)
beta <- exp(log_beta)
z <- param[, 3:ncol(param), drop = FALSE]
thetas <- plogis(z)  # same as 1/(1+exp(-z)), more stable

print(summary(alpha))
print(summary(beta))
apply(thetas,2,mean)
