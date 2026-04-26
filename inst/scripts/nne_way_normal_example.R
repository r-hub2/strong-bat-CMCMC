# One-way Normal random-effects kernel example for CMCMC (CMCMC or INCA).
#
# Kernel: `NneWayNormal` (centered)
# State (DIM = 2 + J): (mu, log(tau), theta_1..theta_J) with tau = exp(log(tau)).
#
# Data layout:
#   Xi = c(J, n_1..n_J) (integer)
#   X  = c(sigma2, ybar_1..ybar_J)
#
# In RStudio: edit the parameters below, then select the code and press Run.

suppressPackageStartupMessages(library(CMCMC))

sampler <- "cmcmc"  # "cmcmc" or "inca"
J <- 20L
K <- 1000L
it <- 1000L
covit <- 10L
device <- 0L
tail_iters <- 200L
seed <- 1L

sampler <- tolower(sampler)

if (J <= 0) stop("J must be > 0.")
if (K <= 0) stop("K must be > 0.")
if (it <= 0) stop("it must be > 0.")
if (covit <= 0) stop("covit must be > 0.")
if (K %% 2L != 0L) {
  warning("K must be even; incrementing K by 1.")
  K <- K + 1L
}


sampler_fun <- switch(
  sampler,
  cmcmc = CMCMC::cmcmc,
  inca = CMCMC::inca,
  stop("sampler must be 'cmcmc' or 'inca'.")
)

extract_iter_mean_trace <- function(samples, dim) {
  stopifnot(is.matrix(samples))
  iter <- samples[, 1]
  vals <- samples[, 3:(2 + dim), drop = FALSE]
  sums <- rowsum(vals, group = iter, reorder = TRUE)
  counts <- as.numeric(table(iter)[rownames(sums)])
  means <- sweep(sums, 1, counts, "/")
  list(iter = as.integer(rownames(sums)), mean = means)
}

set.seed(seed)

# Simulate data via sufficient statistics ybar_j.
mu_true <- 1.0
tau_true <- 0.7
sigma2 <- 1.5^2
n <- sample(5:50, J, replace = TRUE)
theta_true <- rnorm(J, mean = mu_true, sd = tau_true)
ybar <- rnorm(J, mean = theta_true, sd = sqrt(sigma2 / n))

Xi <- as.integer(c(J, n))
Xf <- c(sigma2, ybar)

# Initialisation near empirical summaries.
mu0 <- mean(ybar)
tau0 <- sd(ybar)
if (!is.finite(tau0) || tau0 <= 0) tau0 <- 1
theta0 <- ybar

init_center <- c(mu0, log(tau0), theta0)
dim <- length(init_center)
stopifnot(dim == (2L + J))
init <- matrix(rnorm(K * dim, mean = init_center, sd = 0.2),
               nrow = K, ncol = dim, byrow = TRUE)

samples <- sampler_fun(
  init = init,
  GPUkernel = "NneWayNormal",
  it = it,
  covit = covit,
  p = K,
  X = Xf,
  Xi = Xi,
  saved_iterations = 0,
  seed = seed,
  device = device
)

stopifnot(is.matrix(samples))

iter <- samples[, 1]
theta <- samples[, 3:(2 + dim), drop = FALSE]
mu <- theta[, 1]
tau <- exp(theta[, 2])
theta_j <- theta[, 3:ncol(theta), drop = FALSE]
theta_nat <- cbind(mu = mu, tau = tau, theta_j)

tr <- extract_iter_mean_trace(cbind(iter = iter, idx = samples[, 2], theta_nat), dim)
last_iter <- max(tr$iter)
tail_iters <- max(1L, min(tail_iters, last_iter))
tail_rows <- theta_nat[iter > (last_iter - tail_iters), , drop = FALSE]
post_mean <- colMeans(tail_rows)

param_names <- c("mu", "tau", paste0("theta[", seq_len(J), "]"))
posterior_summary <- data.frame(
  param = param_names,
  mean = post_mean,
  true = c(mu_true, tau_true, theta_true)
)

cat(sprintf("NneWayNormal (%s): J=%d, K=%d, it=%d, covit=%d\n", toupper(sampler), J, K, it, covit))
cat(sprintf("Posterior means using last %d iteration(s) (pooled over K):\n", tail_iters))
print(posterior_summary)

dims_to_plot <- seq_len(min(5L, length(post_mean)))
true_vals <- c(mu_true, tau_true, theta_true)[dims_to_plot]
op <- par(no.readonly = TRUE)
par(mfrow = c(length(dims_to_plot), 1), mar = c(3.2, 3.6, 2.0, 1.0), oma = c(0, 0, 2.5, 0))
for (j in dims_to_plot) {
  ylim <- range(c(tr$mean[, j], true_vals[j]), finite = TRUE)
  pad <- 0.05 * diff(ylim)
  if (!is.finite(pad) || pad <= 0) pad <- 0.2
  ylim <- ylim + c(-pad, pad)

  plot(tr$iter, tr$mean[, j], type = "l", lwd = 1.1, ylim = ylim,
       xlab = "Iteration", ylab = sprintf("mean(%s)", param_names[j]))
  abline(h = true_vals[j], col = "gray40", lty = 2, lwd = 1)
}
mtext(sprintf("NneWayNormal: population mean traces (%s), J=%d, K=%d; dashed = true value",
              toupper(sampler), J, K),
      outer = TRUE, cex = 1.05)
par(op)
