# Hierarchical Beta–Binomial (joint) kernel example for CMCMC (CMCMC or INCA).
#
# Kernel: `HierBetaBinomJoint`
# State (DIM = 2 + J):
#   theta[1]     = log(alpha)
#   theta[2]     = log(beta)
#   theta[3+j]   = z_j = logit(omega_j), j=1..J
# where omega_j is the Binomial probability in experiment j.
#
# Data layout:
#   Xi = c(J, n_1..n_J, y_1..y_J) (integer)
#   X  = numeric(0) (unused by this kernel)
#
# In RStudio: edit the parameters below, then select the code and press Run.

suppressPackageStartupMessages(library(CMCMC))

sampler <- "cmcmc"  # "cmcmc" or "inca"
J <- 8L
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

# Simulate hierarchical Beta–Binomial data.
alpha_true <- 2
beta_true <- 8
n <- sample(20:200, J, replace = TRUE)
omega_true <- rbeta(J, alpha_true, beta_true)
y <- rbinom(J, size = n, prob = omega_true)

Xi <- as.integer(c(J, n, y))
Xf <- numeric(0)

# Initialisation near empirical rates.
p0 <- (y + 0.5) / (n + 1.0)
p0 <- pmin(pmax(p0, 1e-6), 1 - 1e-6)
z0 <- qlogis(p0)
init_center <- c(log(1.0), log(1.0), z0)
dim <- length(init_center)
stopifnot(dim == (2L + J))

init <- matrix(rnorm(K * dim, mean = init_center, sd = 0.2),
               nrow = K, ncol = dim, byrow = TRUE)

samples <- sampler_fun(
  init = init,
  GPUkernel = "HierBetaBinomJoint",
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

# Transform to natural parameters: alpha, beta, omega_j.
iter <- samples[, 1]
theta <- samples[, 3:(2 + dim), drop = FALSE]
alpha <- exp(theta[, 1])
beta <- exp(theta[, 2])
omega <- plogis(theta[, 3:ncol(theta), drop = FALSE])
theta_nat <- cbind(alpha = alpha, beta = beta, omega)

tr <- extract_iter_mean_trace(cbind(iter = iter, idx = samples[, 2], theta_nat), dim)

last_iter <- max(tr$iter)
tail_iters <- max(1L, min(tail_iters, last_iter))
tail_rows <- theta_nat[iter > (last_iter - tail_iters), , drop = FALSE]
post_mean <- colMeans(tail_rows)

param_names <- c("alpha", "beta", paste0("omega[", seq_len(J), "]"))
posterior_summary <- data.frame(
  param = param_names,
  mean = post_mean,
  true = c(alpha_true, beta_true, omega_true)
)

cat(sprintf("HierBetaBinomJoint (%s): J=%d, K=%d, it=%d, covit=%d\n", toupper(sampler), J, K, it, covit))
cat(sprintf("Posterior means using last %d iteration(s) (pooled over K):\n", tail_iters))
print(posterior_summary)

dims_to_plot <- seq_len(min(5L, length(post_mean)))
true_vals <- c(alpha_true, beta_true, omega_true)[dims_to_plot]
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
mtext(sprintf("HierBetaBinomJoint: population mean traces (%s), J=%d, K=%d; dashed = true value",
              toupper(sampler), J, K),
      outer = TRUE, cex = 1.05)
par(op)
