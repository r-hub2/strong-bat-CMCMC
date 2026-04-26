# MVNorm kernel example for CMCMC (CMCMC or INCA).
#
# Kernel: `MVNorm`
# State space: theta in R^D
# Data layout:
#   Xi = integer(0)
#   X  = c(mu, SigmaInv) where mu is length D and SigmaInv is D*D (column-major)
#
# In RStudio: edit the parameters below, then select the code and press Run.

suppressPackageStartupMessages(library(CMCMC))

sampler <- "cmcmc"  # "cmcmc" or "inca"
D <- 10L
rho <- 0.9
K <- 1000L
it <- 1000L
covit <- 10L
device <- 0L
tail_iters <- 200L
seed <- 1L

sampler <- tolower(sampler)

if (!is.finite(rho) || rho <= -1 || rho >= 1) stop("rho must be in (-1,1).")
if (D <= 0) stop("D must be > 0.")
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

extract_iter_mean_trace <- function(samples, d) {
  stopifnot(is.matrix(samples))
  iter <- samples[, 1]
  vals <- samples[, 3:(2 + d), drop = FALSE]
  sums <- rowsum(vals, group = iter, reorder = TRUE)
  counts <- as.numeric(table(iter)[rownames(sums)])
  means <- sweep(sums, 1, counts, "/")
  list(iter = as.integer(rownames(sums)), mean = means)
}

set.seed(seed)

mu <- rep(0, D)
Sigma <- matrix(rho, nrow = D, ncol = D)
diag(Sigma) <- 1
SigmaInv <- as.vector(solve(Sigma))
X <- c(mu, SigmaInv)
Xi <- integer(0)

init <- matrix(runif(K * D, min = -5, max = 6), nrow = K, ncol = D)

samples <- sampler_fun(
  init = init,
  GPUkernel = "MVNorm",
  it = it,
  covit = covit,
  p = K,
  X = X,
  Xi = Xi,
  saved_iterations = 0,
  seed = seed,
  device = device
)

stopifnot(is.matrix(samples))
tr <- extract_iter_mean_trace(samples, D)

last_iter <- max(tr$iter)
tail_iters <- max(1L, min(tail_iters, last_iter))
tail_rows <- samples[samples[, 1] > (last_iter - tail_iters), , drop = FALSE]
theta_tail <- tail_rows[, 3:(2 + D), drop = FALSE]
post_mean <- colMeans(theta_tail)

posterior_summary <- data.frame(
  param = paste0("theta[", seq_len(D), "]"),
  mean = post_mean,
  true = mu
)

cat(sprintf("MVNorm (%s): D=%d, K=%d, it=%d, covit=%d, rho=%.3f\n", toupper(sampler), D, K, it, covit, rho))
cat(sprintf("Posterior means using last %d iteration(s) (pooled over K):\n", tail_iters))
print(posterior_summary)

dims_to_plot <- seq_len(min(5L, D))
op <- par(no.readonly = TRUE)
par(mfrow = c(length(dims_to_plot), 1), mar = c(3.2, 3.6, 2.0, 1.0), oma = c(0, 0, 2.5, 0))
for (j in dims_to_plot) {
  plot(tr$iter, tr$mean[, j], type = "l", lwd = 1.1,
       xlab = "Iteration", ylab = sprintf("mean(theta[%d])", j))
  abline(h = 0, col = "gray40", lty = 2, lwd = 1)
}
mtext(sprintf("MVNorm: population mean traces (%s), D=%d, K=%d; dashed = true mean 0", toupper(sampler), D, K),
      outer = TRUE, cex = 1.05)
par(op)
