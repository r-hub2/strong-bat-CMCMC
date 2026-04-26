# Logistic regression kernel example for CMCMC (CMCMC or INCA).
#
# Kernel: `Logistic`
# State space: beta in R^d
# Data layout:
#   Xi = n (integer)
#   X  = c(as.vector(t(Xmat)), y, prior_var) where Xmat is n x d and `t()` gives row-major
#
# In RStudio: edit the parameters below, then select the code and press Run.

suppressPackageStartupMessages(library(CMCMC))

sampler <- "cmcmc"  # "cmcmc" or "inca"
d <- 10L
n <- 500L
K <- 1000L
it <- 1000L
covit <- 10L
prior_var <- 100
device <- 0L
tail_iters <- 200L
seed <- 1L

sampler <- tolower(sampler)

if (d <= 0) stop("d must be > 0.")
if (n <= 0) stop("n must be > 0.")
if (K <= 0) stop("K must be > 0.")
if (it <= 0) stop("it must be > 0.")
if (covit <= 0) stop("covit must be > 0.")
if (!is.finite(prior_var) || prior_var <= 0) stop("prior_var must be > 0.")
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

# Simulate data.
Xmat <- matrix(rnorm(n * d), nrow = n, ncol = d)
beta_true <- rnorm(d, sd = 0.5)
eta <- as.vector(Xmat %*% beta_true)
p <- 1 / (1 + exp(-eta))
y <- rbinom(n, size = 1, prob = p)

X_row_major <- as.vector(t(Xmat))
Xf <- c(X_row_major, as.numeric(y), prior_var)
Xi <- as.integer(n)

init <- matrix(rnorm(K * d, sd = 0.1), nrow = K, ncol = d)

samples <- sampler_fun(
  init = init,
  GPUkernel = "Logistic",
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
tr <- extract_iter_mean_trace(samples, d)
last_iter <- max(tr$iter)
tail_iters <- max(1L, min(tail_iters, last_iter))
tail_rows <- samples[samples[, 1] > (last_iter - tail_iters), , drop = FALSE]
beta_tail <- tail_rows[, 3:(2 + d), drop = FALSE]
post_mean <- colMeans(beta_tail)

posterior_summary <- data.frame(
  param = paste0("beta[", seq_len(d), "]"),
  mean = post_mean,
  true = beta_true
)

cat(sprintf("Logistic (%s): d=%d, n=%d, K=%d, it=%d, covit=%d, prior_var=%.3g\n",
            toupper(sampler), d, n, K, it, covit, prior_var))
cat(sprintf("Posterior means using last %d iteration(s) (pooled over K):\n", tail_iters))
print(posterior_summary)

dims_to_plot <- seq_len(min(5L, d))
op <- par(no.readonly = TRUE)
par(mfrow = c(length(dims_to_plot), 1), mar = c(3.2, 3.6, 2.0, 1.0), oma = c(0, 0, 2.5, 0))
for (j in dims_to_plot) {
  ylim <- range(c(tr$mean[, j], beta_true[j]), finite = TRUE)
  pad <- 0.05 * diff(ylim)
  if (!is.finite(pad) || pad <= 0) pad <- 0.2
  ylim <- ylim + c(-pad, pad)

  plot(tr$iter, tr$mean[, j], type = "l", lwd = 1.1, ylim = ylim,
       xlab = "Iteration", ylab = sprintf("mean(beta[%d])", j))
  abline(h = beta_true[j], col = "gray40", lty = 2, lwd = 1)
}
mtext(sprintf("Logistic: population mean traces (%s), d=%d, K=%d; dashed = true beta", toupper(sampler), d, K),
      outer = TRUE, cex = 1.05)
par(op)
