# Smoke test: logistic regression target (DIM=10) using CMCMC.
#
# In RStudio: edit the parameters below, then select the code and press Run.

suppressPackageStartupMessages(library(CMCMC))


set.seed(1)
d <- 5
n <- 10
K <- 200 # even
it <- 20
covit <- 5
dims_to_plot <- seq_len(min(d, 5L))

extract_iter_mean_trace <- function(samples, d) {
  stopifnot(is.matrix(samples))
  iter <- samples[, 1]
  vals <- samples[, 3:(2 + d), drop = FALSE]
  sums <- rowsum(vals, group = iter, reorder = TRUE)
  counts <- as.numeric(table(iter)[rownames(sums)])
  means <- sweep(sums, 1, counts, "/")
  list(iter = as.integer(rownames(sums)), mean = means)
}

# Simulate data
X <- matrix(rnorm(n * d), nrow = n, ncol = d)
beta_true <- rnorm(d, sd = 0.5)
eta <- as.vector(X %*% beta_true)
p <- 1 / (1 + exp(-eta))
y <- rbinom(n, size = 1, prob = p)

# Pack kernel inputs:
# Xf = c(as.vector(t(X))? no: kernel expects row-major, so as.vector(X) in R is column-major.
# We must send row-major layout: use t(X) then as.vector gives row-major by original rows.
X_row_major <- as.vector(t(X))
prior_var <- 100
Xf <- c(X_row_major, as.numeric(y), prior_var)
Xi <- as.integer(n)

# Init population
init <- matrix(rnorm(K * d, sd = 0.1), nrow = K, ncol = d)

samples_cmcmc <- CMCMC::cmcmc(
  init = init,
  GPUkernel = "Logistic",
  it = it,
  covit = covit,
  p = K,
  X = Xf,
  Xi = Xi,
  saved_iterations = 0,
  seed = 123,
  device = 0
)

samples_inca <- CMCMC::inca(
  init = init,
  GPUkernel = "Logistic",
  it = it,
  covit = covit,
  p = K,
  X = Xf,
  Xi = Xi,
  saved_iterations = 0,
  seed = 123,
  device = 0
)

stopifnot(is.matrix(samples_cmcmc), is.matrix(samples_inca))
print(list(cmcmc_dim = dim(samples_cmcmc), inca_dim = dim(samples_inca)))

last_cmcmc <- max(samples_cmcmc[, 1])
last_inca <- max(samples_inca[, 1])
posterior_summary <- data.frame(
  param = paste0("beta[", seq_len(d), "]"),
  true = beta_true,
  cmcmc_mean = colMeans(samples_cmcmc[samples_cmcmc[, 1] == last_cmcmc, 3:(2 + d), drop = FALSE]),
  inca_mean = colMeans(samples_inca[samples_inca[, 1] == last_inca, 3:(2 + d), drop = FALSE])
)
print(posterior_summary)

tr_cmcmc <- extract_iter_mean_trace(samples_cmcmc, d)
tr_inca <- extract_iter_mean_trace(samples_inca, d)
op <- par(no.readonly = TRUE)

par(mfrow = c(length(dims_to_plot), 2), mar = c(3, 3.2, 2, 1), oma = c(0, 0, 2, 0))
for (j in dims_to_plot) {
  ylim <- range(c(tr_cmcmc$mean[, j], tr_inca$mean[, j], beta_true[j]), finite = TRUE)
  pad <- 0.05 * diff(ylim)
  if (!is.finite(pad) || pad == 0) pad <- 0.1
  ylim <- ylim + c(-pad, pad)

  plot(tr_cmcmc$iter, tr_cmcmc$mean[, j],
       type = "l", xlab = "iter", ylab = paste0("beta[", j, "]"),
       main = "CMCMC", col = "steelblue", lwd = 1, ylim = ylim)
  abline(h = beta_true[j], col = "gray40", lty = 2)

  plot(tr_inca$iter, tr_inca$mean[, j],
       type = "l", xlab = "iter", ylab = paste0("beta[", j, "]"),
       main = "INCA", col = "firebrick", lwd = 1, ylim = ylim)
  abline(h = beta_true[j], col = "gray40", lty = 2)
}
mtext(
  sprintf("Logistic mean trace (dims=%s, K=%d, it=%d, covit=%d); dashed = true beta",
          paste(dims_to_plot, collapse = ","), K, it, covit),
  outer = TRUE, cex = 1.1
)
par(op)
