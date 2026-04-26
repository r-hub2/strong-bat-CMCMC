# Legacy comparison script.
# For one-script-per-kernel examples, see inst/scripts/mvnorm_example.R.
# Compare CMCMC vs INCA on a multivariate Normal (MVNorm) target.
#
# Produces:
#  - Timings printed to stdout (excluding first-time kernel compilation via warmup)
#  - Trace plot (per-iteration mean across chains, first 5 dims) comparing both methods
#
# In RStudio: edit the parameters below, then select the code and press Run.

library(CMCMC)


set.seed(1)
d <- 2
K <- 1000
it <- 1000
covit <- 10
device <- 0

dims_to_plot <- seq_len(min(d, 5L))

# MVNorm kernel expects X = c(mu, SigmaInv) where mu is length d and SigmaInv is d*d.
mu <- rep(0, d)
tau <- matrix(0.9,d,d)
diag(tau) <- 1
Sigma <- tau
SigmaInv <- as.vector(solve(Sigma))
X <- c(mu, SigmaInv)

init <- matrix(runif(K * d,min = -5,max = 6), nrow = K, ncol = d)

extract_iter_mean_trace <- function(samples, d) {
  stopifnot(is.matrix(samples))
  iter <- samples[, 1]
  vals <- samples[, 3:(2 + d), drop = FALSE]
  sums <- rowsum(vals, group = iter, reorder = TRUE)
  counts <- as.numeric(table(iter)[rownames(sums)])
  means <- sweep(sums, 1, counts, "/")
  list(iter = as.integer(rownames(sums)), mean = means)
}

warmup <- function(fn) {
  invisible(fn(it = 2, saved_iterations = 1))
}

run_cmcmc <- function(it, saved_iterations) {
  CMCMC::cmcmc(
    init = init,
    GPUkernel = "MVNorm",
    it = it,
    covit = covit,
    p = K,
    X = X,
    Xi = integer(0),
    saved_iterations = saved_iterations,
    seed = 123,
    device = device
  )
}

run_inca <- function(it, saved_iterations) {
  CMCMC::inca(
    init = init,
    GPUkernel = "MVNorm",
    it = it,
    covit = covit,
    p = K,
    X = X,
    Xi = integer(0),
    saved_iterations = saved_iterations,
    seed = 123,
    device = device
  )
}

cat("Warming up (compile kernels)...\n")
warmup(run_cmcmc)
warmup(run_inca)

cat("\nRunning CMCMC...\n")
t_cmcmc <- system.time({
  samps_cmcmc <- run_cmcmc(it = it, saved_iterations = 0)
})

cat("\nRunning INCA...\n")
t_inca <- system.time({
  samps_inca <- run_inca(it = it, saved_iterations = 0)
})

cat("\nTimings (seconds):\n")
print(list(cmcmc = t_cmcmc, inca = t_inca))


cov_ml <- function(x) {
  n <- nrow(x)
  if (n <= 1) return(matrix(NA_real_, ncol(x), ncol(x)))
  cov(x) * (n - 1) / n
}

extract_final_population <- function(samples, it, d) {
  s <- samples[samples[, 1] == it, , drop = FALSE]
  s[, 3:(2 + d), drop = FALSE]
}

cat("\nCovariance check (population covariance, divisor n):\n")
final_cmcmc <- samps_cmcmc[samps_cmcmc[, 1] == it, , drop = FALSE]
popA <- final_cmcmc[final_cmcmc[, 2] <= (K / 2), 3:(2 + d), drop = FALSE]
popB <- final_cmcmc[final_cmcmc[, 2] >  (K / 2), 3:(2 + d), drop = FALSE]
covA_r <- cov_ml(popA)
covB_r <- cov_ml(popB)
cat("  CMCMC group A covariance:\n")
print(covA_r)
cat("  CMCMC group B covariance:\n")
print(covB_r)

pop_all <- extract_final_population(samps_inca, it, d)
cov_inca_r <- cov_ml(pop_all)
cat("  INCA covariance:\n")
print(cov_inca_r)

tr_cmcmc <- extract_iter_mean_trace(samps_cmcmc, d)
tr_inca <- extract_iter_mean_trace(samps_inca, d)

op <- par(no.readonly = TRUE)

par(mfrow = c(length(dims_to_plot), 2), mar = c(3, 3.2, 2, 1), oma = c(0, 0, 2, 0))
for (j in dims_to_plot) {
  plot(tr_cmcmc$iter, tr_cmcmc$mean[, j],
       type = "l", xlab = "iter", ylab = paste0("dim ", j),
       main = "CMCMC", col = "steelblue", lwd = 1, ylim = c(min(tr_cmcmc$mean[, j],-0.1), max(tr_cmcmc$mean[, j])) )
  abline(h = 0, col = "gray60", lty = 2)
  
  plot(tr_inca$iter, tr_inca$mean[, j],
       type = "l", xlab = "iter", ylab = paste0("dim ", j),
       main = "INCA", col = "firebrick", lwd = 1, ylim = c(min(tr_inca$mean[, j],-0.1), max(tr_inca$mean[, j])))
  abline(h = 0, col = "gray60", lty = 2)
}
mtext(sprintf("MVNorm mean trace: dims=%s (d=%d), K=%d, it=%d, covit=%d",
              paste(dims_to_plot, collapse = ","), d, K, it, covit),
      outer = TRUE, cex = 1.1)
par(op)
