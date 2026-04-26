# Legacy script (kept for thesis workflow).
# For a minimal kernel example, see inst/scripts/hier_beta_binom_joint_example.R.
# Beta-binomial hierarchical model for rat tumor data (LearnBayes::rat), run with CMCMC.
#
# Reference: Rat tumor example, hierarchical Beta–Binomial with hyperprior
#   p(alpha, beta) ∝ (alpha + beta)^(-5/2).
#
# GPU kernel: `HierBetaBinomJoint`
# State (DIM = 2 + J):
#   theta[1]     = log(alpha)
#   theta[2]     = log(beta)
#   theta[3+j]   = z_j = logit(omega_j), j = 1..J
# where omega_j is the tumor rate for experiment j.
#
# Data:
#   Xi = c(J, n_1..n_J, y_1..y_J)   (integer)
#   X  = numeric(0)                 (not used by this kernel)
#
# In RStudio: edit the parameters below, then select the code and press Run.

suppressPackageStartupMessages({
  library(CMCMC)
  library(LearnBayes)
  library(MASS)
})


u <- "https://sites.stat.columbia.edu/gelman/book/data/rats.asc"

txt <- paste(readLines(u, warn = FALSE), collapse = " ")

# keep everything starting at the header "y N"
txt <- sub("^.*\\by\\s+N\\b", "y N", txt)

# extract all integers
nums <- as.integer(unlist(regmatches(txt, gregexpr("[0-9]+", txt))))
stopifnot(length(nums) %% 2 == 0)

rats <- data.frame(
  y = nums[seq(1, length(nums), by = 2)],
  N = nums[seq(2, length(nums), by = 2)]
)



# Use the rat tumor dataset prepared by the user.
# Expected format: a data.frame with columns `y` (tumor counts) and `N` (sample sizes),
# matching the table shown in the book.
if (!exists("rats", inherits = FALSE)) {
  data(rat)
  rats <- rat
}
stopifnot(is.data.frame(rats))
stopifnot("y" %in% names(rats))
stopifnot(("N" %in% names(rats)) || ("n" %in% names(rats)))

y <- as.integer(rats$y)
n <- as.integer(if ("N" %in% names(rats)) rats$N else rats$n)
stopifnot(length(y) == length(n), length(y) > 0)
stopifnot(all(is.finite(y)), all(is.finite(n)))
stopifnot(all(y >= 0), all(n > 0), all(y <= n))

J <- length(y)
dim <- 2 + J

Xi <- as.integer(c(J, n, y))

logit <- function(p) log(p / (1 - p))
inv_logit <- function(z) 1 / (1 + exp(-z))

# Initialisation: start near the point estimate used in the text (alpha,beta) ~ (1.4,8.6)
# and set z_j to the empirical logit rate with mild smoothing.
alpha0 <- 1.4
beta0 <- 8.6

p0 <- (y + 0.5) / (n + 1.0)
p0 <- pmin(pmax(p0, 1e-6), 1 - 1e-6)
z0 <- logit(p0)

init_center <- c(log(alpha0), log(beta0), z0)
stopifnot(length(init_center) == dim)

K <- 1024*4
it <- 20000
covit <- 20

init <- matrix(rnorm(K * dim, mean = init_center, sd = 0.15),
               nrow = K, ncol = dim, byrow = TRUE)

samples <- CMCMC::cmcmc(
  init = init,
  GPUkernel = "HierBetaBinomJoint",
  it = it,
  covit = covit,
  p = K,
  X = numeric(0),
  Xi = Xi,
  saved_iterations = 1,
  seed = 123,
  device = 0
)

stopifnot(is.matrix(samples))
cat("samples dim:", paste(dim(samples), collapse = " x "), "\n")

param <- samples[, 3:(2 + dim), drop = FALSE]
log_alpha <- param[, 1]
log_beta <- param[, 2]
z <- param[, 3:ncol(param), drop = FALSE]

alpha <- exp(log_alpha)
beta <- exp(log_beta)
mu <- alpha / (alpha + beta)
S <- alpha + beta

cat("\nPosterior summaries (CMCMC, joint kernel):\n")
print(summary(alpha))
print(summary(beta))
cat("\nDerived:\n")
print(summary(mu))
print(summary(S))

# Posterior mean tumor rates per experiment.
omega_mean <- colMeans(inv_logit(z))
cat("\nFirst 10 posterior mean omega_j:\n")
print(head(omega_mean, 10))

# Compare to the PDF's contour parameterisation: (log(alpha/beta), log(alpha+beta)).
eta1 <- log_alpha - log_beta
eta2 <- log(alpha + beta)
cat("\nPosterior summaries for contour parameterisation:\n")
print(summary(eta1))
print(summary(eta2))

if (interactive()) {
  op <- par(no.readonly = TRUE)
  
  # Posterior density diagnostics.
  par(mfrow = c(2, 2))
  plot(density(mu), main = "Posterior density of mean mu", xlab = "mu = alpha/(alpha+beta)")
  plot(density(S), main = "Posterior density of size S", xlab = "S = alpha+beta")
  plot(density(eta1), main = "Posterior density of log(alpha/beta)", xlab = "log(alpha/beta)")
  plot(density(eta2), main = "Posterior density of log(alpha+beta)", xlab = "log(alpha+beta)")
  
  # Recreate the style of the book's Figure 5.3 using CMCMC draws:
  # (a) 2D contour of the marginal posterior density of (log(alpha/beta), log(alpha+beta))
  #     approximated via a Gaussian kernel density estimate.
  # (b) Scatterplot of 1000 draws in the same coordinates.
  set.seed(1)
  keep <- if (length(eta1) >= 1000) sample.int(length(eta1), 1000) else seq_along(eta1)
  eta1_s <- eta1[keep]
  eta2_s <- eta2[keep]
  
  xlim <- as.numeric(quantile(eta1, c(0.01, 0.99), names = FALSE))
  ylim <- as.numeric(quantile(eta2, c(0.01, 0.99), names = FALSE))
  
  kde <- MASS::kde2d(eta1, eta2, n = 120, lims = c(xlim, ylim))
  zmax <- max(kde$z, na.rm = TRUE)
  levels <- zmax * seq(0.05, 0.95, by = 0.10)
  
  par(mfrow = c(1, 2), mar = c(4.2, 4.2, 1.2, 0.8), mgp = c(2.2, 0.7, 0))
  contour(kde$x, kde$y, kde$z,
          xlab = "log(alpha/beta)", ylab = "log(alpha + beta)",
          levels = levels, drawlabels = FALSE)
  plot(eta1_s, eta2_s, pch = 16, cex = 0.4,
       xlab = "log(alpha/beta)", ylab = "log(alpha + beta)",
       xlim = range(kde$x), ylim = range(kde$y))
  par(op)
} else {
  message("Non-interactive session: skipping plots.")
}

draws <- data.frame(
  log_alpha = log_alpha,
  log_beta = log_beta,
  alpha = alpha,
  beta = beta,
  mu = mu,
  S = S,
  log_alpha_over_beta = eta1,
  log_alpha_plus_beta = eta2
)

omega <- inv_logit(z)
omega_summary <- data.frame(
  j = seq_len(J),
  y = y,
  n = n,
  y_over_n = y / n,
  mean = colMeans(omega),
  median = apply(omega, 2, median),
  q05 = apply(omega, 2, quantile, probs = 0.05),
  q95 = apply(omega, 2, quantile, probs = 0.95)
)

hyper_summary <- data.frame(
  quantity = c("alpha", "beta", "mu", "S", "log(alpha/beta)", "log(alpha+beta)"),
  min = c(min(alpha), min(beta), min(mu), min(S), min(eta1), min(eta2)),
  q25 = c(quantile(alpha, 0.25), quantile(beta, 0.25), quantile(mu, 0.25), quantile(S, 0.25),
          quantile(eta1, 0.25), quantile(eta2, 0.25)),
  median = c(median(alpha), median(beta), median(mu), median(S), median(eta1), median(eta2)),
  mean = c(mean(alpha), mean(beta), mean(mu), mean(S), mean(eta1), mean(eta2)),
  q75 = c(quantile(alpha, 0.75), quantile(beta, 0.75), quantile(mu, 0.75), quantile(S, 0.75),
          quantile(eta1, 0.75), quantile(eta2, 0.75)),
  max = c(max(alpha), max(beta), max(mu), max(S), max(eta1), max(eta2))
)

cat("\nHyperparameter summary table:\n")
print(hyper_summary)
cat("\nFirst 10 omega summary rows:\n")
print(head(omega_summary, 10))
