# Legacy script.
# For a minimal kernel example with trace plots, see inst/scripts/nne_way_normal_example.R.
library(CMCMC)
library(mcmc)


set.seed(1)
J <- 10
n <- sample(5:10, J, replace=TRUE)

mu_true <- 1.0
tau_true <- 0.7
sigma_true <- 1.5
sigma2 <- sigma_true^2

theta_true <- rnorm(J, mu_true, tau_true)
x <- lapply(1:J, \(j) rnorm(n[j], theta_true[j], sigma_true))
ybar <- sapply(x, mean)

# INCA
Xi <- as.integer(c(J, n))
Xf <- c(sigma2, ybar)
K <- 1000
dim <- J + 2
covit <- 5
init <- matrix(0, nrow=K, ncol=dim)

samps <- CMCMC::cmcmc(init=init, GPUkernel="NneWayNormal",
                           it=10000, covit=covit, p=K, X=Xf, Xi=Xi,
                           saved_iterations=1, seed=123, device=0)
mu_inca <- samps[, 3]

# metrop on the SAME x
post <- function(eta) {
  theta <- eta[1:J]
  mu <- eta[J+1]
  tau <- exp(eta[J+2])
  sum(vapply(1:J, \(j) sum(dnorm(x[[j]], mean=theta[j], sd=sigma_true, log=TRUE)), 0.0)) +
    sum(dnorm(theta, mean=mu, sd=tau, log=TRUE))
}

res <- metrop(post, rep(0, J+2), 20000, scale=0.2)
burn <- 2000
mu_met <- res$batch[(burn+1):nrow(res$batch), J+1]

lines(density(mu_met), col="black")


plot(density(mu_met), col="black")
lines(density(mu_inca), col="green")
abline(v=mu_true)
