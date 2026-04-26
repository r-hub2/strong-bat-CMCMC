# Based on the original CMCMC implementation by Louis Aslett, April 2018.
# Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
#

#' Run a contemporaneous MCMC sampler
#'
#' This runs a contemporaneous MCMC sampler using the GPU and returns a
#' specified number of iterations.
#'
#' @details
#' CMCMC runs a population of \eqn{K} parallel random-walk Metropolis (RWM) chains
#' on the GPU (in this interface, \eqn{K} is given by the argument \code{p}). Let
#' \eqn{\pi(\theta)\propto f(\theta)}{pi(theta) proportional to f(theta)} denote the target
#' density on \eqn{\mathbb{R}^D}{R^D}, evaluated by the compiled CUDA kernel
#' \code{GPUkernel}.
#'
#' \strong{Random-walk Metropolis.}
#' Given a current state \eqn{\theta}{theta}, RWM proposes
#' \deqn{\theta^{\star}=\theta+\xi,\qquad \xi\sim\mathcal{N}(0,\Sigma_{\mathrm{prop}}),}{
#' theta* = theta + xi,  xi ~ N(0, Sigma_prop),
#' }
#' and accepts with probability
#' \deqn{\alpha(\theta,\theta^{\star})=\min\left\{1,\frac{f(\theta^{\star})}{f(\theta)}\right\},}{
#' alpha(theta, theta*) = min(1, f(theta*) / f(theta)),
#' }
#' since the Gaussian random-walk proposal is symmetric.
#'
#' \strong{Progress output (acceptance rate).}
#' During sampling, the implementation prints a progress line of the form
#' \sQuote{Executing iteration t of T (acceptance rate: ...)}. The acceptance
#' rate displayed at iteration \eqn{t} is the fraction of the \eqn{K} proposals
#' accepted at that iteration,
#' \deqn{\mathrm{ar}_t=\frac{1}{K}\sum_{k=1}^{K}\mathbf{1}\{\text{particle $k$ accepted at iteration $t$}\}.}{
#' ar_t = (1/K) * sum_{k=1}^K I(accepted at iteration t).
#' }
#' This is an iteration-level statistic (not a cumulative acceptance rate). The
#' last printed value corresponds to \eqn{t=T}{t=T} and is computed from the
#' \eqn{K} accept/reject outcomes at the final iteration.
#'
#' \strong{Two-group contemporaneous covariance exchange.}
#' CMCMC splits the \eqn{K} particles into two equal groups \eqn{A} and \eqn{B}
#' (so \eqn{K} must be even). Let \eqn{\Theta_t^{A}}{Theta_t^A} and \eqn{\Theta_t^{B}}{Theta_t^B} denote
#' the \eqn{(K/2)\times D}{(K/2) x D} matrices whose rows are the group states at iteration
#' \eqn{t}. At covariance update times, we compute the (population) empirical
#' covariance within each group, for example
#' \deqn{
#' \bar{\theta}_t^{B}=\frac{2}{K}\sum_{k\in B}\theta_t^{(k)},\qquad
#' \widehat{\Sigma}_t^{B}=\frac{2}{K}\sum_{k\in B}\left(\theta_t^{(k)}-\bar{\theta}_t^{B}\right)\left(\theta_t^{(k)}-\bar{\theta}_t^{B}\right)^{\top}.}{
#' theta_bar_t^B = (2/K) * sum_{k in B} theta_t^(k),
#' Sigma_hat_t^B = (2/K) * sum_{k in B} (theta_t^(k) - theta_bar_t^B)(theta_t^(k) - theta_bar_t^B)^T.
#' }
#' (and analogously for \eqn{\widehat{\Sigma}_t^{A}}{Sigma_hat_t^A}).
#'
#' The key CMCMC interaction is that the proposal covariance for one group is
#' taken from the other group: particles in \eqn{A} are updated using
#' \eqn{\widehat{\Sigma}^{B}}{Sigma_hat^B} and particles in \eqn{B} are updated using
#' \eqn{\widehat{\Sigma}^{A}}{Sigma_hat^A}. Between update times, the last covariance factors
#' are reused. The update frequency is controlled by \code{covit}.
#'
#' \strong{Cholesky factorisation on the GPU.}
#' To generate Gaussian proposal increments efficiently, the implementation
#' forms a Cholesky factor \eqn{L} such that
#' \deqn{\Sigma_{\mathrm{prop}} = L L^{\top},}{Sigma_prop = L L^T,}
#' then draws \eqn{z\sim\mathcal{N}(0,I_D)}{z ~ N(0, I_D)} and sets
#' \eqn{\xi = z L^{\top}}{xi = z L^T}.
#' The Cholesky factorisation is computed with NVIDIA cuSOLVER
#' (\code{cusolverDnSpotrf}), standard-normal variates are generated with cuRAND,
#' and the triangular transform is applied with cuBLAS (\code{cublasStrmm}). If
#' the empirical covariance is numerically near-singular (single precision), a
#' small diagonal jitter is added internally so the Cholesky factorisation
#' succeeds.
#'
#' \strong{GPU kernels and data layout.}
#' The argument \code{GPUkernel} selects which CUDA target-density kernel to
#' compile and run. The kernel name must correspond to a \code{.cu} file in
#' \code{inst/kernels/} (for example, \code{"MVNorm"} or \code{"MVNorm.cu"}).
#' The auxiliary inputs \code{X} (numeric) and \code{Xi} (integer) are passed to
#' the device log-density function \code{logdens(theta, Xf, Xi)}; their expected
#' layout depends on \code{GPUkernel}. This package treats \code{X} as a flat
#' numeric vector (copied to the GPU as single-precision floats) and \code{Xi} as
#' a flat integer vector (copied as 32-bit integers).
#'
#' Currently documented kernels:
#' \itemize{
#' \item \code{"MVNorm"} (\code{inst/kernels/MVNorm.cu}): multivariate normal
#'   target with known mean and precision. Here \code{Xi} is not used and can be
#'   empty. \code{X} must be a numeric vector of length \eqn{D + D^2} containing
#'   \code{c(mu, SigmaInv)}, where \code{mu} is length \eqn{D} and \code{SigmaInv}
#'   is the \eqn{D\times D}{D x D} inverse covariance (precision) matrix flattened in
#'   column-major order (the default in R). For example:
#'   \preformatted{
#'   X  <- c(mu, as.numeric(solve(Sigma)))
#'   Xi <- integer(0)
#'   }
#'
#' \item \code{"Logistic"} (\code{inst/kernels/Logistic.cu}): logistic regression
#'   posterior with an independent N(0, var) prior on coefficients. Let \eqn{n}
#'   be the number of observations and \eqn{D} the coefficient dimension. Set
#'   \code{Xi <- as.integer(n)} (so that \code{Xi[1]} in R, i.e. \code{Xi[0]} on
#'   the device, is \eqn{n}). The numeric vector \code{X} must be
#'   \code{c(X, y, var)}, where \code{X} contains the \eqn{n\times D}{n x D} design
#'   matrix in \emph{row-major} order, \code{y} is length \eqn{n} with values in
#'   \{0,1\}, and \code{var} is a single prior variance value (if \code{var <= 0},
#'   the kernel defaults to 100). In R, a convenient way to form row-major
#'   \code{X} from an \code{n x D} matrix \code{Xmat} is \code{as.numeric(t(Xmat))}:
#'   \preformatted{
#'   Xi <- as.integer(n)
#'   X  <- c(as.numeric(t(Xmat)), as.numeric(y), as.numeric(var))
#'   }
#'
#' \item \code{"HierBetaBinomJoint"} (\code{inst/kernels/HierBetaBinomJoint.cu}):
#'   hierarchical Beta--Binomial (joint) posterior.
#'   This kernel uses an unconstrained parameterisation with
#'   \eqn{\theta = (\log\alpha,\log\beta,z_1,\dots,z_J)}{theta = (log alpha, log beta, z_1..z_J)}
#'   where \eqn{z_j=\mathrm{logit}(\omega_j)}{z_j = logit(omega_j)}.
#'   Here \code{X} is not used and can be empty. \code{Xi} must be the integer
#'   vector \code{c(J, n_1..n_J, y_1..y_J)}. The compile-time dimension must
#'   match: \code{D = 2 + J}. For example:
#'   \preformatted{
#'   J  <- length(y)
#'   X  <- numeric(0)
#'   Xi <- as.integer(c(J, n, y))
#'   }
#'
#' \item \code{"NneWayNormalNC"} (\code{inst/kernels/NneWayNormalNC.cu}): one-way
#'   Normal random-effects model in a non-centred parameterisation. The state is
#'   \eqn{\theta = (\mu,\log\tau,z_1,\dots,z_J)}{theta = (mu, log tau, z_1..z_J)}
#'   with \eqn{\tau>0}{tau>0} and random effects \eqn{\theta_j=\mu+\tau z_j}{theta_j = mu + tau z_j}.
#'   Data are passed via \code{Xi} and \code{X} as:
#'   \code{Xi <- c(J, n_1..n_J)} and \code{X <- c(sigma2, ybar_1..ybar_J)}, where
#'   \code{sigma2} is the within-group variance and \code{ybar_j} are the group means.
#'   Again \code{D = 2 + J} must match at compile time. For example:
#'   \preformatted{
#'   J  <- length(ybar)
#'   Xi <- as.integer(c(J, n))
#'   X  <- c(sigma2, as.numeric(ybar))
#'   }
#' }
#'
#' @param init the initial population of samples (matrix) or a single starting
#'   state, to which normal noise will be added to get an initial population (vector)
#' @param GPUkernel target density GPU kernel to evaluate (see \sQuote{GPU kernels
#'   and data layout} in Details).
#' @param it total number of MCMC iterations to run
#' @param covit number of iterations between covariance updates
#' @param p number of particles to use in the population.
#' @param X numeric vector passed to the GPU kernel as \code{Xf} (layout depends
#'   on \code{GPUkernel}).
#' @param Xi numeric/integer vector passed to the GPU kernel as \code{Xi} and
#'   coerced to integer type (layout depends on \code{GPUkernel}).
#' @param saved_iterations number of iterations to save: 0 saves all iterations;
#'   s>0 saves the last s iterations (i.e. the last s populations).
#' @param initial.sd if init is a vector, this is the standard deviation used
#'   to seed the initial population based on perturbing init.
#' @param cflags optional compiler flags to pass when compiling the GPU kernel.
#' @param seed random number generator seed passed through to CUDA.
#' @param device GPU device number
#'
#' @return
#' A numeric matrix with \code{2 + D} columns, where \code{D} is the parameter
#' dimension (\code{ncol(init)}). The first two columns are metadata:
#' \itemize{
#' \item column 1: \code{iter} (iteration index, 1-based; for \code{saved_iterations=k>0},
#'   this will typically run from \code{it-k+1} to \code{it}),
#' \item column 2: \code{particle} (particle/chain index within the population, 1..K where K=\code{p}),
#' \item columns 3..(2+D): the state vector \code{theta}.
#' }
#' Let \eqn{N}{N} denote the number of saved iterations. Then the number of rows
#' is \eqn{K\times N}{K * N}. If \code{saved_iterations = 0} (save all), then
#' \eqn{N = it}{N = it}. If \code{saved_iterations = s > 0}, then
#' \eqn{N = \min(s, it)}{N = min(s, it)}.
#'
#'
#' @examples
#' # Multivariate normal target (D = 10, equicorrelation rho = 0.9).
#' # Requires a working CUDA toolchain. CMCMC attempts to detect CUDA paths
#' # automatically when the package is loaded.
#' \dontrun{
#' library(CMCMC)
#'
#' D <- 10
#' rho <- 0.9
#' mu <- rep(0, D)
#' Sigma <- matrix(rho, nrow = D, ncol = D)
#' diag(Sigma) <- 1
#'
#' # MVNorm kernel expects X = c(mu, SigmaInv) with SigmaInv in column-major order.
#' X <- c(mu, as.numeric(solve(Sigma)))
#' Xi <- integer(0)
#'
#' # Run CMCMC with K particles and T iterations.
#' K <- 1024
#' T <- 2500
#' samples <- CMCMC::cmcmc(
#'   init = mu,
#'   GPUkernel = "MVNorm",
#'   it = T,
#'   covit = 5,
#'   p = K,
#'   X = X,
#'   Xi = Xi,
#'   saved_iterations = 1,
#'   seed = 123,
#'   device = 0
#' )
#'
#' # Posterior sample is the last population (K draws in R^D).
#' dim(samples)
#' colMeans(samples[, 3:(2 + D), drop = FALSE])
#' }
#'
#' @useDynLib CMCMC
#' @importFrom stats rnorm
#' @export
cmcmc <- function(init, GPUkernel, it, covit,
                  p = if(is.matrix(init)) nrow(init) else length(init)*log(length(init)),
                  X = c(), Xi = c(), saved_iterations = 1, initial.sd = 1, cflags = "", seed = 0,
                  device = 0) {
  if(!is.vector(X, "numeric") && !is.null(X)) {
    stop("X must be a numeric vector")
  }
  if(!is.vector(Xi, "numeric") && !is.null(Xi)) {
    stop("Xi must be a numeric vector")
  }
  if(!is.vector(Xi, "integer")) {
    warning("Xi being coerced to integer")
  }
  if(!is.numeric(p) || length(p)!=1) {
    stop("p must be a number specifying particle population size.")
  }
  if(!is.numeric(it) || length(it)!=1) {
    stop("it must be a number specifying number of MCMC iterations to perform.")
  }
  if(!is.numeric(saved_iterations) || length(saved_iterations) != 1 || is.na(saved_iterations)) {
    stop("saved_iterations must be a single non-missing number (0 = save all, k > 0 = save last k).")
  }
  if(saved_iterations < 0) {
    stop("saved_iterations must be >= 0.")
  }
  if(saved_iterations != floor(saved_iterations)) {
    stop("saved_iterations must be an integer (0 = save all, k > 0 = save last k).")
  }
  if(!is.matrix(init) && !is.vector(init, "numeric")) {
    stop("init must be a matrix or vector.")
  }
  if(is.matrix(init)) {
    if(nrow(init) != p) stop("Number of rows in init must equal size of population, or else be a single vector.")
  }
  if(is.vector(init)) {
    init <- matrix(rnorm(p*length(init), init, initial.sd), ncol = length(init), byrow = TRUE)
  }
  dim <- ncol(init)

  # Add this as an option once more methods available
  metrop.method <- paste(path.package("CMCMC"), "/metrop_1gpu_sm30.cu", sep = "")

  kernel <- compilekernel(metrop.method, GPUkernel, dim, p, cflags)

  .Call("cmcmcR", kernel, 1, init,
        X, suppressWarnings(as.integer(Xi)), p, it, covit, as.integer(saved_iterations),
        dim, seed,
        device, PACKAGE = "CMCMC")
}

# NOTES:
#   Require a starting position from which can infer problem dimension
# Dimension passed via compiler
# Number of particles passed via compiler
