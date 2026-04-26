#' Install a user-provided CUDA target kernel into the package
#'
#' Copies a user `.cu` file into the installed package `kernels/` directory so it
#' can be compiled and used via `cmcmc(..., GPUkernel=...)` or
#' `inca(..., GPUkernel=...)`.
#'
#' The kernel must define:
#' `__device__ float logdens(const float *theta, const float *Xf, const int *Xi);`
#'
#' The sampler calls `logdens()` for each chain state and expects the return value
#' to be an \emph{unnormalized log target density} for the current `theta`.
#'
#' \strong{Kernel contract}
#' \itemize{
#'   \item `theta` is a contiguous vector of length `DIM` (compile-time macro).
#'   \item `Xf` is the numeric vector passed from R through `X`, converted to
#'   `float*` on the CUDA side.
#'   \item `Xi` is the integer vector passed from R through `Xi`, converted to
#'   `int*` on the CUDA side.
#'   \item Return a finite log density when `theta` is valid, and `-INFINITY`
#'   for out-of-support/invalid states.
#'   \item Do not return `NaN`; this can break Metropolis acceptance logic.
#'   \item The function should be deterministic for fixed inputs (no RNG inside
#'   `logdens`).
#' }
#'
#' \strong{Compile-time macros available in your kernel}
#' \itemize{
#'   \item `DIM`: parameter dimension (`ncol(init)` at runtime).
#'   \item `P`: population size (`p` argument).
#' }
#'
#' \strong{Simple kernel example}
#'
#' This example defines the standard multivariate normal target
#' \eqn{\pi(\theta)\propto\exp\{-\frac{1}{2}\sum_{j=1}^{DIM}\theta_j^2\}},
#' i.e., \eqn{\theta \sim N(0, I_{DIM})}. The returned value
#' `-0.5 * sum(theta^2)` is the log density up to an additive constant; the
#' normalizing term \eqn{-\frac{DIM}{2}\log(2\pi)} is intentionally omitted,
#' which is valid for Metropolis--Hastings because constants cancel in
#' acceptance ratios. In this minimal example, `Xf` and `Xi` are unused, so you
#' can call the sampler with `X = numeric()` and `Xi = integer()`.
#'
#' \preformatted{
#' #include <math.h>
#'
#' __device__ float logdens(const float *theta, const float *Xf, const int *Xi) {
#'   (void)Xf;  // unused in this minimal example
#'   (void)Xi;  // unused in this minimal example
#'
#'   // Standard normal target on R^DIM (up to additive constant)
#'   float q = 0.0f;
#'   for (int j = 0; j < DIM; ++j) q += theta[j] * theta[j];
#'   return -0.5f * q;
#' }
#' }
#'
#' @param cu_path path to a `.cu` file defining `__device__ float logdens(...)`.
#' @param name optional destination filename (defaults to basename of `cu_path`).
#'
#' @return The kernel name you can pass as `GPUkernel` (the filename).
#'
#' @examples
#' \dontrun{
#' cu <- tempfile(fileext = ".cu")
#' writeLines(c(
#'   "#include <math.h>",
#'   "__device__ float logdens(const float *theta, const float *Xf, const int *Xi) {",
#'   "  (void)Xf; (void)Xi;",
#'   "  float q = 0.0f;",
#'   "  for (int j = 0; j < DIM; ++j) q += theta[j] * theta[j];",
#'   "  return -0.5f * q;",
#'   "}"
#' ), cu)
#'
#' kname <- load_target_kernel(cu, name = "DiagNormal.cu")
#' # Then use: cmcmc(..., GPUkernel = kname, X = numeric(), Xi = integer())
#' # or:      inca(...,  GPUkernel = kname, X = numeric(), Xi = integer())
#' }
#' @export
load_target_kernel <- function(cu_path, name = basename(cu_path)) {
  if (!is.character(cu_path) || length(cu_path) != 1L || is.na(cu_path) || !nzchar(cu_path)) {
    stop("cu_path must be a single non-empty string.")
  }
  if (!file.exists(cu_path)) {
    stop("File not found: ", cu_path)
  }
  if (!is.character(name) || length(name) != 1L || is.na(name) || !nzchar(name)) {
    stop("name must be a single non-empty string.")
  }
  if (!grepl("\\.cu$", name)) {
    stop("name must end with '.cu' (got: ", name, ")")
  }

  kernels_dir <- file.path(path.package("CMCMC"), "kernels")
  if (!dir.exists(kernels_dir)) {
    stop("Package kernels directory not found: ", kernels_dir)
  }

  dest <- file.path(kernels_dir, name)
  ok <- file.copy(cu_path, dest, overwrite = TRUE)
  if (!isTRUE(ok)) {
    stop("Failed to copy kernel to: ", dest)
  }

  name
}
