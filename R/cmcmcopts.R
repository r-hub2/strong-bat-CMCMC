#
# Based on the original CMCMC implementation by Louis Aslett, April 2018.
# Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
#

#' Set options related to the package
#'
#' @param PATH the path to the CUDA binary directory
#' @param DYLD_LIBRARY_PATH the path to the CUDA library directory. On Windows,
#'   this directory is also prepended to \code{PATH} while compiling kernels.
#' @param verbose_compile logical; if \code{TRUE}, prints the \code{nvcc} command
#'   and compiler output when compiling GPU kernels at runtime.
#'
#' @details
#' CMCMC tries to detect CUDA paths automatically when the package is loaded. It
#' checks \code{CMCMC_CUDA_BIN}, \code{CMCMC_CUDA_LIB64}, CUDA environment
#' variables such as \code{CUDA_HOME} and \code{CUDA_PATH}, and common Linux,
#' macOS, and Windows install locations. Use \code{cmcmcopt()} only when you need
#' to override the detected paths.
#'
#' @export
cmcmcopt <- function(PATH = NA, DYLD_LIBRARY_PATH = NA, verbose_compile = NA) {
  if (!is.na(PATH)) {
    assign("PATH", PATH, cmcmcopts)
  }
  if (!is.na(DYLD_LIBRARY_PATH)) {
    assign("DYLD_LIBRARY_PATH", DYLD_LIBRARY_PATH, cmcmcopts)
  }
  if (!is.na(verbose_compile)) {
    assign("verbose_compile", isTRUE(verbose_compile), cmcmcopts)
  }
}

`%||%` <- function(a, b) {
  if (is.null(a) || !length(a)) return(b)
  if (length(a) == 1L && is.na(a)) return(b)
  a
}

.first_existing_dir <- function(paths) {
  paths <- unique(paths[!is.na(paths) & nzchar(paths)])
  if (!length(paths)) return("")
  paths <- normalizePath(paths, winslash = "/", mustWork = FALSE)
  paths[dir.exists(paths)][1] %||% ""
}

.first_cuda_bin_dir <- function(paths) {
  paths <- unique(paths[!is.na(paths) & nzchar(paths)])
  if (!length(paths)) return("")
  paths <- normalizePath(paths, winslash = "/", mustWork = FALSE)
  nvcc_names <- if (.Platform$OS.type == "windows") c("nvcc.exe", "nvcc") else "nvcc"
  for (path in paths) {
    if (dir.exists(path) && any(file.exists(file.path(path, nvcc_names)))) {
      return(path)
    }
  }
  ""
}

.cuda_env_roots <- function() {
  env <- Sys.getenv()
  cuda_path_vars <- names(env)[grepl("^CUDA_PATH", names(env))]
  roots <- unique(c(
    Sys.getenv("CUDA_HOME", unset = ""),
    Sys.getenv("CUDA_ROOT", unset = ""),
    Sys.getenv("CUDA_PATH", unset = ""),
    unname(env[cuda_path_vars])
  ))
  roots <- normalizePath(roots[!is.na(roots) & nzchar(roots)], winslash = "/", mustWork = FALSE)
  roots[roots != "/"]
}

.detect_cuda_paths <- function() {
  is_windows <- .Platform$OS.type == "windows"
  cuda_bin <- Sys.getenv("CMCMC_CUDA_BIN", unset = "")
  cuda_lib <- Sys.getenv("CMCMC_CUDA_LIB64", unset = "")

  if (is_windows) {
    program_files <- unique(c(
      Sys.getenv("ProgramW6432", unset = ""),
      Sys.getenv("ProgramFiles", unset = ""),
      "C:/Program Files"
    ))
    cuda_bases <- file.path(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
    cuda_roots <- unlist(lapply(cuda_bases, function(base) {
      sort(Sys.glob(file.path(base, "v*")), decreasing = TRUE)
    }), use.names = FALSE)
    roots <- unique(c(.cuda_env_roots(), cuda_roots))

    bin_candidates <- c(cuda_bin, file.path(roots, "bin"))
    lib_candidates <- c(cuda_lib, file.path(roots, "lib", "x64"))
  } else {
    roots <- unique(c(
      .cuda_env_roots(),
      "/usr/local/cuda",
      sort(Sys.glob("/usr/local/cuda-*"), decreasing = TRUE),
      "/opt/cuda"
    ))

    bin_candidates <- c(cuda_bin, file.path(roots, "bin"))
    lib_candidates <- c(cuda_lib, file.path(roots, "lib64"), file.path(roots, "lib"))
  }

  list(
    bin = .first_cuda_bin_dir(bin_candidates),
    lib = .first_existing_dir(lib_candidates)
  )
}

.onLoad <- function(libname, pkgname) {
  assign("cmcmcopts", new.env(), envir = parent.env(environment()))
  cuda <- .detect_cuda_paths()
  assign("PATH", cuda$bin, cmcmcopts)
  assign("DYLD_LIBRARY_PATH", cuda$lib, cmcmcopts)
  assign("verbose_compile", FALSE, cmcmcopts)
}
