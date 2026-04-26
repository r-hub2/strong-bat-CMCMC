compilekernel <- function(metrop.method, GPUkernel, dim, p, cflags) {
  kernel_base <- sub("\\.cu$", "", GPUkernel)
  kernel_dir <- paste(path.package("CMCMC"), "/kernels/", sep = "")
  cu_path <- paste(kernel_dir, kernel_base, ".cu", sep = "")

  metrop_base <- tools::file_path_sans_ext(basename(metrop.method))
  # IMPORTANT: both `DIM` and `P` are compile-time constants used by the sampler and
  # target kernel. Include them in the output filename so changing `dim`/`p` forces
  # recompilation instead of reusing an incompatible .so.
  so_path <- paste(kernel_dir, kernel_base, "__", metrop_base,
                   "__d", as.integer(dim), "__p", as.integer(p), ".so", sep = "")

  if (!file.exists(cu_path)) {
    stop("The GPU kernel '", GPUkernel, "' is not recognised.")
  }
  if (file.exists(so_path)) {
    cu_mtime <- file.info(cu_path)$mtime
    so_mtime <- file.info(so_path)$mtime
    metrop_mtime <- file.info(metrop.method)$mtime
    if (!is.na(cu_mtime) && !is.na(so_mtime) && !is.na(metrop_mtime) &&
        so_mtime >= cu_mtime && so_mtime >= metrop_mtime) {
      return(so_path)
    }
    file.remove(so_path)
  }

  cmcmc_path <- get0("PATH", cmcmcopts, ifnotfound = "")
  cmcmc_dyld <- get0("DYLD_LIBRARY_PATH", cmcmcopts, ifnotfound = "")
  verbose_compile <- isTRUE(get0("verbose_compile", cmcmcopts, ifnotfound = FALSE))

  `%||%` <- function(a, b) if (is.null(a)) b else a

  prepend_env <- function(new, old) {
    new <- trimws(as.character(new %||% ""))
    old <- trimws(as.character(old %||% ""))
    if (!nzchar(new)) return(old)
    if (!nzchar(old)) return(new)
    paste(new, old, sep = .Platform$path.sep)
  }

  path_extra <- cmcmc_path
  if (.Platform$OS.type == "windows" && nzchar(cmcmc_dyld)) {
    path_extra <- paste(c(cmcmc_path, cmcmc_dyld)[nzchar(c(cmcmc_path, cmcmc_dyld))],
                        collapse = .Platform$path.sep)
  }

  env <- c(
    PATH = prepend_env(path_extra, Sys.getenv("PATH", unset = "")),
    DYLD_LIBRARY_PATH = prepend_env(cmcmc_dyld, Sys.getenv("DYLD_LIBRARY_PATH", unset = "")),
    LD_LIBRARY_PATH = prepend_env(cmcmc_dyld, Sys.getenv("LD_LIBRARY_PATH", unset = ""))
  )
  env_kv <- paste0(names(env), "=", unname(env))

  args <- c(
    metrop.method,
    cu_path,
    "-o", so_path,
    "-rdc=true",
    "--shared",
    "--use_fast_math",
    "--compiler-options", "-fPIC"
  )
  if (verbose_compile) {
    args <- c(args, "--ptxas-options=-v")
  }
  args <- c(args, paste0("-DDIM=", as.integer(dim)), paste0("-DP=", as.integer(p)))

  cflags <- trimws(as.character(cflags %||% ""))
  if (nzchar(cflags)) {
    args <- c(args, strsplit(cflags, "\\s+")[[1]])
  }

  cat(
    sprintf(
      "Compiling CUDA plugin: GPUkernel=%s, engine=%s, DIM=%d, P=%d\n  output: %s\n",
      kernel_base, metrop_base, as.integer(dim), as.integer(p), so_path
    )
  )
  if (verbose_compile) {
    cat("  command: ", paste("nvcc", paste(args, collapse = " ")), "\n", sep = "")
  }

  nvcc_names <- if (.Platform$OS.type == "windows") c("nvcc.exe", "nvcc") else "nvcc"
  nvcc_candidates <- unname(Sys.which(nvcc_names))
  if (nzchar(cmcmc_path)) {
    nvcc_candidates <- c(nvcc_candidates, file.path(cmcmc_path, nvcc_names))
  }
  nvcc_candidates <- unique(nvcc_candidates[nzchar(nvcc_candidates)])
  nvcc_bin <- nvcc_candidates[file.exists(nvcc_candidates)][1]
  if (!length(nvcc_bin) || !nzchar(nvcc_bin)) {
    stop(
      "Could not find `nvcc`. ",
      "Install the CUDA toolkit and/or set `CMCMC::cmcmcopt(PATH=...)` to the CUDA bin directory. ",
      "Tried: ", paste(nvcc_candidates, collapse = ", ")
    )
  }
  if (file.access(nvcc_bin, 1) != 0) {
    stop("`nvcc` exists but is not executable: ", nvcc_bin)
  }

  t0 <- proc.time()[["elapsed"]]
  out <- tryCatch(
    system2(nvcc_bin, args = args, env = env_kv, stdout = TRUE, stderr = TRUE),
    error = function(e) {
      stop(
        "Failed to run `nvcc` (", nvcc_bin, ").\n",
        "PATH=", env[["PATH"]], "\n",
        "LD_LIBRARY_PATH=", env[["LD_LIBRARY_PATH"]], "\n",
        conditionMessage(e)
      )
    }
  )
  t1 <- proc.time()[["elapsed"]]
  status <- attr(out, "status")
  if (is.null(status)) status <- 0L
  if (status != 0L) {
    cat(paste(out, collapse = "\n"), "\n")
    stop("Kernel compilation failed for '", GPUkernel, "'.")
  }
  if (verbose_compile && length(out)) {
    cat(paste(out, collapse = "\n"), "\n")
  }
  cat(sprintf("Compilation finished in %.2fs (cached).\n", t1 - t0))
  so_path
}
