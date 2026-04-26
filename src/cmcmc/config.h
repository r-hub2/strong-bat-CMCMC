/*
 Minimal configuration for building the Louis CMCMC engine inside an R package.
 */
#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <stdlib.h>

#ifdef RUN_IN_R
#include <R.h>
#include <R_ext/Print.h>
#include <R_ext/Error.h>
#define PRINT Rprintf
#define EPRINT REprintf
#define error_return(msg) error("%s", msg)
#else
#define PRINT(...) fprintf(stdout, __VA_ARGS__)
#define EPRINT(...) fprintf(stderr, __VA_ARGS__)
#define error_return(msg) do { EPRINT("%s\n", msg); exit(1); } while (0)
#endif

#define ALLOC(n, sz) malloc((size_t)(n) * (size_t)(sz))
#define FREE(ptr) free(ptr)

// Louis' code expects this macro to exist (no-op here).
#define RANDINIT
#define RANDFREE

#endif
