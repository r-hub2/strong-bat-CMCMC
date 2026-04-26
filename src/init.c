#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP cmcmcR(SEXP _CUDAlib, SEXP _metropmethod, SEXP _theta_init,
            SEXP _Xf, SEXP _Xi, SEXP _p, SEXP _it, SEXP _cov_it, SEXP _saved_iterations,
            SEXP _dim, SEXP _seed,
            SEXP _device);

SEXP incaR(SEXP _CUDAlib, SEXP _metropmethod, SEXP _theta_init,
           SEXP _Xf, SEXP _Xi, SEXP _p, SEXP _it, SEXP _cov_it, SEXP _saved_iterations,
           SEXP _burnin, SEXP _dim, SEXP _seed,
           SEXP _device);

static const R_CallMethodDef CallEntries[] = {
  {"cmcmcR", (DL_FUNC) &cmcmcR, 12},
  {"incaR", (DL_FUNC) &incaR, 13},
  {NULL, NULL, 0}
};

void R_init_CMCMC(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
