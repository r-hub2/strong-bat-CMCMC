/*
 R bridge for INCA (global covariance) engine.
 */

#include <R.h>
#include <Rinternals.h>
#include <stdint.h>
#include "cmcmc/cmcmc.h"

SEXP incaR(SEXP _CUDAlib, SEXP _metropmethod, SEXP _theta_init,
           SEXP _Xf, SEXP _Xi, SEXP _p, SEXP _it, SEXP _cov_it, SEXP _saved_iterations,
           SEXP _burnin, SEXP _dim, SEXP _seed,
           SEXP _device) {
  int p = asInteger(_p);
  int it = asInteger(_it);
  int cov_it = asInteger(_cov_it);
  int saved_iterations = asInteger(_saved_iterations);
  int burnin = asInteger(_burnin);
  int save_last = 0;
  int saveit = it;
  if(saved_iterations < 0) {
    error_return("saved_iterations must be >= 0.");
  }
  if(saved_iterations == 0) {
    save_last = 0;   // save all
    saveit = it;
  } else {
    save_last = saved_iterations; // save last k
    if(save_last > it) save_last = it;
    saveit = save_last;
  }
  int dim = asInteger(_dim);
  int device = asInteger(_device);
  int metropmethod = asInteger(_metropmethod);
  int64_t seed = (int64_t) asInteger(_seed);
  const double* Xd = REAL(_Xf);
  const int* Xi = INTEGER(_Xi);
  const char* CUDAlib = CHAR(STRING_ELT(_CUDAlib,0));
  const double* theta_initd = REAL(_theta_init);

  if(nrows(_theta_init) != p) error_return("Incorrect number of rows in initial population.");
  if(ncols(_theta_init) != dim) error_return("Incorrect number of columns in initial population.");

  // Convert theta_init from double to float (p x dim, column-major).
  float *theta_init = (float*) R_alloc((size_t)p*(size_t)dim, sizeof(float));
  for(int j = 0; j < dim; j++) {
    for(int i = 0; i < p; i++) {
      theta_init[i + j*p] = (float) theta_initd[i + j*p];
    }
  }

  // Convert Xf from double to float
  float *Xf = (float*) R_alloc(length(_Xf), sizeof(float));
  for(int i = 0; i < length(_Xf); i++) {
    Xf[i] = (float) Xd[i];
  }

  // Allocate result store
  SEXP _samps = PROTECT(allocMatrix(REALSXP, saveit*p, 2+dim));
  double *samps = REAL(_samps);

  if( inca(CUDAlib, metropmethod, theta_init,
           samps, p, it, cov_it, burnin, save_last, dim,
           (float) ((2.38*2.38)/dim),
           seed, device,
           Xf, length(_Xf), Xi, length(_Xi)) ) {
    UNPROTECT(1);
    error_return("Sampler halted.")
  }

  UNPROTECT(1);
  return(_samps);
}
