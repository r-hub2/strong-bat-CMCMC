/*
 INCA sampler: global covariance across all particles.

 Single-population (P particles, no split) implementation using the default CUDA stream.
 The Metropolis accept/reject step is provided by a runtime-compiled CUDA plugin that
 exports:
   - metrop_1gpu_sm30_setup_D
   - metrop_1gpu_sm30_D
 */

#include "config.h"
#include "check.h"
#include "covariance.h"
#include "generate_mvn.h"
#include "savesamps.h"
#include "metrop.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <time.h>

typedef void (*metrop_setup_fn)(int, int, float*, float*, const float*, const int*);
typedef void (*metrop_step_fn)(int, int, const float*, const float*, float*, float*, const float*, const int*, const float*, int*);

static int sum_ints(const int* xs, int n) {
  int s = 0;
  for(int i = 0; i < n; i++) s += xs[i];
  return s;
}

// Returns 0 on success, or the cuSOLVER `info` value on failure (>0).
static int info_chol_upper_inplace(cusolverDnHandle_t solver,
                                   float* A_d, int n,
                                   float* work_d, int lwork,
                                   int* info_d) {
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSetStream(solver, 0) );

  // Symmetrize and (if needed) add a tiny diagonal jitter to avoid failing on
  // near-singular covariances due to single-precision roundoff.
  const size_t bytes = (size_t)n * (size_t)n * sizeof(float);
  float* h = (float*)malloc(bytes);
  CHECK_NONNULL(h);

  if(cudaSuccess != cudaMemcpy(h, A_d, bytes, cudaMemcpyDeviceToHost)) {
    EPRINT("Error: failed to copy covariance to host for Cholesky\n");
    free(h);
    return 1;
  }

  for(int j = 0; j < n; j++) {
    for(int i = 0; i < j; i++) {
      h[i + j*n] = h[j + i*n];
    }
  }

  int info_h = -1;
  double jitter_scale = 1e-7;
  for(int attempt = 0; attempt < 6; attempt++) {
    if(cudaSuccess != cudaMemcpy(A_d, h, bytes, cudaMemcpyHostToDevice)) {
      EPRINT("Error: failed to copy covariance back to device for Cholesky\n");
      free(h);
      return 1;
    }

    CHECK_TRUE( cudaSuccess == cudaMemset(info_d, 0, sizeof(int)) );
    CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSpotrf(
        solver, CUBLAS_FILL_MODE_LOWER, n, A_d, n, work_d, lwork, info_d) );
    CHECK_TRUE( cudaSuccess == cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_TRUE( cudaSuccess == cudaDeviceSynchronize() );
    if(info_h == 0) {
      free(h);
      return 0;
    }

    int nonfinite = 0;
    double mean_diag = 0.0;
    double min_diag = 1e300;
    double max_diag = -1e300;
    for(int i = 0; i < n; i++) {
      const double d = (double)h[i + i*n];
      if(!(d == d) || d == (1.0/0.0) || d == -(1.0/0.0)) nonfinite++;
      mean_diag += d;
      if(d < min_diag) min_diag = d;
      if(d > max_diag) max_diag = d;
    }
    mean_diag /= (double)n;
    double base = (mean_diag < 0.0) ? -mean_diag : mean_diag;
    if(base < 1.0) base = 1.0;
    const double jitter = base * jitter_scale;
    jitter_scale *= 10.0;
    for(int i = 0; i < n; i++) h[i + i*n] = (float)((double)h[i + i*n] + jitter);

    if(attempt == 5) {
      EPRINT("Error: Cholesky failed (info=%d); diag[min,max]=[%.6g,%.6g] nonfinite_diag=%d\n",
             info_h, min_diag, max_diag, nonfinite);
      free(h);
      return info_h;
    }
  }

  free(h);
  return info_h;
}

int inca(const char* restrict CUDAlib, const metropmethod_t metropmethod,
         const float* restrict theta_init,
         double* restrict samps, const int p, const int it, const int cov_it, const int burnin, const int save_last, const int dim,
         const float scaleCov,
         const int64_t seed, const int device,
         const float* restrict Xf, const int lenXf, const int* restrict Xi, const int lenXi) {
  RANDINIT

  if(metropmethod != METROP_1GPU_SM30) {
    EPRINT("Error: only METROP_1GPU_SM30 is supported\n");
    return 1;
  }
  if(it>0 && save_last < 0) {
    EPRINT("Error: 'Invalid number of iterations to save from end (%d)'\n", save_last);
    return 1;
  }
  if(it<1) {
    EPRINT("Error: 'Invalid number of iterations (%d)'\n", it);
    return 1;
  }
  if(p < 2) {
    EPRINT("Error: 'Invalid number of particles (%d)'\n", p);
    return 1;
  }
  if(cov_it<1) {
    EPRINT("Error: 'Invalid covariance update period (%d)'\n", cov_it);
    return 1;
  }
  if(burnin < 0) {
    EPRINT("Error: 'Invalid burn-in length (%d)'\n", burnin);
    return 1;
  }
  if(scaleCov<=0) {
    EPRINT("Error: 'Invalid covariance scaling factor (%f)'\n", scaleCov);
    return 1;
  }

  const int saveit = (save_last == 0) ? it : ((save_last > it) ? it : save_last);
  const int first_label = (save_last == 0) ? 1 : (it - saveit + 1);

  CHECK_TRUE( cudaSuccess == cudaSetDevice(device) );

  void* lib = dlopen(CUDAlib, RTLD_LAZY|RTLD_LOCAL);
  if(!lib) {
    EPRINT("Error: '%s'\n", dlerror());
    return 1;
  }

  metrop_setup_fn setup = NULL;
  metrop_step_fn step = NULL;
  *(void **) (&setup) = dlsym(lib, "metrop_1gpu_sm30_setup_D");
  *(void **) (&step)  = dlsym(lib, "metrop_1gpu_sm30_D");
  if(!setup || !step) {
    EPRINT("Error: missing required entrypoints in CUDA plugin: %s\n", dlerror());
    EPRINT("Expected symbols: metrop_1gpu_sm30_setup_D and metrop_1gpu_sm30_D\n");
    dlclose(lib);
    return 1;
  }

  cublasHandle_t cublas;
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasCreate(&cublas) );

  curandGenerator_t rng;
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT) );
  if(seed == (int64_t) 0) {
    int64_t s = (int64_t) time(NULL);
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(rng, (unsigned long long) s) );
  } else {
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(rng, (unsigned long long) seed) );
  }

  cusolverDnHandle_t solver;
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnCreate(&solver) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSetStream(solver, 0) );

  float *theta_all = (float*) ALLOC((size_t)p*(size_t)dim, sizeof(float));
  float *onesV = (float*) ALLOC((size_t)p, sizeof(float));
  int *acc_h = (int*) ALLOC((size_t)p, sizeof(int));
  for(int i = 0; i < p; i++) onesV[i] = 1.0f;

  // INCA running (cumulative) covariance state on host
  double *sum_x = (double*) ALLOC((size_t)dim, sizeof(double));
  double *sum_xx = (double*) ALLOC((size_t)dim * (size_t)dim, sizeof(double));
  double *mean_all = (double*) ALLOC((size_t)dim, sizeof(double));
  double *cov_all = (double*) ALLOC((size_t)dim * (size_t)dim, sizeof(double));
  float *mean_b = (float*) ALLOC((size_t)dim, sizeof(float));
  float *cov_b = (float*) ALLOC((size_t)dim * (size_t)dim, sizeof(float));
  float *cov_scaled = (float*) ALLOC((size_t)dim * (size_t)dim, sizeof(float));
  long long n_all = 0;
  for(int i = 0; i < dim; i++) sum_x[i] = 0.0;
  for(int i = 0; i < dim * dim; i++) sum_xx[i] = 0.0;

  float *Xf_d = NULL;
  int *Xi_d = NULL;
  if(lenXf > 0) CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&Xf_d, (size_t)lenXf*sizeof(float)) );
  if(lenXi > 0) CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&Xi_d, (size_t)lenXi*sizeof(int)) );

  float *theta_cur_d = NULL, *theta_nxt_d = NULL;
  float *logdens_cur_d = NULL, *logdens_nxt_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&theta_cur_d, (size_t)p*(size_t)dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&theta_nxt_d, (size_t)p*(size_t)dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&logdens_cur_d, (size_t)p*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&logdens_nxt_d, (size_t)p*sizeof(float)) );

  float *unif_d = NULL;
  int *acc_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&unif_d, (size_t)p*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&acc_d, (size_t)p*sizeof(int)) );

  // Batch covariance + mean (used to update running sums)
  float *cov_batch_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&cov_batch_d, (size_t)dim*(size_t)dim*sizeof(float)) );

  float *cov_d = NULL;
  float *ws_d = NULL;
  float *onesV_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&cov_d, (size_t)dim*(size_t)dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&ws_d, (size_t)dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&onesV_d, (size_t)p*sizeof(float)) );

  int lwork = 0;
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSpotrf_bufferSize(
      solver, CUBLAS_FILL_MODE_LOWER, dim, cov_d, dim, &lwork) );
  CHECK_TRUE( lwork > 0 );
  float *work_d = NULL;
  int *info_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&work_d, (size_t)lwork*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&info_d, sizeof(int)) );

  if(lenXf > 0) CHECK_TRUE( cudaSuccess == cudaMemcpy(Xf_d, Xf, (size_t)lenXf*sizeof(float), cudaMemcpyHostToDevice) );
  if(lenXi > 0) CHECK_TRUE( cudaSuccess == cudaMemcpy(Xi_d, Xi, (size_t)lenXi*sizeof(int), cudaMemcpyHostToDevice) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(onesV_d, onesV, (size_t)p*sizeof(float), cudaMemcpyHostToDevice) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_cur_d, theta_init, (size_t)p*(size_t)dim*sizeof(float), cudaMemcpyHostToDevice) );

  const int threads = 256;
  const int blocks = (p + threads - 1) / threads;

  setup(threads, blocks, theta_cur_d, logdens_cur_d, Xf_d, Xi_d);
  CHECK_TRUE( cudaSuccess == cudaDeviceSynchronize() );

  // Initialise proposal covariance as scaleCov * I (as in CUDA/src/inca.cu).
  for(int i = 0; i < dim * dim; i++) cov_scaled[i] = 0.0f;
  for(int i = 0; i < dim; i++) cov_scaled[i + i*dim] = scaleCov;
  CHECK_TRUE( cudaSuccess == cudaMemcpy(cov_d, cov_scaled, (size_t)dim*(size_t)dim*sizeof(float), cudaMemcpyHostToDevice) );
  {
    const int info0 = info_chol_upper_inplace(solver, cov_d, dim, work_d, lwork, info_d);
    if(info0 != 0) {
      EPRINT("Error: Cholesky failed (info=%d) at init\n", info0);
      return 1;
    }
  }

  if(save_last == 0 || 1 >= first_label) {
    CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_all, theta_cur_d, (size_t)p*(size_t)dim*sizeof(float), cudaMemcpyDeviceToHost) );
    const size_t slot0 = (size_t)(1 - first_label);
    savesamps_all(samps, slot0, (size_t)saveit, 1, theta_all, (size_t)p, (size_t)dim);
  }

  // Update running sums using the initial population at iteration 1.
  CHECK_ZERO( cov_gpu_mean_cov(&cublas, theta_cur_d, cov_batch_d,
                              (size_t)p, (size_t)dim,
                              onesV_d, ws_d, 1.0f) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(mean_b, ws_d, (size_t)dim*sizeof(float), cudaMemcpyDeviceToHost) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(cov_b, cov_batch_d, (size_t)dim*(size_t)dim*sizeof(float), cudaMemcpyDeviceToHost) );
  for(int j = 0; j < dim; j++) {
    for(int i = j + 1; i < dim; i++) {
      cov_b[i + j*dim] = cov_b[j + i*dim];
    }
  }
  for(int i = 0; i < dim; i++) {
    double mu_i = (double)mean_b[i];
    sum_x[i] += mu_i * (double)p;
  }
  for(int j = 0; j < dim; j++) {
    for(int i = 0; i < dim; i++) {
      double mu_i = (double)mean_b[i];
      double mu_j = (double)mean_b[j];
      double cov_ij = (double)cov_b[i + j*dim];
      sum_xx[i + j*dim] += (double)p * (cov_ij + mu_i * mu_j);
    }
  }
  n_all += (long long)p;

  for(int itn = 1; itn < it; itn++) {
    CHECK_ZERO( generateMVN_gpu(&rng, &cublas, theta_nxt_d, (size_t)p, cov_d, (size_t)dim) );
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandGenerateUniform(rng, unif_d, (size_t)p) );
    CHECK_TRUE( cudaSuccess == cudaMemset(acc_d, 0, (size_t)p*sizeof(int)) );

    step(threads, blocks,
         theta_cur_d, logdens_cur_d,
         theta_nxt_d, logdens_nxt_d,
         Xf_d, Xi_d,
         unif_d, acc_d);
    CHECK_TRUE( cudaSuccess == cudaDeviceSynchronize() );

    CHECK_TRUE( cudaSuccess == cudaMemcpy(acc_h, acc_d, (size_t)p*sizeof(int), cudaMemcpyDeviceToHost) );
    int totacc = sum_ints(acc_h, p);
    float ar = ((float)totacc)/((float)p);
    PRINT("\rExecuting iteration %d of %d (acceptance rate: %f)", itn+1, it, ar);
    if(itn == it-1) PRINT("\n");

    float *tmp;
    tmp = theta_cur_d; theta_cur_d = theta_nxt_d; theta_nxt_d = tmp;
    tmp = logdens_cur_d; logdens_cur_d = logdens_nxt_d; logdens_nxt_d = tmp;

    // Per-iteration batch mean/cov, used to update the cumulative INCA covariance.
    CHECK_ZERO( cov_gpu_mean_cov(&cublas, theta_cur_d, cov_batch_d,
                                (size_t)p, (size_t)dim,
                                onesV_d, ws_d, 1.0f) );
    CHECK_TRUE( cudaSuccess == cudaMemcpy(mean_b, ws_d, (size_t)dim*sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_TRUE( cudaSuccess == cudaMemcpy(cov_b, cov_batch_d, (size_t)dim*(size_t)dim*sizeof(float), cudaMemcpyDeviceToHost) );

    // Fill lower triangle on host (cuBLAS fills only upper in SYRK).
    for(int j = 0; j < dim; j++) {
      for(int i = j + 1; i < dim; i++) {
        cov_b[i + j*dim] = cov_b[j + i*dim];
      }
    }

    // Update running sums (treat each iteration as a "batch" of size p).
    for(int i = 0; i < dim; i++) {
      double mu_i = (double)mean_b[i];
      sum_x[i] += mu_i * (double)p;
    }
    for(int j = 0; j < dim; j++) {
      for(int i = 0; i < dim; i++) {
        double mu_i = (double)mean_b[i];
        double mu_j = (double)mean_b[j];
        double cov_ij = (double)cov_b[i + j*dim];
        sum_xx[i + j*dim] += (double)p * (cov_ij + mu_i * mu_j);
      }
    }
    n_all += (long long)p;

    // Refresh proposal covariance:
    // - Do not use the learned covariance during burn-in (first `burnin` iterations).
    // - Still accumulate running sums from iteration 1 onwards.
    // - Apply the first learned covariance immediately after burn-in ends (at itn == burnin),
    //   then continue refreshing every cov_it iterations thereafter.
    int do_refresh = 0;
    if(itn < it - 1) {
      if(burnin > 0) {
        if(itn == burnin) do_refresh = 1;
        else if(itn > burnin && (cov_it > 0) && ((itn % cov_it) == 0)) do_refresh = 1;
      } else {
        if((cov_it > 0) && ((itn % cov_it) == 0)) do_refresh = 1;
      }
    }

    if(do_refresh) {
      for(int i = 0; i < dim; i++) mean_all[i] = sum_x[i] / (double)n_all;
      for(int j = 0; j < dim; j++) {
        for(int i = 0; i < dim; i++) {
          cov_all[i + j*dim] = (sum_xx[i + j*dim] / (double)n_all) - mean_all[i] * mean_all[j];
        }
      }

      // Scale cumulative covariance and compute its Cholesky for the MVN proposal.
      for(int j = 0; j < dim; j++) {
        for(int i = 0; i < dim; i++) {
          cov_scaled[i + j*dim] = (float)(scaleCov * cov_all[i + j*dim]);
        }
      }
      CHECK_TRUE( cudaSuccess == cudaMemcpy(cov_d, cov_scaled, (size_t)dim*(size_t)dim*sizeof(float), cudaMemcpyHostToDevice) );
      const int info1 = info_chol_upper_inplace(solver, cov_d, dim, work_d, lwork, info_d);
      if(info1 != 0) {
        float min_diag = cov_scaled[0];
        float max_diag = cov_scaled[0];
        for(int i = 0; i < dim; i++) {
          const float v = cov_scaled[i + i*dim];
          if(v < min_diag) min_diag = v;
          if(v > max_diag) max_diag = v;
        }
        EPRINT("\n[inca] chol failed at iter %d (info=%d). diag min=%g max=%g\n",
               itn, info1, min_diag, max_diag);
        return 1;
      }
    }

    const int iter_label = itn + 1;
    if(save_last == 0 || iter_label >= first_label) {
      const size_t slot = (size_t)(iter_label - first_label);
      CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_all, theta_cur_d, (size_t)p*(size_t)dim*sizeof(float), cudaMemcpyDeviceToHost) );
      savesamps_all(samps, slot, (size_t)saveit, (size_t)iter_label, theta_all, (size_t)p, (size_t)dim);
    }
  }

  if(lenXf > 0) CHECK_TRUE( cudaSuccess == cudaFree(Xf_d) );
  if(lenXi > 0) CHECK_TRUE( cudaSuccess == cudaFree(Xi_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(theta_cur_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(theta_nxt_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(logdens_cur_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(logdens_nxt_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(unif_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(acc_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(cov_batch_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(cov_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(ws_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(onesV_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(work_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(info_d) );

  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandDestroyGenerator(rng) );
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasDestroy(cublas) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnDestroy(solver) );

  dlclose(lib);

  FREE(theta_all);
  FREE(onesV);
  FREE(acc_h);
  FREE(sum_x);
  FREE(sum_xx);
  FREE(mean_all);
  FREE(cov_all);
  FREE(mean_b);
  FREE(cov_b);
  FREE(cov_scaled);

  RANDFREE;
  return 0;
}
