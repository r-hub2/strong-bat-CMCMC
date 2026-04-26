/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
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
typedef void (*metrop_setup_stream_fn)(int, int, float*, float*, const float*, const int*, cudaStream_t);
typedef void (*metrop_step_stream_fn)(int, int, const float*, const float*, float*, float*, const float*, const int*, const float*, int*, cudaStream_t);

static int sum_ints(const int* xs, int n) {
  int s = 0;
  for(int i = 0; i < n; i++) s += xs[i];
  return s;
}

static int chol_upper_inplace(cusolverDnHandle_t solver,
                              cudaStream_t stream,
                              float* A_d, int n,
                              float* work_d, int lwork,
                              int* info_d) {
  // NOTE: cuBLAS covariance fills only the requested triangle. Some libraries/paths
  // may still read the other triangle, so we explicitly symmetrize to be safe and
  // to match the CUDA reference implementation which keeps the full matrix symmetric.
  const size_t bytes = (size_t)n * (size_t)n * sizeof(float);
  float* h = (float*)malloc(bytes);
  if(!h) {
    EPRINT("Error: failed to allocate host buffer for Cholesky (%zu bytes)\n", bytes);
    return(1);
  }

  if(cudaSuccess != cudaMemcpyAsync(h, A_d, bytes, cudaMemcpyDeviceToHost, stream) ||
     cudaSuccess != cudaStreamSynchronize(stream)) {
    EPRINT("Error: failed to copy covariance to host for symmetrization\n");
    free(h);
    return(1);
  }

  // Symmetrize: copy lower -> upper.
  for(int j = 0; j < n; j++) {
    for(int i = 0; i < j; i++) {
      h[i + j*n] = h[j + i*n];
    }
  }

  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSetStream(solver, stream) );

  int info_h = -1;
  double jitter_scale = 1e-7;
  for(int attempt = 0; attempt < 6; attempt++) {
    if(cudaSuccess != cudaMemcpyAsync(A_d, h, bytes, cudaMemcpyHostToDevice, stream) ||
       cudaSuccess != cudaStreamSynchronize(stream)) {
      EPRINT("Error: failed to copy covariance back to device before Cholesky\n");
      free(h);
      return(1);
    }

    CHECK_TRUE( cudaSuccess == cudaMemsetAsync(info_d, 0, sizeof(int), stream) );
    CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSpotrf(
        solver, CUBLAS_FILL_MODE_LOWER, n, A_d, n, work_d, lwork, info_d) );
    CHECK_TRUE( cudaSuccess == cudaMemcpyAsync(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost, stream) );
    CHECK_TRUE( cudaSuccess == cudaStreamSynchronize(stream) );
    if(info_h == 0) {
      free(h);
      return(0);
    }

    // Retry with a small diagonal jitter (helps if the covariance is near-singular
    // due to single-precision roundoff). Keep this minimal and increase geometrically.
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
      return(1);
    }
  }

  // Unreachable, but keep compiler happy.
  free(h);
  EPRINT("Error: Cholesky failed (info=%d)\n", info_h);
  return(1);
}

int cmcmc(const char* restrict CUDAlib, const metropmethod_t metropmethod,
          const float* restrict theta_init_p1, const float* restrict theta_init_p2,
          double* restrict samps, const int p, const int it, const int cov_it, const int save_last, const int dim,
          const float scaleCov,
          const int64_t seed, const int device,
          const float* restrict Xf, const int lenXf, const int* restrict Xi, const int lenXi) {
  RANDINIT

  if(metropmethod != METROP_1GPU_SM30) {
    EPRINT("Error: only METROP_1GPU_SM30 is supported\n");
    return(1);
  }
  if(it>0 && save_last < 0) {
    EPRINT("Error: 'Invalid number of iterations to save from end (%d)'\n", save_last);
    return(1);
  }
  if(it<1) {
    EPRINT("Error: 'Invalid number of iterations (%d)'\n", it);
    return(1);
  }
  if(p%2!=0) {
    EPRINT("Error: 'Invalid number of particles (%d) ... must be even'\n", p);
    return(1);
  }
  if(cov_it<1) {
    EPRINT("Error: 'Invalid covariance update period (%d)'\n", cov_it);
    return(1);
  }
  if(scaleCov<=0) {
    EPRINT("Error: 'Invalid covariance scaling factor (%f)'\n", scaleCov);
    return(1);
  }

  const int saveit = (save_last == 0) ? it : ((save_last > it) ? it : save_last);
  const int first_label = (save_last == 0) ? 1 : (it - saveit + 1);

  CHECK_TRUE( cudaSuccess == cudaSetDevice(device) );

  // Attach GPU library (runtime-compiled CUDA plugin)
  void* lib = dlopen(CUDAlib, RTLD_LAZY|RTLD_LOCAL);
  if(!lib) {
    EPRINT("Error: '%s'\n", dlerror());
    return(1);
  }

  metrop_setup_stream_fn setup_stream = NULL;
  metrop_step_stream_fn step_stream = NULL;

  *(void **) (&setup_stream) = dlsym(lib, "metrop_1gpu_sm30_setup_D_stream");
  *(void **) (&step_stream)  = dlsym(lib, "metrop_1gpu_sm30_D_stream");
  if(!setup_stream || !step_stream) {
    EPRINT("Error: missing required stream entrypoints in CUDA plugin: %s\n", dlerror());
    EPRINT("Expected symbols: metrop_1gpu_sm30_setup_D_stream and metrop_1gpu_sm30_D_stream\n");
    dlclose(lib);
    return(1);
  }

  // Streams + events (one per group)
  cudaStream_t streamA, streamB;
  CHECK_TRUE( cudaSuccess == cudaStreamCreate(&streamA) );
  CHECK_TRUE( cudaSuccess == cudaStreamCreate(&streamB) );
  cudaEvent_t eventA, eventB;
  CHECK_TRUE( cudaSuccess == cudaEventCreate(&eventA) );
  CHECK_TRUE( cudaSuccess == cudaEventCreate(&eventB) );

  // cuBLAS handles (one per stream)
  cublasHandle_t cublasA, cublasB;
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasCreate(&cublasA) );
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasCreate(&cublasB) );
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasSetStream(cublasA, streamA) );
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasSetStream(cublasB, streamB) );

  // cuRAND generators (one per stream)
  curandGenerator_t rngA, rngB;
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandCreateGenerator(&rngA, CURAND_RNG_PSEUDO_DEFAULT) );
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandCreateGenerator(&rngB, CURAND_RNG_PSEUDO_DEFAULT) );
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetStream(rngA, streamA) );
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetStream(rngB, streamB) );
  if(seed == (int64_t) 0) {
    int64_t s = (int64_t) time(NULL);
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(rngA, (unsigned long long) s) );
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(rngB, (unsigned long long) (s + 1)) );
  } else {
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(rngA, (unsigned long long) seed) );
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(rngB, (unsigned long long) (seed + 1)) );
  }

  // cuSOLVER handles + workspaces (one per stream)
  cusolverDnHandle_t solverA, solverB;
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnCreate(&solverA) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnCreate(&solverB) );

  int lworkA = 0, lworkB = 0;
  float *workA_d = NULL, *workB_d = NULL;
  int *infoA_d = NULL, *infoB_d = NULL;

  // Host-side storage for saving
  float* restrict theta_p1 = (float*) ALLOC((p/2) * dim, sizeof(float));
  float* restrict theta_p2 = (float*) ALLOC((p/2) * dim, sizeof(float));
  float* restrict onesV = (float*) ALLOC(p/2, sizeof(float));
  for(int i = 0; i < p/2; i++) onesV[i] = 1.0f;
  int* accA_h = (int*) ALLOC(p/2, sizeof(int));
  int* accB_h = (int*) ALLOC(p/2, sizeof(int));

  // Device X buffers
  float *Xf_d = NULL;
  if(lenXf > 0) CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&Xf_d, (size_t)lenXf*sizeof(float)) );
  int *Xi_d = NULL;
  if(lenXi > 0) CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&Xi_d, (size_t)lenXi*sizeof(int)) );

  // Populations + proposals
  float *theta_cur_p1_d = NULL, *theta_nxt_p1_d = NULL;
  float *theta_cur_p2_d = NULL, *theta_nxt_p2_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&theta_cur_p1_d, (size_t)(p/2)*dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&theta_nxt_p1_d, (size_t)(p/2)*dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&theta_cur_p2_d, (size_t)(p/2)*dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&theta_nxt_p2_d, (size_t)(p/2)*dim*sizeof(float)) );

  float *logdens_cur_p1_d = NULL, *logdens_nxt_p1_d = NULL;
  float *logdens_cur_p2_d = NULL, *logdens_nxt_p2_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&logdens_cur_p1_d, (size_t)(p/2)*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&logdens_nxt_p1_d, (size_t)(p/2)*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&logdens_cur_p2_d, (size_t)(p/2)*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&logdens_nxt_p2_d, (size_t)(p/2)*sizeof(float)) );

  // Uniforms + accept counters
  float *unifA_d = NULL, *unifB_d = NULL;
  int *accA_d = NULL, *accB_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&unifA_d, (size_t)(p/2)*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&unifB_d, (size_t)(p/2)*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&accA_d, (size_t)(p/2)*sizeof(int)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&accB_d, (size_t)(p/2)*sizeof(int)) );

  // Covariance buffers (one per group) + workspace
  float *covA_d = NULL, *covB_d = NULL;
  float *wsA_d = NULL, *wsB_d = NULL;
  float *onesV_d = NULL;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&covA_d, (size_t)dim*dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&covB_d, (size_t)dim*dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&wsA_d, (size_t)dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&wsB_d, (size_t)dim*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&onesV_d, (size_t)(p/2)*sizeof(float)) );

  // Copy inputs
  if(lenXf > 0) CHECK_TRUE( cudaSuccess == cudaMemcpy(Xf_d, Xf, (size_t)lenXf*sizeof(float), cudaMemcpyHostToDevice) );
  if(lenXi > 0) CHECK_TRUE( cudaSuccess == cudaMemcpy(Xi_d, Xi, (size_t)lenXi*sizeof(int), cudaMemcpyHostToDevice) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(onesV_d, onesV, (size_t)(p/2)*sizeof(float), cudaMemcpyHostToDevice) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_cur_p1_d, theta_init_p1, (size_t)(p/2)*dim*sizeof(float), cudaMemcpyHostToDevice) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_cur_p2_d, theta_init_p2, (size_t)(p/2)*dim*sizeof(float), cudaMemcpyHostToDevice) );

  // Kernel launch dims: must be exactly p/2 threads total (no bounds check in kernel)
  const int threads = GDLT256(p/2);
  const int blocks = (p/2)/threads;

  // Initial log densities
  setup_stream(threads, blocks, theta_cur_p1_d, logdens_cur_p1_d, Xf_d, Xi_d, streamA);
  setup_stream(threads, blocks, theta_cur_p2_d, logdens_cur_p2_d, Xf_d, Xi_d, streamB);
  CHECK_TRUE( cudaSuccess == cudaStreamSynchronize(streamA) );
  CHECK_TRUE( cudaSuccess == cudaStreamSynchronize(streamB) );

  // cuSOLVER workspaces (after buffers allocated)
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSetStream(solverA, streamA) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSetStream(solverB, streamB) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSpotrf_bufferSize(
      solverA, CUBLAS_FILL_MODE_LOWER, dim, covA_d, dim, &lworkA) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnSpotrf_bufferSize(
      solverB, CUBLAS_FILL_MODE_LOWER, dim, covB_d, dim, &lworkB) );
  CHECK_TRUE( lworkA > 0 && lworkB > 0 );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&workA_d, (size_t)lworkA*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&workB_d, (size_t)lworkB*sizeof(float)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&infoA_d, sizeof(int)) );
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&infoB_d, sizeof(int)) );

  // Initial covariance/cholesky for both groups (in parallel streams)
  CHECK_ZERO( cov_gpu(&cublasA, theta_cur_p1_d, covA_d, p/2, dim, onesV_d, wsA_d, scaleCov) );
  CHECK_ZERO( cov_gpu(&cublasB, theta_cur_p2_d, covB_d, p/2, dim, onesV_d, wsB_d, scaleCov) );
  CHECK_ZERO( chol_upper_inplace(solverA, streamA, covA_d, dim, workA_d, lworkA, infoA_d) );
  CHECK_ZERO( chol_upper_inplace(solverB, streamB, covB_d, dim, workB_d, lworkB, infoB_d) );
  CHECK_TRUE( cudaSuccess == cudaEventRecord(eventA, streamA) );
  CHECK_TRUE( cudaSuccess == cudaEventRecord(eventB, streamB) );

  // Save initial state if required
  if(save_last == 0 || 1 >= first_label) {
    CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_p1, theta_cur_p1_d, (size_t)(p/2)*dim*sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_p2, theta_cur_p2_d, (size_t)(p/2)*dim*sizeof(float), cudaMemcpyDeviceToHost) );
    const size_t slot0 = (size_t)(1 - first_label);
    savesamps(samps, slot0, (size_t)saveit, 1, theta_p1, theta_p2, (size_t)p, (size_t)dim);
  }

  // Main loop (updates both groups each iteration; cov updates every cov_it)
  for(int itn = 1; itn < it; itn++) {
    // Refresh covariances/choleskys every cov_it iterations (using current populations)
    if(((itn-1) % cov_it) == 0) {
      CHECK_ZERO( cov_gpu(&cublasA, theta_cur_p1_d, covA_d, p/2, dim, onesV_d, wsA_d, scaleCov) );
      CHECK_ZERO( cov_gpu(&cublasB, theta_cur_p2_d, covB_d, p/2, dim, onesV_d, wsB_d, scaleCov) );
      CHECK_ZERO( chol_upper_inplace(solverA, streamA, covA_d, dim, workA_d, lworkA, infoA_d) );
      CHECK_ZERO( chol_upper_inplace(solverB, streamB, covB_d, dim, workB_d, lworkB, infoB_d) );
      CHECK_TRUE( cudaSuccess == cudaEventRecord(eventA, streamA) );
      CHECK_TRUE( cudaSuccess == cudaEventRecord(eventB, streamB) );
    }

    // Each stream waits for the other group's Cholesky (cross-covariance usage)
    CHECK_TRUE( cudaSuccess == cudaStreamWaitEvent(streamA, eventB, 0) );
    CHECK_TRUE( cudaSuccess == cudaStreamWaitEvent(streamB, eventA, 0) );

    // Proposals for group 1 use cov from group 2 (covB_d); group 2 uses covA_d.
    CHECK_ZERO( generateMVN_gpu(&rngA, &cublasA, theta_nxt_p1_d, (size_t)(p/2), covB_d, (size_t)dim) );
    CHECK_ZERO( generateMVN_gpu(&rngB, &cublasB, theta_nxt_p2_d, (size_t)(p/2), covA_d, (size_t)dim) );

    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandGenerateUniform(rngA, unifA_d, (size_t)(p/2)) );
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandGenerateUniform(rngB, unifB_d, (size_t)(p/2)) );

    CHECK_TRUE( cudaSuccess == cudaMemsetAsync(accA_d, 0, (size_t)(p/2)*sizeof(int), streamA) );
    CHECK_TRUE( cudaSuccess == cudaMemsetAsync(accB_d, 0, (size_t)(p/2)*sizeof(int), streamB) );

    step_stream(threads, blocks,
                theta_cur_p1_d, logdens_cur_p1_d,
                theta_nxt_p1_d, logdens_nxt_p1_d,
                Xf_d, Xi_d,
                unifA_d, accA_d,
                streamA);
    step_stream(threads, blocks,
                theta_cur_p2_d, logdens_cur_p2_d,
                theta_nxt_p2_d, logdens_nxt_p2_d,
                Xf_d, Xi_d,
                unifB_d, accB_d,
                streamB);

    CHECK_TRUE( cudaSuccess == cudaStreamSynchronize(streamA) );
    CHECK_TRUE( cudaSuccess == cudaStreamSynchronize(streamB) );

    CHECK_TRUE( cudaSuccess == cudaMemcpy(accA_h, accA_d, (size_t)(p/2)*sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_TRUE( cudaSuccess == cudaMemcpy(accB_h, accB_d, (size_t)(p/2)*sizeof(int), cudaMemcpyDeviceToHost) );
    int totacc = sum_ints(accA_h, p/2) + sum_ints(accB_h, p/2);
    float ar = ((float)totacc)/((float)p);
    PRINT("\rExecuting iteration %d of %d (acceptance rate: %f)", itn+1, it, ar);
    if(itn == it-1) PRINT("\n");

    // Swap current/next
    float *tmp;
    tmp = theta_cur_p1_d; theta_cur_p1_d = theta_nxt_p1_d; theta_nxt_p1_d = tmp;
    tmp = theta_cur_p2_d; theta_cur_p2_d = theta_nxt_p2_d; theta_nxt_p2_d = tmp;
    tmp = logdens_cur_p1_d; logdens_cur_p1_d = logdens_nxt_p1_d; logdens_nxt_p1_d = tmp;
    tmp = logdens_cur_p2_d; logdens_cur_p2_d = logdens_nxt_p2_d; logdens_nxt_p2_d = tmp;

    const int iter_label = itn + 1;
    if(save_last == 0 || iter_label >= first_label) {
      const size_t slot = (size_t)(iter_label - first_label);
      CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_p1, theta_cur_p1_d, (size_t)(p/2)*dim*sizeof(float), cudaMemcpyDeviceToHost) );
      CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_p2, theta_cur_p2_d, (size_t)(p/2)*dim*sizeof(float), cudaMemcpyDeviceToHost) );
      savesamps(samps, slot, (size_t)saveit, (size_t)iter_label, theta_p1, theta_p2, (size_t)p, (size_t)dim);
    }
  }

  // Cleanup
  if(lenXf > 0) CHECK_TRUE( cudaSuccess == cudaFree(Xf_d) );
  if(lenXi > 0) CHECK_TRUE( cudaSuccess == cudaFree(Xi_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(theta_cur_p1_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(theta_nxt_p1_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(theta_cur_p2_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(theta_nxt_p2_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(logdens_cur_p1_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(logdens_nxt_p1_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(logdens_cur_p2_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(logdens_nxt_p2_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(unifA_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(unifB_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(accA_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(accB_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(covA_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(covB_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(wsA_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(wsB_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(onesV_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(workA_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(workB_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(infoA_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(infoB_d) );

  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandDestroyGenerator(rngA) );
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandDestroyGenerator(rngB) );
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasDestroy(cublasA) );
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasDestroy(cublasB) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnDestroy(solverA) );
  CHECK_TRUE( CUSOLVER_STATUS_SUCCESS == cusolverDnDestroy(solverB) );

  CHECK_TRUE( cudaSuccess == cudaEventDestroy(eventA) );
  CHECK_TRUE( cudaSuccess == cudaEventDestroy(eventB) );
  CHECK_TRUE( cudaSuccess == cudaStreamDestroy(streamA) );
  CHECK_TRUE( cudaSuccess == cudaStreamDestroy(streamB) );

  dlclose(lib);

  FREE(theta_p1);
  FREE(theta_p2);
  FREE(onesV);
  FREE(accA_h);
  FREE(accB_h);

  RANDFREE;
  return(0);
}
