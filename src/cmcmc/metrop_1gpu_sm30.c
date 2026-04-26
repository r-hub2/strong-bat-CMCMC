/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#include "check.h"
#include "config.h"
#include "generate_mvn.h"
#include <dlfcn.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>





curandGenerator_t curng;
cublasHandle_t cublas;
float *unif_d;
int *acc, *acc_d;
void *lib;
void (*metrop_1gpu_sm30_setup_D)(int, int, float*, float*, const float*, const int*);
void (*metrop_1gpu_sm30_D)(int, int, const float*, const float*, float*, float*, const float*, const int*, const float*, int*);





// Setup / Cleanup
int metrop_1gpu_sm30_setup(const char* CUDAlib, int threads, int blocks,
                           float *theta_cur_p1_d, float *logdens_cur_p1_d,
                           float *theta_cur_p2_d, float *logdens_cur_p2_d,
                           int p, int dim, const float *Xf_d, const int *Xi_d,
                           const int64_t seed) {
  if(p % 2) EPRINT("\n\nERROR: Population size must be even.\n\n");

  // Attach GPU library
  lib = dlopen(CUDAlib, RTLD_LAZY|RTLD_LOCAL);
  if(!lib) {
    EPRINT("Error: '%s' at %s:%d", dlerror(), __FILE__, __LINE__);
    return(1);
  }

  // Get function pointers
  *(void **) (&metrop_1gpu_sm30_setup_D) = dlsym(lib, "metrop_1gpu_sm30_setup_D");
  if(!metrop_1gpu_sm30_setup_D) {
    EPRINT("Error: '%s' at %s:%d", dlerror(), __FILE__, __LINE__);
    dlclose(lib);
    return(1);
  }

  *(void **) (&metrop_1gpu_sm30_D) = dlsym(lib, "metrop_1gpu_sm30_D");
  if(!metrop_1gpu_sm30_D) {
    EPRINT("Error: '%s' at %s:%d", dlerror(), __FILE__, __LINE__);
    dlclose(lib);
    return(1);
  }

  // cuRAND init
  // For <sm35 we do this via host API since cuBLAS can't run on device anyway
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT) );
  if(seed == (int64_t) 0) {
    PRINT("No seed set, randomly initialising.\n");
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(curng, (int64_t) time(NULL)) );
  } else {
    PRINT("Using seed %lld\n", seed);
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandSetPseudoRandomGeneratorSeed(curng, seed) );
  }

  // cuBLAS init
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasCreate(&cublas) );

  // Compute logdens for initial population
  metrop_1gpu_sm30_setup_D(threads, blocks,
                           theta_cur_p1_d, logdens_cur_p1_d,
                           Xf_d, Xi_d);
  metrop_1gpu_sm30_setup_D(threads, blocks,
                           theta_cur_p2_d, logdens_cur_p2_d,
                           Xf_d, Xi_d);

  // Storage for uniforms used in MH accept/reject
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&unif_d, (p/2)*sizeof(float)) );
  // Storage for computing acceptance rate
  acc = (int*) ALLOC((p/2), sizeof(int));
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&acc_d, (p/2)*sizeof(int)) );

  return(0);
}





int metrop_1gpu_sm30_cleanup() {
  // Detach library
  CHECK_ZERO( dlclose(lib) );

  // cuRAND finalise
  CHECK_TRUE( CURAND_STATUS_SUCCESS == curandDestroyGenerator(curng) );

  // cuBLAS finalise
  CHECK_TRUE( CUBLAS_STATUS_SUCCESS == cublasDestroy(cublas) );

  // Free memory
  FREE(acc);
  CHECK_TRUE( cudaSuccess == cudaFree(acc_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(unif_d) );

  return(0);
}





// Iterate
int metrop_1gpu_sm30(int threads, int blocks, int its, float *ar,
                     float **theta_cur_d, float **logdens_cur_d,
                     float **theta_nxt_d, float **logdens_nxt_d,
                     int p, int dim, const float *theta_cov_d,
                     const float *Xf_d, const int *Xi_d) {
  if(p % 2) EPRINT("\n\nERROR: Population size must be even.\n\n");

  if(its <= 0) return(1);

  float *tmp_d; // Temporary pointer for swapping current/next

  // Reset acceptance rate counter
  CHECK_TRUE( cudaSuccess == cudaMemset(acc_d, 0, (p/2)*sizeof(int)) );
  for(int it = 0; it < its; it++) {
    // Generate MVN proposal disturbances for particles to be updated
    generateMVN_gpu(&curng, &cublas, *theta_nxt_d, p/2, theta_cov_d, dim);

    // Generate uniforms for the accept/reject step
    CHECK_TRUE( CURAND_STATUS_SUCCESS == curandGenerateUniform(curng, unif_d, p/2) );

    // Perform a MH step
    metrop_1gpu_sm30_D(threads, blocks,
                       *theta_cur_d, *logdens_cur_d,
                       *theta_nxt_d, *logdens_nxt_d,
                       Xf_d, Xi_d,
                       unif_d, acc_d);

    // Swap current and next populations
    tmp_d = *theta_cur_d;
    *theta_cur_d = *theta_nxt_d;
    *theta_nxt_d = tmp_d;

    tmp_d = *logdens_cur_d;
    *logdens_cur_d = *logdens_nxt_d;
    *logdens_nxt_d = tmp_d;
  }
  CHECK_TRUE( cudaSuccess == cudaMemcpy(acc, acc_d, (p/2)*sizeof(int), cudaMemcpyDeviceToHost) );
  int totacc;
  for(int i = 0; i<(p/2); i++) {
    totacc += acc[i];
  }
  *ar = ((float)totacc)/((float)(its*(p/2)));

  return(0);
}





// Testing (unique to this metrop method, used for testing kernel log density evaluations)
float metrop_1gpu_sm30_test(const char* CUDAlib, float *theta, int dim,
                            const float *Xf, int lenXf, const int *Xi, int lenXi) {
  CHECK_TRUE( cudaSuccess == cudaSetDevice(0) );

  // Attach GPU library
  lib = dlopen(CUDAlib, RTLD_LAZY|RTLD_LOCAL);
  if(!lib) {
    EPRINT("Error: '%s' at %s:%d", dlerror(), __FILE__, __LINE__);
    return(1);
  }

  // Get function pointers
  *(void **) (&metrop_1gpu_sm30_setup_D) = dlsym(lib, "metrop_1gpu_sm30_setup_D");
  if(!metrop_1gpu_sm30_setup_D) {
    EPRINT("Error: '%s' at %s:%d", dlerror(), __FILE__, __LINE__);
    dlclose(lib);
    return(1);
  }

  // Allocate minimal GPU memory
  float *Xf_d;
  if(lenXf > 0)
    CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&Xf_d, lenXf*sizeof(float)) );
  int *Xi_d;
  if(lenXi > 0)
    CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&Xi_d, lenXi*sizeof(int)) );

  float *theta_d;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&theta_d, dim*sizeof(float)) );

  float *logdens_d;
  CHECK_TRUE( cudaSuccess == cudaMalloc((void **)&logdens_d, sizeof(float)) );

  // Send X's and theta to GPU
  if(lenXf > 0)
    CHECK_TRUE( cudaSuccess == cudaMemcpy(Xf_d, Xf, lenXf*sizeof(float), cudaMemcpyHostToDevice) );
  if(lenXi > 0)
    CHECK_TRUE( cudaSuccess == cudaMemcpy(Xi_d, Xi, lenXi*sizeof(int), cudaMemcpyHostToDevice) );
  CHECK_TRUE( cudaSuccess == cudaMemcpy(theta_d, theta, dim*sizeof(float), cudaMemcpyHostToDevice) );

  // Evaluate the kernel for just this theta
  metrop_1gpu_sm30_setup_D(1, 1,
                           theta_d, logdens_d,
                           Xf_d, Xi_d);

  // Pull back log density value
  float logdens;
  CHECK_TRUE( cudaSuccess == cudaMemcpy(&logdens, logdens_d, sizeof(float), cudaMemcpyDeviceToHost) );

  // Tidy up
  if(lenXf > 0)
    CHECK_TRUE( cudaSuccess == cudaFree(Xf_d) );
  if(lenXi > 0)
    CHECK_TRUE( cudaSuccess == cudaFree(Xi_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(theta_d) );
  CHECK_TRUE( cudaSuccess == cudaFree(logdens_d) );

  // Detach library
  CHECK_ZERO( dlclose(lib) );

  return(logdens);
}
