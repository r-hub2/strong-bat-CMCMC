/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#include "kernels/cmcmc_kernel.h"
#include <cuda_runtime.h>





// Setup / Cleanup
__global__ void metrop_1gpu_sm30_setup_d(float *theta_cur, float *logdens_cur,
                                         const float *Xf, const int *Xi) {
  // Initial offset
  int off = threadIdx.x + blockIdx.x*blockDim.x;

  // Shift pointers
  theta_cur   += off;
  logdens_cur += off;

  // Gather the strided theta vector into a contiguous local array for the target kernel.
  float theta_local[DIM];
  for (int i = 0; i < DIM; i++) {
    theta_local[i] = theta_cur[i * (P / 2)];
  }

  // Compute log density for initial population
  *logdens_cur = logdens(theta_local, Xf, Xi);
}

extern "C" {
  void metrop_1gpu_sm30_setup_D_stream(int threads, int blocks,
                                       float *theta_cur, float *logdens_cur,
                                       const float *Xf, const int *Xi,
                                       cudaStream_t stream) {
    metrop_1gpu_sm30_setup_d<<<blocks, threads, 0, stream>>>(theta_cur, logdens_cur, Xf, Xi);
  }

  void metrop_1gpu_sm30_setup_D(int threads, int blocks,
                                float *theta_cur, float *logdens_cur,
                                const float *Xf, const int *Xi) {
    metrop_1gpu_sm30_setup_D_stream(threads, blocks, theta_cur, logdens_cur, Xf, Xi, 0);
  }
}





// Iterate
__global__ void metrop_1gpu_sm30_d(const float *theta_cur, const float *logdens_cur,
                                   float *theta_nxt, float *logdens_nxt,
                                   const float *Xf, const int *Xi,
                                   const float *unif, int *acc) {
  // Shift pointers to our particle in current and proposal streams, as well as RNG stream
  int off = threadIdx.x + blockIdx.x*blockDim.x;
  theta_cur   += off;
  logdens_cur += off;
  theta_nxt   += off;
  logdens_nxt += off;
  unif        += off;
  acc         += off;

  // Add proposal to current position
  for(int i = 0; i < DIM; i++) {
    theta_nxt[i*(P/2)] += theta_cur[i*(P/2)];
  }

  // Compute log target density at proposal
  float theta_nxt_local[DIM];
  for (int i = 0; i < DIM; i++) {
    theta_nxt_local[i] = theta_nxt[i * (P / 2)];
  }
  *logdens_nxt = logdens(theta_nxt_local, Xf, Xi);

  if(*logdens_nxt - *logdens_cur > log(*unif)) {
    // Accept
    // ... do nothing
    (*acc)++;
  } else {
    // Reject
    // ... copy forward cur to nxt
    for(int i = 0; i < DIM; i++) {
      theta_nxt[i*(P/2)] = theta_cur[i*(P/2)];
      *logdens_nxt = *logdens_cur;
    }
  }
}

extern "C" {
  void metrop_1gpu_sm30_D_stream(int threads, int blocks,
                                 const float *theta_cur, const float *logdens_cur,
                                 float *theta_nxt, float *logdens_nxt,
                                 const float *Xf, const int *Xi,
                                 const float *unif_d, int *acc,
                                 cudaStream_t stream) {
    metrop_1gpu_sm30_d<<<blocks, threads, 0, stream>>>(theta_cur, logdens_cur,
                                                      theta_nxt, logdens_nxt,
                                                      Xf, Xi,
                                                      unif_d, acc);
  }

  void metrop_1gpu_sm30_D(int threads, int blocks,
                          const float *theta_cur, const float *logdens_cur,
                          float *theta_nxt, float *logdens_nxt,
                          const float *Xf, const int *Xi,
                          const float *unif_d, int *acc) {
    metrop_1gpu_sm30_D_stream(threads, blocks,
                              theta_cur, logdens_cur,
                              theta_nxt, logdens_nxt,
                              Xf, Xi,
                              unif_d, acc,
                              0);
  }
}
