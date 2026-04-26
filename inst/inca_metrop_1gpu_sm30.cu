/*
 INCA Metropolis kernel (single population, no streams).
 */

#include "kernels/cmcmc_kernel.h"
#include <cuda_runtime.h>

__global__ void metrop_1gpu_sm30_setup_d(float *theta_cur, float *logdens_cur,
                                        const float *Xf, const int *Xi) {
  int off = threadIdx.x + blockIdx.x * blockDim.x;
  if (off >= P) return;

  theta_cur += off;
  logdens_cur += off;

  float theta_local[DIM];
  for (int i = 0; i < DIM; i++) {
    theta_local[i] = theta_cur[i * P];
  }
  *logdens_cur = logdens(theta_local, Xf, Xi);
}

extern "C" {
  void metrop_1gpu_sm30_setup_D(int threads, int blocks,
                                float *theta_cur, float *logdens_cur,
                                const float *Xf, const int *Xi) {
    metrop_1gpu_sm30_setup_d<<<blocks, threads>>>(theta_cur, logdens_cur, Xf, Xi);
  }
}

__global__ void metrop_1gpu_sm30_d(const float *theta_cur, const float *logdens_cur,
                                  float *theta_nxt, float *logdens_nxt,
                                  const float *Xf, const int *Xi,
                                  const float *unif, int *acc) {
  int off = threadIdx.x + blockIdx.x * blockDim.x;
  if (off >= P) return;

  theta_cur += off;
  logdens_cur += off;
  theta_nxt += off;
  logdens_nxt += off;
  unif += off;
  acc += off;

  for (int i = 0; i < DIM; i++) {
    theta_nxt[i * P] += theta_cur[i * P];
  }

  float theta_nxt_local[DIM];
  for (int i = 0; i < DIM; i++) {
    theta_nxt_local[i] = theta_nxt[i * P];
  }
  *logdens_nxt = logdens(theta_nxt_local, Xf, Xi);

  if (*logdens_nxt - *logdens_cur > logf(*unif)) {
    (*acc)++;
  } else {
    for (int i = 0; i < DIM; i++) {
      theta_nxt[i * P] = theta_cur[i * P];
    }
    *logdens_nxt = *logdens_cur;
  }
}

extern "C" {
  void metrop_1gpu_sm30_D(int threads, int blocks,
                          const float *theta_cur, const float *logdens_cur,
                          float *theta_nxt, float *logdens_nxt,
                          const float *Xf, const int *Xi,
                          const float *unif_d, int *acc) {
    metrop_1gpu_sm30_d<<<blocks, threads>>>(theta_cur, logdens_cur,
                                           theta_nxt, logdens_nxt,
                                           Xf, Xi,
                                           unif_d, acc);
  }
}

