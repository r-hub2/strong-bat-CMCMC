/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#ifndef METROP_1GPU_SM30_H
#define METROP_1GPU_SM30_H

// Setup / Cleanup
int metrop_1gpu_sm30_setup(const char* CUDAlib, int threads, int blocks,
                           float *theta_cur_p1_d, float *logdens_cur_p1_d,
                           float *theta_cur_p2_d, float *logdens_cur_p2_d,
                           int p, int dim, const float *Xf_d, const int *Xi_d,
                           const int64_t seed);

int metrop_1gpu_sm30_cleanup();

// Iterate
int metrop_1gpu_sm30(int threads, int blocks, int its, float *ar,
                     float **theta_cur_d, float **logdens_cur_d,
                     float **theta_nxt_d, float **logdens_nxt_d,
                     int p, int dim, const float *theta_cov_d,
                     const float *Xf_d, const int *Xi_d);

// Kernel testing
float metrop_1gpu_sm30_test(const char* CUDAlib, float *theta, int dim,
                            const float *Xf, int lenXf, const int *Xi, int lenXi);

#endif
