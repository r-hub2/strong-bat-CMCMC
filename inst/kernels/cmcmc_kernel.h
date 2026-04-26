/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#ifndef CMCMC_KERNEL_H
#define CMCMC_KERNEL_H

// Note that DIM and P should be automatically defined when each kernel is
// compiled

#define THETA(d) theta[d*(P/2)]

__device__ float logdens(const float *theta, const float *Xf, const int *Xi);

#endif
