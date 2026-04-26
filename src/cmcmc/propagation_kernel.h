/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#ifndef PROPAGATION_KERNEL_H
#define PROPAGATION_KERNEL_H

// This defines the function prototype to which dynamically loaded kernels must conform
void propagateParticles(float *betaItCur, float *betaItDel, int p, int dim, float *X, int Xr, int Xc, int threads, int blocks, float *unif);

#endif
