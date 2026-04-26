/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#ifndef CMCMC_H
#define CMCMC_H

#include "metrop.h"

// Run the contemporaneous MCMC sampler
//   CUDAlib:       string containing path to CUDA function determining target density
//   metropmethod:  the kind of sampler setup to run (see enumeration in metrop.h above)
//   theta_init_p1: initial population samples
//   samps:         is overwritten with saveit MCMC samples ((p*saveit) x dim matrix)
//                  in col major format
//   p:             number of particles in the population
//   it:            number of MCMC iterations to perform
//   save_last:     number of iterations to save from the end (0 => save all)
//   dim:           dimension of one particle
//   scaleCov:      scaling factor for the covariance
//   seed:          RNG seed (0 => use time)
//   device:        what GPU device to run on
//   Xf:            optional vector of floats to forward to target density (eg for fixed params/data)
//   lenXf:         size of Xf
//   Xi:            optional vector of ints to forward to target density (eg for fixed params/data)
//   lenXi:         size of Xi
//
// Returns 0 on success
int cmcmc(const char* restrict CUDAlib, const metropmethod_t metropmethod,
          const float* restrict theta_init_p1, const float* restrict theta_init_p2,
          double* restrict samps, const int p, const int it, const int cov_it, const int save_last, const int dim,
          const float scaleCov,
          const int64_t seed, const int device,
          const float* restrict Xf, const int lenXf, const int* restrict Xi, const int lenXi);

// Run the INCA sampler (global covariance across all particles)
int inca(const char* restrict CUDAlib, const metropmethod_t metropmethod,
         const float* restrict theta_init,
         double* restrict samps, const int p, const int it, const int cov_it, const int burnin, const int save_last, const int dim,
         const float scaleCov,
          const int64_t seed, const int device,
         const float* restrict Xf, const int lenXf, const int* restrict Xi, const int lenXi);

#endif
