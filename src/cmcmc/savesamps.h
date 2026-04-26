/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#ifndef SAVESAMPS_H
#define SAVESAMPS_H

#include <stdlib.h>

// Save the iterations in the populations theta_p1 and theta_p2 to samps
//   samps:     col major matrix ((saveit*p) x (dim+2))
//   it:        the iteration number we're saving, zero offset
//   saveit:    total number of iterations being saved
//   theta_p1:  col major matrix for first population ((p/2) x dim)
//   theta_p2:  col major matrix for second population ((p/2) x dim)
//   p:         number of particles in the population
//   dim:       dimension of the sample space
void savesamps(double* restrict samps,
               const size_t slot, const size_t saveit,
               const size_t iter_label,
               const float* theta_p1, const float* theta_p2, const size_t p, const size_t dim);

void savesamps_all(double* restrict samps,
                   const size_t slot, const size_t saveit,
                   const size_t iter_label,
                   const float* theta_all, const size_t p, const size_t dim);

#endif
