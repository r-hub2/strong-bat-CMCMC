/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#include <stdlib.h>

// Save the iterations in the populations theta_p1 and theta_p2 to samps
//   samps:     col major matrix ((saveit*p) x (dim+2))
//   slot:      the output slot we're saving into, zero offset (0..saveit-1)
//   saveit:    total number of iterations being saved
//   iter_label: iteration label stored in column 1 (typically 1..T)
//   theta_p1:  col major matrix for first population ((p/2) x dim)
//   theta_p2:  col major matrix for second population ((p/2) x dim)
//   p:         number of particles in the population
//   dim:       dimension of the sample space
void savesamps(double* restrict samps,
               const size_t slot, const size_t saveit,
               const size_t iter_label,
               const float* theta_p1, const float* theta_p2, const size_t p, const size_t dim) {
  for(int i=0; i<p/2; i++) {
    samps[slot*p + i] = (double)iter_label;
    samps[slot*p + i + saveit*p] = (double)(i+1);
    for(int j=0; j<dim; j++) {
      samps[slot*p + i + (j+2)*saveit*p] = (double)(theta_p1[i + j*(p/2)]);
    }
  }
  for(int i=0; i<p/2; i++) {
    samps[slot*p + p/2+i] = (double)iter_label;
    samps[slot*p + p/2+i + saveit*p] = (double)(p/2+i+1);
    for(int j=0; j<dim; j++) {
      samps[slot*p + p/2+i + (j+2)*saveit*p] = (double)(theta_p2[i + j*(p/2)]);
    }
  }
}

void savesamps_all(double* restrict samps,
                   const size_t slot, const size_t saveit,
                   const size_t iter_label,
                   const float* theta_all, const size_t p, const size_t dim) {
  for(size_t i = 0; i < p; i++) {
    samps[slot*p + i] = (double)iter_label;
    samps[slot*p + i + saveit*p] = (double)(i+1);
    for(size_t j = 0; j < dim; j++) {
      samps[slot*p + i + (j+2)*saveit*p] = (double)(theta_all[i + j*p]);
    }
  }
}
