/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#ifndef AUTOTUNE_H
#define AUTOTUNE_H

#include "metrop.h"

// Autotune the grid and block sizes.
int autotune(int* threads, int* blocks, const metropmethod_t metropmethod, int cov_it,
             float** theta_cur, float** logdens_cur,
             float** theta_nxt, float** logdens_nxt,
             int p, int dim, float* cov_d, const float* Xf_d, const int* Xi_d);

#endif
