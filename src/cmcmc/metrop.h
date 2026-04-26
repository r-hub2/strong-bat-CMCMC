/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#ifndef METROP_H
#define METROP_H

typedef enum {
  METROP_1GPU_SM30 = 1,
  METROP_1GPU_SM35 = 2,
  METROP_2GPU_SM30 = 3,
  METROP_2GPU_SM35 = 4
} metropmethod_t;

#include "metrop_1gpu_sm30.h"

#endif
