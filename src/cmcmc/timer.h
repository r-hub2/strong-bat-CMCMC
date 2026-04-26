/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include "config.h"

static inline double timer(char *str) {
  struct timeval t;
  static double stTimer = 0;

  double last_time = stTimer;

  gettimeofday(&t, NULL);
  stTimer = t.tv_sec + t.tv_usec*1.0e-6;

  if(str != NULL) {
    PRINT("%s: %lf secs\n", str, stTimer - last_time);
  }
  return(stTimer - last_time);
}

#endif
