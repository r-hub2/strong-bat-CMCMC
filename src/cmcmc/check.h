/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */
#ifndef CHECK_H
#define CHECK_H

#include "config.h"
#include <cuda_runtime.h>

#define CHECK_ZERO(x) do { \
    int retval = (x); \
    if (retval != 0) { \
      EPRINT("Error: %s returned %d at %s:%d\n", #x, retval, __FILE__, __LINE__); \
      return(1); \
    } \
  } while (0)

#define CHECK_TRUE(x) do { \
    if( !(x) ) { \
      EPRINT("Error: %s not true at %s:%d\n", #x, __FILE__, __LINE__); \
      return(1); \
    } \
  } while (0)

#define CHECK_NONNULL(x) do { \
    if((x) == NULL) { \
      EPRINT("Error: %s returned NULL at %s:%d\n", #x, __FILE__, __LINE__); \
      return(1); \
    } \
  } while (0)

#define CHECK_FULL_ERR(x) { CHECK_ZERO( gpuAssert((x), __FILE__, __LINE__) ); }
    inline int gpuAssert(cudaError_t code, const char *file, int line) {
      if (code != cudaSuccess) {
        EPRINT("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        return(1);
      }
      return(0);
    }

#define GDLT256(x) gdlt256(x)
    inline int gdlt256(int x) {
      int res;
      for(res = 256; res >= 1; res--) {
        if(x%res == 0)
          break;
      }
      return(res);
    }

#endif
