/*
 * Based on the original CMCMC implementation by Louis Aslett, April 2018.
 * Revised for this package by Ahmad ALQabandi and Louis Aslett, 2026.
 */

#include "config.h"
#include "check.h"
#include "generate_mvn.h"
#include "propagation_kernel.h"
#include "metrop.h"

#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>





// Beginning of GPU Architecture definitions from helper_cuda.h in CUDA samples
inline int _ConvertSMVer2Cores(int major, int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct sSMtoCores
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
    {
    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
    { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
    { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
    { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
    { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
    { 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class

    {   -1, -1 }
    };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
  return nGpuArchCoresPerSM[index-1].Cores;
}
// End of GPU Architecture definitions from helper_cuda.h in CUDA samples

static inline double att() {
  struct timeval t;
  static double stTimer = 0;

  double last_time = stTimer;

  gettimeofday(&t, NULL);
  stTimer = t.tv_sec + t.tv_usec*1.0e-6;

  return(stTimer - last_time);
}

int autotune(int* threads, int* blocks, const metropmethod_t metropmethod, int cov_it,
             float** theta_cur_d, float** logdens_cur_d,
             float** theta_nxt_d, float** logdens_nxt_d,
             int p, int dim, float* cov_d, const float* Xf_d, const int* Xi_d) {
  int device;
  struct cudaDeviceProp deviceProp;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&deviceProp, device);

  int SMs = deviceProp.multiProcessorCount, cores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
  PRINT("  Detected %2d multiprocessors with %3d cores each.\n", SMs, cores);
  if(p < SMs*cores) {
    EPRINT("  WARNING: size of population (%d) is less than the number of cores (%d).\n", p, SMs*cores);
  }

  int b=1, t;
  double cur=0.0, best=1e9;
  float ar; // Dummy for acceptance rate

  CHECK_FULL_ERR( cudaDeviceSynchronize() );

  PRINT("Timings are for %d population samples per MH iteration.\n\n", p/2);
  PRINT("   blocks x threads | avg time (secs)\n");
  PRINT("  ------------------+----------------\n");
  *blocks = 1;
  while((p/2)/b > 0) {
    if(b > deviceProp.maxGridSize[0])
      break;
    if((p/2)%b!=0 || !(t=(p/2)/b) || t>deviceProp.maxThreadsDim[0]) {
      b++;
      continue;
    }

    printf("   %6d x %7d | ", b, t);
    // TIMING
    for(int i=0; i<3; i++) {
      CHECK_FULL_ERR( cudaDeviceSynchronize() ); // <-- must do to get genuine timings
      att();
      switch(metropmethod) {
      case METROP_1GPU_SM30:
        metrop_1gpu_sm30(t, b, cov_it, &ar,
                         theta_cur_d, logdens_cur_d,
                         theta_nxt_d, logdens_nxt_d,
                         p, dim, cov_d,
                         Xf_d, Xi_d);
        break;
      default:
        EPRINT("Unknown metropmethod.");
      }
      if(gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__)) {
        EPRINT("failed\n");
        b++;
        continue;
      }
      CHECK_FULL_ERR( cudaDeviceSynchronize() ); // <-- must do to get genuine timings
      cur += att();
    }

    if(cur < best) {
      *blocks = b;
      *threads = t;
      best = cur;
    }
    printf("%lf\n", cur/(3.0*((double) cov_it)));
    b++;
    if(cur > 10*best)
      break;
  }
  if(best==1e9) {
    EPRINT("  Error: no working autotuned values found.\n");
    return(1);
  }
  printf("\n  Using autotuned selection of %d x %d\n", *blocks, *threads);

  return(0);
}
