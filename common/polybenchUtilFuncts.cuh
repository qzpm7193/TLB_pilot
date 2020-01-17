#include <cuda.h>
#include <sys/time.h>
#include <cstdio>
#define SMALL_FLOAT_VAL 0.00000001f
#define GPU_DEVICE 0
void GPU_argv_init()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
}
float absVal(float a)
{
  if (a < 0)
  {
    return (a * -1);
  }
  else
  {
    return a;
  }
}

float percentDiff(double val1, double val2)
{
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
  {
    return 0.0f;
  }

  else
  {
    return 100.0f *
           (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
  }
}
