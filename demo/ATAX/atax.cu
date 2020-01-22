/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "../../runtime.cuh"
#include "../../common/polybenchUtilFuncts.cuh"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

/* Problem size. */
#define NX 4096
#define NY 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
  int i, j;

  for (i = 0; i < NX; i++)
  {
    x[i] = i * M_PI;
    for (j = 0; j < NY; j++)
    {
      A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
}

void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
  int i, fail;
  fail = 0;

  for (i = 0; i < NY; i++)
  {
    if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
    {
      fail++;
    }
  }

  // print results
  printf(
      "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
      "%d\n",
      PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp,
                             Policy con)
{
  __shared__ int bid[1];
  __shared__ int miss_num[1];
  if (!runable_retreat(bid, miss_num, con))
    return;

  int i = bid[0] * blockDim.x + threadIdx.x;

  if (i < NX)
  {
    int j;
    for (j = 0; j < NY; j++)
    {
      tmp[i] += A[i * NY + j] * x[j];
    }
  }
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp,
                             Policy con)
{
  __shared__ int bid[1];
  __shared__ int miss_num[1];
  if (!runable_retreat(bid, miss_num, con))
    return;

  int j = bid[0] * blockDim.x + threadIdx.x;

  if (j < NY)
  {
    int i;
    for (i = 0; i < NX; i++)
    {
      y[j] += A[i * NY + j] * tmp[i];
    }
  }
}

void atax_cpu(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
  int i, j;

  for (i = 0; i < NY; i++)
  {
    y[i] = 0;
  }

  for (i = 0; i < NX; i++)
  {
    tmp[i] = 0;

    for (j = 0; j < NY; j++)
    {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }

    for (j = 0; j < NY; j++)
    {
      y[j] = y[j] + A[i * NY + j] * tmp[i];
    }
  }
}

void ataxGpu(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp,
             DATA_TYPE *y_outputFromGpu, const int &app_id)
{
  DATA_TYPE *A_gpu;
  DATA_TYPE *x_gpu;
  DATA_TYPE *y_gpu;
  DATA_TYPE *tmp_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
  cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY);
  cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY);
  cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX);

  cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
  cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
  cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)(ceil(((float)NX) / ((float)block.x))), 1);
  dim3 grid2((size_t)(ceil(((float)NY) / ((float)block.x))), 1);
  //============================================
  App app(app_id);
  Policy task1 = dispatcher(app, grid1, block);
  // app.release();
  Policy task2 = dispatcher(app, grid2, block);
  double t_start, t_end;
  bool flag = true;
  t_start = rtclock();
  if (flag)
  {
    kernel_splitting_send(app_id);
    atax_kernel1<<<task1.block_num, block>>>(A_gpu, x_gpu, tmp_gpu, task1);
  }
  else
  {
    int PART1 = 14;
    int PART2 = grid1.x - PART1;
    dim3 grid1_1(PART1);
    dim3 grid1_2(PART2);
    Policy task1_1 = dispatcher(app, grid1_1, block);
    app.release();
    task_destroy(task2);
    Policy task1_2 = dispatcher(app, grid1_2, block, PART1);
    task2 = dispatcher(app, grid2, block);
    auto stream1 = getStream();
    auto stream2 = getStream();
    kernel_splitting_send(app_id);
    atax_kernel1<<<task1_1.block_num, block, 0, stream1>>>(A_gpu, x_gpu,
                                                           tmp_gpu, task1_1);
    kernel_splitting_receive(app_id);
    atax_kernel1<<<task1_2.block_num, block, 0, stream2>>>(A_gpu, x_gpu,
                                                           tmp_gpu, task1_2);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    task_destroy(task1_1);
    task_destroy(task1_2);
  }
  atax_kernel2<<<task2.block_num, block>>>(A_gpu, y_gpu, tmp_gpu, task2);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();
  t_end = rtclock();
  if (flag)
    kernel_splitting_receive(app_id);
  task_destroy(task1);
  task_destroy(task2);
  //============================================
  fprintf(stdout, "time_in_frame_app_%d: %0.6lfms\n", app_id,
          (t_end - t_start) * 1000);
  cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX,
             cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(x_gpu);
  cudaFree(y_gpu);
  cudaFree(tmp_gpu);
}

int main(int argc, char **argv)
{
  printf("start atax...\n");
  // GPU_argv_init();
  double t_start, t_end;
  int app_id = atoi(argv[1]);
  DATA_TYPE *A;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  
      tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  init_array(x, A);
  ataxGpu(A, x, y, tmp, y_outputFromGpu, app_id);

  t_start = rtclock();
  atax_cpu(A, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_outputFromGpu);

  free(A);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}
