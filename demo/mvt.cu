/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "../runtime.cuh"
#include "../common/polybenchUtilFuncts.cuh"
// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define N 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
                DATA_TYPE *y2)
{
  int i, j;

  for (i = 0; i < N; i++)
  {
    x1[i] = ((DATA_TYPE)i) / N;
    x2[i] = ((DATA_TYPE)i + 1) / N;
    y1[i] = ((DATA_TYPE)i + 3) / N;
    y2[i] = ((DATA_TYPE)i + 4) / N;
    for (j = 0; j < N; j++)
    {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void runMvt(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
            DATA_TYPE *y2)
{
  int i, j;

  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      x1[i] = x1[i] + a[i * N + j] * y1[j];
    }
  }

  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      x2[i] = x2[i] + a[j * N + i] * y2[j];
    }
  }
}

void compareResults(DATA_TYPE *x1, DATA_TYPE *x1_outputFromGpu, DATA_TYPE *x2,
                    DATA_TYPE *x2_outputFromGpu)
{
  int i, fail;
  fail = 0;

  for (i = 0; i < N; i++)
  {
    if (percentDiff(x1[i], x1_outputFromGpu[i]) >
        PERCENT_DIFF_ERROR_THRESHOLD)
    {
      fail++;
    }

    if (percentDiff(x2[i], x2_outputFromGpu[i]) >
        PERCENT_DIFF_ERROR_THRESHOLD)
    {
      fail++;
    }
  }

  // Print results
  printf(
      "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
      "%d\n",
      PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

__global__ void mvt_kernel1(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1,
                            Policy con)
{
  __shared__ int bid[1];
  __shared__ int miss_num[1];
  if (!runable_retreat(bid, miss_num, con))
    return;

  int i = bid[0] * blockDim.x + threadIdx.x;
  if (i < N)
  {
    int j;
    for (j = 0; j < N; j++)
    {
      x1[i] += a[i * N + j] * y_1[j];
    }
  }
}

__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2,
                            Policy con)
{
  __shared__ int bid[1];
  __shared__ int miss_num[1];
  if (!runable_retreat(bid, miss_num, con))
    return;

  int i = bid[0] * blockDim.x + threadIdx.x;

  if (i < N)
  {
    int j;
    for (j = 0; j < N; j++)
    {
      x2[i] += a[j * N + i] * y_2[j];
    }
  }
}

void mvtCuda(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y_1,
             DATA_TYPE *y_2, DATA_TYPE *x1_outputFromGpu,
             DATA_TYPE *x2_outputFromGpu, const int &app_id)
{
  DATA_TYPE *a_gpu;
  DATA_TYPE *x1_gpu;
  DATA_TYPE *x2_gpu;
  DATA_TYPE *y_1_gpu;
  DATA_TYPE *y_2_gpu;
  cudaMalloc((void **)&a_gpu, sizeof(DATA_TYPE) * N * N);
  cudaMalloc((void **)&x1_gpu, sizeof(DATA_TYPE) * N);
  cudaMalloc((void **)&x2_gpu, sizeof(DATA_TYPE) * N);
  cudaMalloc((void **)&y_1_gpu, sizeof(DATA_TYPE) * N);
  cudaMalloc((void **)&y_2_gpu, sizeof(DATA_TYPE) * N);
  cudaMemcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  //===============================
  App app(app_id);
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil((float)N / ((float)DIM_THREAD_BLOCK_X)), 1);
  Policy task1 = dispatcher(app, grid, block);
  Policy task2 = dispatcher(app, grid, block);
  double t_start, t_end;
  bool flag = true;
  t_start = rtclock();
  if (flag)
  {
    //if it is a short-running kernel
    kernel_splitting_send(app_id);
    mvt_kernel1<<<task1.block_num, block>>>(a_gpu, x1_gpu, y_1_gpu, task1);
  }
  else
  {
    //if it is a  long-running kernel
    int PART1 = 14;
    int PART2 = grid.x - PART1;
    dim3 grid1_1(PART1);
    dim3 grid1_2(PART2);
    Policy task1_1 = dispatcher(app, grid1_1, block);
    app.release();
    task_destroy(task2);
    Policy task1_2 = dispatcher(app, grid1_2, block, PART1);
    task2 = dispatcher(app, grid, block);
    auto stream1 = getStream();
    auto stream2 = getStream();
    kernel_splitting_send(app_id);
    mvt_kernel1<<<task1_1.block_num, block, 0, stream1>>>(a_gpu, x1_gpu,
                                                          y_1_gpu, task1_1);
    kernel_splitting_receive(app_id);
    mvt_kernel1<<<task1_2.block_num, block, 0, stream2>>>(a_gpu, x1_gpu,
                                                          y_1_gpu, task1_2);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    task_destroy(task1_1);
    task_destroy(task1_2);
  }

  mvt_kernel2<<<task2.block_num, block>>>(a_gpu, x2_gpu, y_2_gpu, task2);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();
  if (flag)
    kernel_splitting_receive(app_id);
  t_end = rtclock();
  task_destroy(task1);
  task_destroy(task2);
  //==============================
  fprintf(stdout, "time_in_frame_app_%d: %0.6lfms\n", app_id,
          (t_end - t_start) * 1000);
  cudaMemcpy(x1_outputFromGpu, x1_gpu, sizeof(DATA_TYPE) * N,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(x2_outputFromGpu, x2_gpu, sizeof(DATA_TYPE) * N,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(a_gpu);
  cudaFree(x1_gpu);
  cudaFree(x2_gpu);
  cudaFree(y_1_gpu);
  cudaFree(y_2_gpu);
}

int main(int argc, char **argv)
{
  printf("start mvt...\n");
  GPU_argv_init();
  int app_id = atoi(argv[1]);
  double t_start, t_end;
  DATA_TYPE *a;
  DATA_TYPE *x1;
  DATA_TYPE *x2;
  DATA_TYPE *x1_outputFromGpu;
  DATA_TYPE *x2_outputFromGpu;
  DATA_TYPE *y_1;
  DATA_TYPE *y_2;

  a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  init_array(a, x1, x2, y_1, y_2);
  mvtCuda(a, x1, x2, y_1, y_2, x1_outputFromGpu, x2_outputFromGpu, app_id);

  t_start = rtclock();

  // run the algorithm on the CPU
  runMvt(a, x1, x2, y_1, y_2);

  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);

  free(a);
  free(x1);
  free(x2);
  free(x1_outputFromGpu);
  free(x2_outputFromGpu);
  free(y_1);
  free(y_2);

  return 0;
}