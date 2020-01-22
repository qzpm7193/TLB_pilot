#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
using namespace std;
int steps = 2048 * 40;
int threadnum = 128;
__device__ void sleep_overflow(int64_t num_cycles)
{
	int64_t cycles = 0;
	int64_t start = clock64();
	while (cycles < num_cycles)
	{
		cycles = clock64() - start;
	}
}

double GetMSTime(void)
{
	struct timeval stNowTime;

	gettimeofday(&stNowTime, NULL);

	return (1.0 * (stNowTime.tv_sec) * 1000 + 1.0 * (stNowTime.tv_usec) / 1000);
}
__device__ __inline__ int ld_gbl_cg(unsigned int *addr)
{
	unsigned int return_value;
	asm("ld.global.cg.s32 %0, [%1];"
		: "=r"(return_value)
		: "l"(addr));
	return return_value;
}

void add_diff(unsigned int *a, unsigned int *b)
{
	unsigned int *tmp = NULL;
	if (a > b)
	{
		tmp = a;
		a = b;
		b = tmp;
	}
	printf("add1=%p,add2=%p,address diff =%0.2f MB\n", a, b, (double)((unsigned long)b - (unsigned long)a) / 1024 / 1024);
}
__global__ void tlb2_normal2(unsigned int **my_array, int steps, int *compiler_array, int memNum)
{

	int k;
	unsigned int j = 0;
	unsigned int w = threadIdx.x % memNum;
	unsigned int compile = 0;
#pragma unroll 1
	for (k = 0; k < steps; k++)
	{
		j = ld_gbl_cg(my_array[w] + j);
		asm volatile("membar.cta;");
		w = (w + 1) % memNum;
		compile += j;
	}

	compiler_array[blockDim.x * blockIdx.x + threadIdx.x] = j;
	compiler_array[blockDim.x * blockIdx.x + threadIdx.x + 1] = my_array[w][j];
	compiler_array[blockDim.x * blockIdx.x + threadIdx.x + 2] = compile;
}

void pip_write()
{
	int fd;
	char myfifo[] = "/tmp/myfifo";
	mkfifo(myfifo, 0666);

	fd = open(myfifo, O_WRONLY);
	write(fd, "=================", sizeof("================="));
	close(fd);

	unlink(myfifo);
}
void tlb2_normal(int normal_tlbsize, int normal_stride)
{
	unsigned int devNo = 0;
	cudaSetDevice(devNo);
	int memNum = normal_tlbsize / normal_stride;
	printf("memNum=%d\n", memNum);
	unsigned int *t_arr[memNum];
	unsigned int **d_arr;
	cudaMalloc((void **)&d_arr, sizeof(unsigned int *) * memNum);
	for (int i = 0; i < memNum; i++)
	{
		cudaMalloc((void **)&(t_arr[i]), sizeof(unsigned int) * 2 * 1024 * 256);
		cudaMemset(t_arr[i], 0, sizeof(unsigned int) * 2 * 1024 * 256);
	}
	cudaMemcpy(d_arr, t_arr, sizeof(unsigned int *) * memNum, cudaMemcpyHostToDevice);
	int SMcount = 0;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, devNo);
	cout << "#" << props.name << ": cuda " << props.major << "." << props.minor << endl;
	SMcount = props.multiProcessorCount;
	printf("NORMAL PROGRAM\n");
	printf("SMcount=%d\n", SMcount);
	int *compiler_array;
	cudaMalloc((void **)&compiler_array, sizeof(int) * (SMcount * 3 * threadnum));
	cudaMemset(compiler_array, 0, sizeof(int) * SMcount * threadnum * 3);
	double t1, t2;
	fprintf(stderr, "normal_tlbsize=%d,normal_stride=%d,steps=%d,threadnum=%d\n", normal_tlbsize, normal_stride, steps, threadnum);
	pip_write();
	t1 = GetMSTime();

	tlb2_normal2<<<SMcount, threadnum>>>(d_arr, steps, compiler_array, memNum);

	cudaDeviceSynchronize();
	t2 = GetMSTime();

	fprintf(stderr, "GPU-concatenation-time: %4.2lf ms\n", (t2 - t1));
	for (int i = 0; i < memNum; i++)
	{
		cudaFree(t_arr[i]);
	}
	cudaFree(d_arr);
}

int main(int argc, char **argv)
{
	tlb2_normal(256, 2);
	return 0;
}
