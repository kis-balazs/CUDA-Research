#include <iostream>
#include <cuda_runtime.h>

// CHECK_CUDA_ERROR definition for detailed logging

#define LEN 1000000
#define BLOCK_SIZE 256

__global__ void k1(float *data, int n) {
	int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thrIdx < n)
		data[thrIdx] *= 2;
}

__global__ void k2(float *data, int n) {
	int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thrIdx < n)
		data[thrIdx] += 1;
}


