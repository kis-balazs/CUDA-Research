#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define LEN 2048
#define BLOCK_SIZE 16

void initMat(float *MAT, int n) {
    for (int i = 0; i < n * n; i++) MAT[i] = (float)rand() / RAND_MAX;
}

// multiply square matrices
__global__ void mulMats(float *A, float *B, float *C, int n) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0;
	if (row < n && col < n) {
		for(int i = 0; i < n; i++)
			sum += A[row * n + i] * B[i * n + col];
		C[row * n + col] = sum;
	}
}

void matrixMul(float *A, float *B, float *C, int n) {
	nvtxRangePush("matmul function");

	float *dA, *dB, *dC;
	
	size_t size = n * n * sizeof(float);

	nvtxRangePush("cudamalloc");
	cudaMalloc(&dA, size);
	cudaMalloc(&dB, size);
	cudaMalloc(&dC, size);
	nvtxRangePop();

	nvtxRangePush("host2device");
	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
	nvtxRangePop();

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

	nvtxRangePush("kernel exec");
	mulMats<<<gridDim, blockDim>>>(dA, dB, dC, n);
	cudaDeviceSynchronize();
	nvtxRangePop();

	nvtxRangePush("device2host");
	cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
	nvtxRangePop();

	nvtxRangePush("cudamemdealloc");
	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	nvtxRangePop();

	nvtxRangePop();  // function-level
}

int main() {
	float *A, *B, *C;
	size_t size = LEN * LEN * sizeof(float);

	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);


	srand(time(NULL));
	initMat(A, LEN);
	initMat(B, LEN);

	matrixMul(A, B, C, LEN);

	free(A); free(B); free(C);

	return 0;
}
