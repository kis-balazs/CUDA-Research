#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>


#define N 256
#define M 512
#define K 256
#define BLOCK_SIZE 32  // very nice alignment with the multiplication power

double getTime() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void initMat(float *MAT, int r, int c) {
    for (int i = 0; i < r * c; i++) MAT[i] = (float)rand() / RAND_MAX;
}

// Multiply matrices A and B
// A(2x3) @ B(3x4) = C(2x4) --> (N x M) @ (M x K) = (N x K)

void mulMatsCpu(float *A, float *B, float *C, int n, int m, int k) {
	float sum;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			sum = 0;
			for (int l = 0; l < m; l++)
				sum += A[i * m + l] * B[l * k + j];
			C[i * k + j] = sum;
		}
	}
}

__global__ void mulMatsGpu(float *A, float *B, float *C, int n, int m, int k) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < n && col < k) {
		float sum = 0;
		for (int l = 0; l < m; l++)
			sum += A[row * m + l] * B[l * k + col];
		C[row * k + col] = sum;
	}
}

int main() {
	float *hA, *hB, *hC_cpu, *hC_gpu;
	float *dA, *dB, *dC;

	int sizeA = N * M * sizeof(float);
	int sizeB = M * K * sizeof(float);
	int sizeC = N * K * sizeof(float);

	double cpuTime, gpuTime;
	double startTime;

	// alloc host mem
	hA = (float*)malloc(sizeA);
	hB = (float*)malloc(sizeB);
	hC_cpu = (float*)malloc(sizeC);
	hC_gpu = (float*)malloc(sizeC);

	srand(time(NULL));
	initMat(hA, N, M);
	initMat(hB, M, K);

	// alloc device mem
	cudaMalloc(&dA, sizeA);
	cudaMalloc(&dB, sizeB);
	cudaMalloc(&dC, sizeC);

	// copy data to device
	cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

	
	// define grid and block dims
	dim3 gridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // N x K
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	printf("warm-up runs...\n");
	for (int i = 0; i < 3; i++) {
		mulMatsCpu(hA, hB, hC_cpu, N, M, K);
		mulMatsGpu<<<gridDim, blockDim>>>(dA, dB, dC, N, M, K);
		cudaDeviceSynchronize();
	}

	printf("benchmark CPU...\n");
	cpuTime = 0.0;
	for (int i = 0; i < 20; i++) {
		startTime = getTime();
		mulMatsCpu(hA, hB, hC_cpu, N, M, K);
		cpuTime += (getTime() - startTime);
	}
	cpuTime /= 20;  // average

	printf("benchmark GPU...\n");
	gpuTime = 0.0;
	for (int i = 0; i < 20; i++) {
		cudaMemset(dC, 0, sizeC);
		startTime = getTime();
		mulMatsGpu<<<gridDim, blockDim>>>(dA, dB, dC, N, M, K);
		cudaDeviceSynchronize();
		gpuTime += (getTime() - startTime);
	}
	gpuTime /= 20;

	cudaMemcpy(hC_gpu, dC, sizeC, cudaMemcpyDeviceToHost);
	bool correct = true;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			if (fabs(hC_cpu[i * N + j] - hC_gpu[i * N + j]) > 1e-4) {
				correct = false;
				break;
			}
		}
	}
	printf("Results are %s\n", correct ? "correct" : "incorrect");

	printf("CPU avg runtime: %f ms\n", cpuTime * 1000);
	printf("GPU avg runtime: %f ms\n", gpuTime * 1000);

	printf("speed-up: %.2fx\n", cpuTime / gpuTime);

	// free mem
	free(hA); free(hB); free(hC_cpu); free(hC_gpu);
	cudaFree(dA); cudaFree(dB); cudaFree(dC);

	return 0;
}
