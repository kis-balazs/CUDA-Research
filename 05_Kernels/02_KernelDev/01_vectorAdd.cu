#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define LEN 1000000
#define BLOCK_SIZE 256  // how many threads are in the block

double getTime() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void initVector(float *a, int n) {
        for (int i = 0; i < n; i++) a[i] = (float)rand() / RAND_MAX;
}


// Add vectors A and B
void addVectorsCpu(float *a, float *b, float *c, int n) {
	for (int i = 0; i < n; i++)
		c[i] = a[i] + b[i];
}

__global__ void addVectorsGpu(float *a, float *b, float *c, int n) {
	int thrIdx = blockIdx.x * blockDim.x + threadIdx.x; // get "global" index of thread
	if (thrIdx < n)
		c[thrIdx] = a[thrIdx] + b[thrIdx];
}

int main() {
	float *hA, *hB, *hC_cpu, *hC_gpu;
	float *dA, *dB, *dC;

	size_t size = LEN * sizeof(float);

	double cpuTime, gpuTime;
	double startTime;

	// alloc host mem
	hA = (float*)malloc(size);
	hB = (float*)malloc(size);
	hC_cpu = (float*)malloc(size);
	hC_gpu = (float*)malloc(size);

	srand(time(NULL));
	initVector(hA, LEN);
	initVector(hB, LEN);

	// alloc device mem
	cudaMalloc(&dA, size);
	cudaMalloc(&dB, size);
	cudaMalloc(&dC, size);

	// copy data to device
	cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);


	// define grid and block dims
	int numBlocks = (LEN + BLOCK_SIZE - 1) / BLOCK_SIZE; // integer division and round-up for proper mem allocation

	printf("warm-up runs...\n");
	for (int i = 0; i < 3; i++) {
		addVectorsCpu(hA, hB, hC_cpu, LEN);
		addVectorsGpu<<<numBlocks, BLOCK_SIZE>>>(dA, dB, dC, LEN);
		cudaDeviceSynchronize();
	}

	printf("benchmark CPU...\n");
	cpuTime = 0.0;
	for (int i = 0; i < 20; i++) {
		startTime = getTime();
		addVectorsCpu(hA, hB, hC_cpu, LEN);
		cpuTime += (getTime() - startTime);
	}
	cpuTime /= 20; // average

	printf("benchmark GPU...\n");
	gpuTime = 0.0;
	for (int i = 0; i < 20; i++) {
		cudaMemset(dC, 0, size);
		startTime = getTime();
		addVectorsGpu<<<numBlocks, BLOCK_SIZE>>>(dA, dB, dC, LEN);
		cudaDeviceSynchronize();
		gpuTime += (getTime() - startTime);
	}
	gpuTime /= 20;

	printf("CPU avg runtime: %f ms\n", cpuTime * 1000);
	printf("GPU avg runtime: %f ms\n", gpuTime * 1000);

	printf("speed-up: %.2fx\n", cpuTime / gpuTime);

	cudaMemcpy(hC_gpu, dC, size, cudaMemcpyDeviceToHost);
	bool correct = true;
	for (int i = 0; i < LEN; i++) {
		if (fabs(hC_cpu[i] - hC_gpu[i]) > 1e-5) {
			correct = false;
			break;
		}
	}
	printf("Results are %s\n", correct ? "correct" : "incorrect");

	// free mem
	free(hA); free(hB); free(hC_cpu); free(hC_gpu);
	cudaFree(dA); cudaFree(dB); cudaFree(dC);

	return 0;
}
