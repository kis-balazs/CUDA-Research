#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>


#define LEN 100000000
#define BLOCK_SIZE_1D 1024  // threads in block
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8  // 1024 = 16 * 8 * 8 threads in 3dimensional structure

double getTime() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void initVector(float *a, int n) {
        for (int i = 0; i < n; i++) a[i] = (float)rand() / RAND_MAX;
}


// add vectors A and B
void addVectorsCpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void addVectorsGpu1D(float *a, float *b, float *c, int n) {
	int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;  // uni-dimensional global index of thread

	if (thrIdx < n)
		c[thrIdx] = a[thrIdx] + b[thrIdx];
}

__global__ void addVectorsGpu3D(float *a, float *b, float *c, int nX, int nY, int nZ) {
	int thrIdxX = blockIdx.x * blockDim.x + threadIdx.x;  // index of thread on #-axis
	int thrIdxY = blockIdx.y * blockDim.y + threadIdx.y;
	int thrIdxZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (thrIdxX < nX && thrIdxY < nY && thrIdxZ < nZ) {
		int thrIdx = thrIdxX + thrIdxY * nX + thrIdxZ * nX * nY; // 3D global thread index
		if (thrIdx < nX * nY * nZ)
			c[thrIdx] = a[thrIdx] + b[thrIdx];
	}
	// the problem is not the parallel operation, but more the overhead of the atomic operations create (mem allocations, multiplications, additions, etc.)
	// forcing dimension abstraction can backfire!
}

int main() {
	float *hA, *hB, *hC_cpu, *hC_gpu1D, *hC_gpu3D;
	float *dA, *dB, *dC1D, *dC3D;

	size_t size = LEN * sizeof(float);

	double cpuTime, gpuTime1D, gpuTime3D;
	double startTime;

	// alloc host mem
	hA = (float*)malloc(size);
	hB = (float*)malloc(size);
	hB = (float*)malloc(size);
	hC_cpu = (float*)malloc(size);
	hC_gpu1D = (float*)malloc(size);
	hC_gpu3D = (float*)malloc(size);

	srand(time(NULL));
	initVector(hA, LEN);
	initVector(hB, LEN);

	// alloc device mem
	cudaMalloc(&dA, size);
	cudaMalloc(&dB, size);
	cudaMalloc(&dC1D, size);
	cudaMalloc(&dC3D, size);

	// copy data to device
	cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);


	// define grid and block dims
	int numBlocks1D = (LEN + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;  // integer division and round-up for proper mem allocation

	int nX = 1000, nY = 100, nZ = 100;  // dividie LEN into three-dimensional configuration
	dim3 blockSize3D(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
	dim3 numBlocks3D(
		(nX + BLOCK_SIZE_3D_X - 1) / BLOCK_SIZE_3D_X,
		(nY + BLOCK_SIZE_3D_Y - 1) / BLOCK_SIZE_3D_Y,
		(nZ + BLOCK_SIZE_3D_Z - 1) / BLOCK_SIZE_3D_Z
	);

	printf("warm-up runs...\n");
	for (int i = 0; i < 3; i++) {
		addVectorsCpu(hA, hB, hC_cpu, LEN);
		addVectorsGpu1D<<<numBlocks1D, BLOCK_SIZE_1D>>>(dA, dB, dC1D, LEN);
		addVectorsGpu3D<<<numBlocks3D, blockSize3D>>>(dA, dB, dC3D, nX, nY, nZ);
		cudaDeviceSynchronize();
	}

	printf("benchmark CPU...\n");
	cpuTime = 0.0;
	for (int i = 0; i < 20; i++) {
		startTime = getTime();
		addVectorsCpu(hA, hB, hC_cpu, LEN);
		cpuTime += (getTime() - startTime);
	}
	cpuTime /= 20;  // average

	printf("benchmark GPU1D...\n");
	gpuTime1D = 0.0;
	for (int i = 0; i < 20; i++) {
		cudaMemset(dC1D, 0, size);
		startTime = getTime();
		addVectorsGpu1D<<<numBlocks1D, BLOCK_SIZE_1D>>>(dA, dB, dC1D, LEN);
		cudaDeviceSynchronize();
		gpuTime1D += (getTime() - startTime);
	}
	gpuTime1D /= 20;

	cudaMemcpy(hC_gpu1D, dC1D, size, cudaMemcpyDeviceToHost);
	bool correct1D = true;
	for (int i = 0; i < LEN; i++) {
		if (fabs(hC_cpu[i] - hC_gpu1D[i]) > 1e-5) {
			correct1D = false;
			break;
		}
	}
	printf("1D results are: %s\n", correct1D ? "correct" : "incorrect");

	printf("benchmark GPU3D...\n");
	gpuTime3D = 0.0;
	for (int i = 0; i < 20; i++) {
		cudaMemset(dC3D, 0, size);
		startTime = getTime();
		addVectorsGpu3D<<<numBlocks3D, blockSize3D>>>(dA, dB, dC3D, nX, nY, nZ);
		cudaDeviceSynchronize();
		gpuTime3D += (getTime() - startTime);
	}
	gpuTime3D /= 20;

	cudaMemcpy(hC_gpu3D, dC3D, size, cudaMemcpyDeviceToHost);
        bool correct3D = true;
        for (int i = 0; i < LEN; i++) {
                if (fabs(hC_cpu[i] - hC_gpu3D[i]) > 1e-5) {
                        correct3D = false;
                        break;
                }
        }
        printf("3D results are: %s\n", correct3D ? "correct" : "incorrect");

	printf("CPU avg runtime: %f ms\n", cpuTime * 1000);
	printf("GPU1D avg runtime: %f ms\n", gpuTime1D * 1000);
	printf("GPU3D avg runtime: %f ms\n", gpuTime3D * 1000);

	printf("speed-up CPU - GPU1D: %.2fx\n", cpuTime / gpuTime1D);
	printf("speed-up CPU - GPU3D: %.2fx\n", cpuTime / gpuTime3D);
	printf("speed-up GPU1D - GPU3D: %.2fx\n", gpuTime1D / gpuTime3D);


	// free mem
	free(hA); free(hB); free(hC_cpu); free(hC_gpu1D); free(hC_gpu3D);
	cudaFree(dA); cudaFree(dB); cudaFree(dC1D); cudaFree(dC3D);

	return 0;
}
