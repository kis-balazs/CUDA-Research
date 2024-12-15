#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <functional>
#include <vector>
#include <numeric>


#define N 2048
#define M 1024
#define K 2048

// CHECK_CUDA_ERROR definition for detailed logging

void initMat(float *MAT, int r, int c) {
    for (int i = 0; i < r * c; i++) MAT[i] = (float)rand() / RAND_MAX;
}

// CUDA event-based timing function
float timeKernel(std::function<void()> kernel) {
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kernel();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
}

float benchmarkKernel(std::function<void()> kernel, int runsWarmup, int runsBenchmark) {
	for (int i = 0; i < runsWarmup; i++) kernel();

	std::vector<float> times;
	for (int i = 0; i < runsBenchmark; i++) times.push_back(timeKernel(kernel));

	return std::accumulate(times.begin(), times.end(), 0.0f) / runsBenchmark;
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
	float A[N * M];
	float B[M * K];

	initMat(A, N, M);
	initMat(B, M, K);

	float CcuBLASFp32[N * K], CcuBLASFp16[N * K];
	float CcuBLASLtFp32[N * K], CcuBLASLtFp16[N * K];

	size_t sizeA = N * M * sizeof(float);
	size_t sizeB = M * K * sizeof(float);
	size_t sizeC = N * K * sizeof(float);
	
	size_t sizeAh = N * M * sizeof(half);
	size_t sizeBh = M * K * sizeof(half);
	size_t sizeCh = N * K * sizeof(half);

	float *dA, *dB, *dC;
	cudaMalloc(&dA, sizeA);
	cudaMalloc(&dB, sizeB);
	cudaMalloc(&dC, sizeC);

	half *dAh, *dBh, *dCh;
	cudaMalloc(&dAh, sizeAh);
	cudaMalloc(&dBh, sizeBh);
	cudaMalloc(&dCh, sizeCh);

	cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

	half Ah[N * M], Bh[M * K];
	for (int i = 0; i < N * M; i++) Ah[i] = __float2half(A[i]);
	for (int i = 0; i < M * K; i++) Bh[i] = __float2half(B[i]);

	cudaMemcpy(dAh, Ah, sizeAh, cudaMemcpyHostToDevice);
	cudaMemcpy(dBh, Bh, sizeBh, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasLtHandle_t handleLt;
	cublasLtCreate(&handleLt);



	return 0;
}
