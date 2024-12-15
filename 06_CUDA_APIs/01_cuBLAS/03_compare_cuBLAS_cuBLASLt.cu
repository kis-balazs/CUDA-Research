#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <functional>
#include <vector>
#include <numeric>


#define N 512
#define M 256
#define K 512

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
	
	float C[N * K];
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

	const float alpha = 1.0f, beta = 0.0f; // comes from the generic matmul operation being constructed to mimic linear layer forward
	const float alphah = __float2half(1.0f), betah = __float2half(0.0f);

	const int runsWarmup = 3;
	const int runsBenchmark = 20;

	// -----
       	// naive GPU matrix multiplication
	dim3 blockDim(32, 32);
	dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
	float naiveCUDATime = benchmarkKernel([&]() {
		mulMatsGpu<<<gridDim, blockDim>>>(dA, dB, dC, N, M, K);
	}, runsWarmup, runsBenchmark);
	printf("naive matmul average time: %lf ms\n", naiveCUDATime);
	cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

	// -----
	// cuBLAS FP32
	float cuBLASFp32Time = benchmarkKernel([&]() {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, dB, K, dA, M, &beta, dC, K);
	}, runsWarmup, runsBenchmark);
	printf("cuBLAS FP32 average time: %lf ms\n", cuBLASFp32Time);
	cudaMemcpy(CcuBLASFp32, dC, sizeC, cudaMemcpyDeviceToHost);
	
	// -----
	// cuBLASLt FP32
	// set up matrix & multiplication descriptors for float32
	cublasLtMatrixLayout_t lA, lB, lC;
	cublasLtMatrixLayoutCreate(&lA, CUDA_R_32F, M, N, M);
	cublasLtMatrixLayoutCreate(&lB, CUDA_R_32F, K, M, K);
	cublasLtMatrixLayoutCreate(&lC, CUDA_R_32F, M, N, M);

	cublasLtMatmulDesc_t mmDesc;
	cublasLtMatmulDescCreate(&mmDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

	float cuBLASLtFp32Time = benchmarkKernel([&]() {
		cublasLtMatmul(handleLt, mmDesc, &alpha, dB, lB, dA, lA, &beta, dC, lC, dC, lC, NULL, NULL, 0, 0);
	}, runsWarmup, runsBenchmark);
	printf("cuBLASLt FP32 average time: %lf ms\n", cuBLASLtFp32Time);
	cudaMemcpy(CcuBLASLtFp32, dC, sizeC, cudaMemcpyDeviceToHost);

	printf("\n\n");
	bool cuBLASFp32Correct = true, cuBLASLtFp32Correct = true;
	for (int i = 0; i < N * K; i++) {
		if (fabs(C[i] - CcuBLASFp32[i]) > 1e-4) {
			cuBLASFp32Correct = false;
			break;
		}
		if (fabs(C[i] - CcuBLASLtFp32[i]) > 1e-4) {
			cuBLASLtFp32Correct = false;
			break;
		}
	}
	printf("cuBLAS FP32 results are %s\n", cuBLASFp32Correct ? "correct" : "incorrect");
	printf("cuBLAS-Lt FP32 results are %s\n", cuBLASLtFp32Correct ? "correct" : "incorrect");

	return 0;
}
