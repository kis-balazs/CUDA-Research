#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <functional>
#include <stdio.h>
#include <numeric>

#define N 512
#define M 512
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

int main() {
    float A[N * M], B[M * K];
    float CCuBLAS[N * K], CCuBLASXt[N * K];

    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * M * sizeof(float);

    initMat(A, N, M);
    initMat(B, M, K);

    const int runsWarmup = 3;
	const int runsBenchmark = 20;

    const float alpha = 1.0f, beta = 0.0f; // comes from the generic matmul operation being constructed to mimic linear layer forward
	
    // cuBLAS
    {
        float *dA, *dB, *dC;
        cudaMalloc(&dA, sizeA);
        cudaMalloc(&dB, sizeB);
        cudaMalloc(&dC, sizeC);

        cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
	    cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        float cuBLASTime = benchmarkKernel([&]() {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, dB, K, dA, M, &beta, dC, K);
        }, runsWarmup, runsBenchmark);
        printf("cuBLAS average time: %lf ms\n", cuBLASTime);
        cudaMemcpy(CCuBLAS, dC, sizeC, cudaMemcpyDeviceToHost);

        cublasDestroy(handle);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // cuBLASXt
    {
        cublasXtHandle_t handleXt;
        cublasXtCreate(&handleXt);

        int devices[1] = {0};
        cublasXtDeviceSelect(handleXt, 1, devices);

        float cuBLASXtTime = benchmarkKernel([&]() {
            cublasXtSgemm(handleXt, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, B, K, A, N, &beta, CCuBLASXt, K);
        }, runsWarmup, runsBenchmark);
        printf("cuBLASXt average time: %lf ms\n", cuBLASXtTime);

        cublasXtDestroy(handleXt);
    }

    bool match = true;
	for (int i = 0; i < N * K; i++) {
		if (fabs(CCuBLAS[i] - CCuBLASXt[i]) > 1e-4) {
			match = false;
			break;
		}
    }
    printf("cuBLAS and cuBLASXt results match: %s\n", match ? "correct" : "incorrect");

    return 0;
}