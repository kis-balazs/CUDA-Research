# pragma once

#include <map>
#include <string>
#include <stdio.h>
#include <iostream>
#include "../00_Utils/cudaUtils.cuh"

#define BLOCK_SIZE 32
#define WARMUP_RUNS 3
#define BENCHMARK_RUNS 20

std::function<void()> pcdi = printCudaDeviceInfo;
using cudaFunctionMap_t = std::map<
    std::string,
    std::function<float(int, int, int, float, float*, float*, float, float*, bool)>
>;


int div_ceil(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}

bool verify_matrix(float *matRef, float *matOut, int N) {
    for (int i = 0; i < N; i++) {
        if (std::fabs(matRef[i] - matOut[i]) > 0.01) {
	    std::cout << matRef[i] << " " << matOut[i] << std::endl;
            return false;
        }
    }
    return true;
}

// include SGEMM implementations
#include "01_Naive.cuh"
float runSgemmNaive(int N, int M, int K, float alpha, float *A, float *B,
                     float beta, float *C, bool benchmark) {
    dim3 gridDim(div_ceil(N, BLOCK_SIZE), div_ceil(K, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    if (benchmark)
        return benchmarkKernel([&]() {
            sgemmNaive<<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
        }, WARMUP_RUNS, BENCHMARK_RUNS) / 1000.0f;  // seconds
    else {
        sgemmNaive<<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
        return -1.0f;
    }
}

#include "02_GlobalMemCoalesce.cuh"
float runSgemmGlobalMemCoalesce(int N, int M, int K, float alpha, float *A, float *B,
				float beta, float *C, bool benchmark) {
    dim3 gridDim(div_ceil(N, BLOCK_SIZE), div_ceil(K, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE); // 1d layout, warp-tiling calculated in-kernel!

    if (benchmark)
        return benchmarkKernel([&]() {
	    sgemmGlobalMemCoalesce<BLOCK_SIZE>
	        <<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
	}, WARMUP_RUNS, BENCHMARK_RUNS) / 1000.0f;  // seconds
    else {
	sgemmGlobalMemCoalesce<BLOCK_SIZE>
	    <<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
	return -1.0;
    }
}

#include "03_SharedMemBlocking.cuh"
float runSgemmSharedMemBlocking(int N, int M, int K, float alpha, float *A, float *B,
		                float beta, float *C, bool benchmark) {
    dim3 gridDim(div_ceil(N, BLOCK_SIZE), div_ceil(K, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);

    // L1 cache is non-used, because only SHMEM is used; so, carve out all L1 to SMEM;
    // no diff, since occupancy is limited (register & thread count), but good practice.
    cudaFuncSetAttribute(
	sgemmSharedMemBlocking<BLOCK_SIZE>,
	cudaFuncAttributePreferredSharedMemoryCarveout,
	cudaSharedmemCarveoutMaxShared
    );

    if (benchmark)
	return benchmarkKernel([&]() {
	    sgemmSharedMemBlocking<BLOCK_SIZE>
	        <<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
	}, WARMUP_RUNS, BENCHMARK_RUNS) / 1000.0f;  // seconds
    else {
	sgemmSharedMemBlocking<BLOCK_SIZE>
            <<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
	return -1.0;
    }
}

#include "04_1DBlocktiling.cuh"
float runSgemm1DBlocktiling(int N, int M, int K, float alpha, float *A, float *B,
		            float beta, float *C, bool benchmark) {
    const uint BN = 64;
    const uint BK = 64;
    const uint BM = 8;
    const uint TN = 8;

    dim3 gridDim(div_ceil(K, BK), div_ceil(N, BN));
    dim3 blockDim((BN * BK) / TN);

    if (benchmark)
	return benchmarkKernel([&]() {
	    sgemm1DBLocktiling<BN, BK, BM, TN>
	        <<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
	}, WARMUP_RUNS, BENCHMARK_RUNS) / 1000.0f;  // seconds
    else {
	sgemm1DBLocktiling<BN, BK, BM, TN>
	    <<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
	return -1.0;
    }
}


cudaFunctionMap_t cudaFunctionMap = {
    {"runSgemmNaive", runSgemmNaive},
    {"runSgemmGlobalMemCoalesce", runSgemmGlobalMemCoalesce},
    {"runSgemmSharedMemBlocking", runSgemmSharedMemBlocking},
    {"runSgemm1DBlocktiling", runSgemm1DBlocktiling}
};
