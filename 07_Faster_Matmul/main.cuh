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
    if (std::fabs(matRef[i] - matOut[i]) > 0.01) return false;
  }
  return true;
}

// include SGEMM implementations
#include "01_Naive.cuh"
float run_SGEMMNaive(int N, int M, int K, float alpha, float *A, float *B,
                     float beta, float *C, bool benchmark) {
    dim3 gridDim(div_ceil(N, BLOCK_SIZE), div_ceil(K, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    if (benchmark)
        return benchmarkKernel([&]() {
            sgemm_naive<<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
        }, WARMUP_RUNS, BENCHMARK_RUNS) / 1000.0f;  // seconds
    else
        sgemm_naive<<<gridDim, blockDim>>>(N, M, K, alpha, A, B, beta, C);
        return -1.0f;
}

cudaFunctionMap_t cudaFunctionMap = {
    {"run_SGEMMNaive", run_SGEMMNaive}
};