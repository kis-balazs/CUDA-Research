#pragma once

#include "../00_Utils/cudaUtils.cuh"

// A(N x M) @ B(M x K) = C(N x K)

template <const uint BN, const uint BK, const uint BM, const uint TN>
__global__ void sgemm2DBLocktiling(int N, int M, int K, float alpha, float *A, float *B, float beta, float *C) {
    // brainfart... idea is to create a 2^x * 2^x grid per thread, and use this to populate shared memory; play with this to get result
}