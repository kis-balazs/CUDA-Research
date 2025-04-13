#pragma once

#include "../00_Utils/cudaUtils.cuh"

// Multiply matrices A and B
// A(2x3) @ B(3x4) = C(2x4) --> (N x M) @ (M x K) = (N x K)

__global__ void sgemm_naive(int N, int M, int K, float alpha, float *A, float *B, float beta, float *C) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < K) {
        float tmp = 0.0f;
        for (int m = 0; m < M; m++)
            tmp += A[x * M + m] * B[m * K + y];
        // C = alpha * (A @ B) + beta * C
        C[x * K + y] = alpha * tmp + beta * C[x * K + y];
    }
}