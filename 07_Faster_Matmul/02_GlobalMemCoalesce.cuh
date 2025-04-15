#pragma once

#include "../00_Utils/cudaUtils.cuh"

// A(N x M) @ B(M x K) = C(N x K)

template <const uint BLOCKSIZE>
__global__ void sgemmGlobalMemCoalesce(int N, int M, int K, float alpha, float *A, float *B, float beta, float *C) {
    uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < N && y < K) {
        float tmp = 0.0f;
        for (int m = 0; m < M; m++)
            tmp += A[x * M + m] * B[m * K + y];
        // C = alpha * (A @ B) + beta * C
        C[x * K + y] = alpha * tmp + beta * C[x * K + y];
    }
}
