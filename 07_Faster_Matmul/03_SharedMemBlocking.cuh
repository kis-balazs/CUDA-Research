#pragma once

#include "../00_Utils/cudaUtils.cuh"

// A(N x M) @ B(M x K) = C(N x K)

template <const uint BLOCKSIZE>
__global__ void sgemmSharedMemBlocking(int N, int M, int K, float alpha, float *A, float *B, float beta, float *C) {
    // output block to compute in this threadBlock
    uint x = blockIdx.x;
    uint y = blockIdx.y;

    // buffer for current block in shmem
    __shared__ float ASh[BLOCKSIZE * BLOCKSIZE];
    __shared__ float BSh[BLOCKSIZE * BLOCKSIZE];

    // inside current thread
    uint thrX = threadIdx.x / BLOCKSIZE;
    uint thrY = threadIdx.x % BLOCKSIZE;

    // pointers to starting pos in global mat
    A += x * BLOCKSIZE * M;  		     // row = x, col = 0
    B += y * BLOCKSIZE;      		     // row = 0, col = y
    C += x * BLOCKSIZE * K + y * BLOCKSIZE;  // row = x, col = y

    float tmp = 0.0f;
    for (int blkId = 0; blkId < M; blkId += BLOCKSIZE) {
	// load elems, and use thrX as consecutive index for shmem coalescing
	ASh[thrX * BLOCKSIZE + thrY] = A[thrX * M + thrY];
	BSh[thrX * BLOCKSIZE + thrY] = B[thrX * K + thrY];

	// block threads in this block until cache is full
	__syncthreads();
	A += BLOCKSIZE;
	B += BLOCKSIZE * K;

	for (int i = 0; i < BLOCKSIZE; i++)
	    tmp += ASh[thrX * BLOCKSIZE + i] * BSh[i * BLOCKSIZE + thrY];

	// sync to block faster threads from fetching next block from in front of slower ones
	__syncthreads();
    }
    C[thrX * K + thrY] = alpha * tmp + beta * C[thrX * K + thrY];
}
