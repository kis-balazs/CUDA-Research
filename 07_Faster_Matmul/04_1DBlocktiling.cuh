#pragma once

#include "../00_Utils/cudaUtils.cuh"

// A(N x M) @ B(M x K) = C(N x K)

template <const uint BN, const uint BK, const uint BM, const uint TN>
__global__ void sgemm1DBLocktiling(int N, int M, int K, float alpha, float *A, float *B, float beta, float *C) {
    // flipping x & y gets ~30% less performance. As it is, it
    // ensures that blocks with sequential blockIDs access columns
    // of B sequentially, while sharing the same row of A.
    // Slower: share columns of A, access in B non-seq.
    // Faster: better spatial locality --> better L2 hit rate
    uint x = blockIdx.y;
    uint y = blockIdx.x;

    uint thrX = threadIdx.x / BK;
    uint thrY = threadIdx.x % BK;

    // current blocktile in shmem
    __shared__ float ASh[BN * BM];
    __shared__ float BSh[BM * BK];

    // pointer moving
    A += x * BN * M;
    B += y * BK;
    C += x * BN * K + y * BK;

    assert(BN * BM == blockDim.x);
    assert(BM * BK == blockDim.x);  // simmetry

    uint innerXA = threadIdx.x / BM;  // warp-level GMEM coalescing
    uint innerYA = threadIdx.x % BM;
    uint innerXB = threadIdx.x / BK;
    uint innerYB = threadIdx.x % BK;

    // thread-local cache for results in registerfile
    float thrRes[TN] = {0.0f};
    // blocktile loop
    for (uint blkIdx = 0; blkIdx < M; blkIdx += BM) {
        ASh[innerXA * BM + innerYA] = A[innerXA * M + innerYA];
	BSh[innerXB * BK + innerYB] = B[innerXB * K + innerYB];
	__syncthreads();

	// blocktile advance
	A += BM;
	B += BM * K;

	// per-thread result
	for (uint dotIdx = 0; dotIdx < BM; dotIdx++) {
	    float tmpB = BSh[dotIdx * BK + thrY];
	    for (uint resIdx = 0; resIdx < TN; resIdx++) {
	        thrRes[resIdx] += ASh[(thrX * TN + resIdx) * BM + dotIdx] * tmpB;
	    }
	}
	__syncthreads();
    }
    for (uint resIdx = 0; resIdx < TN; resIdx++) {
	C[(thrX * TN + resIdx) * K + thrY] = alpha * thrRes[resIdx] + beta * C[(thrX * TN + resIdx) * K + thrY];
    }
}
