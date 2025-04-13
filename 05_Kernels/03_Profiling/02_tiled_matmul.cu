#include <cuda_runtime.h>
#include <stdlib.h>

#define LEN 32
#define TILE_SIZE 32

void initMat(float *MAT, int r, int c) {
    for (int i = 0; i < r * c; i++) MAT[i] = (float)rand() / RAND_MAX;
}

// A(n x m) @ B(m x k) = C(n x k)
__global__ void mulMatsOptimized(float *A, float *B, float *C, int n, int m, int k) {
	__shared__ float sharedA[TILE_SIZE][TILE_SIZE];
	__shared__ float sharedB[TILE_SIZE][TILE_SIZE];

	int blockX = blockIdx.x, blockY = blockIdx.y;
	int thrX = threadIdx.x, thrY = threadIdx.y;

	int row = blockY * TILE_SIZE + thrY;
	int col = blockX * TILE_SIZE + thrX;

	float sum = 0;

	for (int tile = 0; tile < (m + TILE_SIZE - 1) / TILE_SIZE; tile++) {
		if (row < n && tile * TILE_SIZE + thrX < m)
			sharedA[thrY][thrX] = A[row * m + tile * TILE_SIZE + thrX];
		else
			sharedA[thrY][thrX] = 0;

		if (col < k && tile * TILE_SIZE + thrY < m)
			sharedB[thrY][thrX] = B[(tile * TILE_SIZE + thrY) * k + col];
		else
			sharedB[thrY][thrX] = 0;

		__syncthreads();
		
		for (int _m = 0; _m < TILE_SIZE; _m++)
			sum += sharedA[thrY][_m] * sharedB[_m][thrX];

		__syncthreads();
	}

	if (row < n && col < k)
		C[row * m + col] = sum;
}

int main() {
	int N = LEN, M = LEN, K = LEN;

	float *A, *B, *C;
	float *dA, *dB, *dC;
	
	size_t sizeA = N * M * sizeof(float);
	size_t sizeB = M * K * sizeof(float);
	size_t sizeC = N * K * sizeof(float);

	A = (float*)malloc(sizeA);
	B = (float*)malloc(sizeB);
	C = (float*)malloc(sizeC);

	cudaMalloc(&dA, sizeA);
	cudaMalloc(&dB, sizeB);
	cudaMalloc(&dC, sizeC);

	srand(time(NULL));
	initMat(A, N, M);
	initMat(B, M, K);


	cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

	dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    mulMatsOptimized<<<gridDim, blockDim>>>(dA, dB, dC, N, M, K);
	cudaDeviceSynchronize();

	cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	free(A); free(B); free(C);

	return 0;
}
