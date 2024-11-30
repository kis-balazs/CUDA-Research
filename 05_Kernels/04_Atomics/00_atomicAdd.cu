#include <stdio.h>
#include <cuda_runtime.h>

#define N_THREADS 100
#define N_BLOCKS 100

__global__ void incrCntNonAtomic(int *counter) {
	// not locking mutex
	int old = *counter;
	*counter = old + 1;
	// not unlocking mutex
	
	// normally, old value is returned, e.g., on CAS operations
}

__global__ void incrCntAtomic(int *counter) {
	atomicAdd(counter, 1);
}


int main() {
	int h_cntNonAtomic = 0;
	int h_cntAtomic = 0;
	int *d_cntNonAtomic, *d_cntAtomic;

	size_t size = sizeof(int);

	cudaMalloc(&d_cntNonAtomic, size);
	cudaMalloc(&d_cntAtomic, size);

	cudaMemcpy(d_cntNonAtomic, &h_cntNonAtomic, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cntAtomic, &h_cntAtomic, size, cudaMemcpyHostToDevice);

	incrCntNonAtomic<<<N_BLOCKS, N_THREADS>>>(d_cntNonAtomic);
	incrCntAtomic<<<N_BLOCKS, N_THREADS>>>(d_cntAtomic);

	cudaMemcpy(&h_cntNonAtomic, d_cntNonAtomic, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_cntAtomic, d_cntAtomic, size, cudaMemcpyDeviceToHost);

	printf("counter [nonAtomic]: \t%d\n", h_cntNonAtomic);
	printf("counter [atomic]: \t%d\n", h_cntAtomic);

	cudaFree(d_cntNonAtomic); cudaFree(d_cntAtomic);

	return 0;
}
