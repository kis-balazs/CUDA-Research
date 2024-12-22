#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#define LEN 10000000
#define BLOCK_SIZE 256

void initVector(float *a, int n) {
        for (int i = 0; i < n; i++) a[i] = (float)rand() / RAND_MAX;
}

__global__ void addVectors(float *a, float *b, float *c, int n) {
	int thrIdx = blockIdx.x * blockDim.x + threadIdx.x; // get "global" index of thread
	if (thrIdx < n)
		c[thrIdx] = a[thrIdx] + b[thrIdx];
}

int main() {
	float *hA, *hB, *hC;
	float *dA, *dB, *dC;

	cudaStream_t stream1, stream2;

	size_t size = LEN * sizeof(float);

	hA = (float*)malloc(size);
	hB = (float*)malloc(size);
	hC = (float*)malloc(size);

	srand(time(NULL));
	initVector(hA, LEN);
	initVector(hB, LEN);

	// all cuda functions can be wrapped in the CHECK_CUDA_ERROR to get precise debugging info
	CHECK_CUDA_ERROR(cudaMalloc(&dA, size));
	cudaMalloc(&dB, size);
	cudaMalloc(&dC, size);

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	cudaMemcpyAsync(dA, hA, size, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(dB, hB, size, cudaMemcpyHostToDevice, stream2);

	int gridSize = (LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
	addVectors<<<gridSize, BLOCK_SIZE, 0, stream1>>>(dA, dB, dC, LEN);

	cudaMemcpyAsync(hC, dC, size, cudaMemcpyDeviceToHost, stream1);

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	bool correct = true;
	for (int i = 0; i < LEN; i++) {
		if (fabs(hA[i] + hB[i] - hC[i]) > 1e-5) {
			correct = false;
			break;
		}
	}
	printf("Results are %s\n", correct ? "correct" : "incorrect");

	free(hA); free(hB); free(hC);
	cudaFree(dA); cudaFree(dB); cudaFree(dC);

	return 0;
}
