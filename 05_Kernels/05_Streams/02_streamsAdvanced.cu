#include <iostream>
#include <cuda_runtime.h>

// CHECK_CUDA_ERROR definition for detailed logging

#define LEN 1000000
#define BLOCK_SIZE 256

__global__ void k1(float *data, int n) {
	int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thrIdx < n)
		data[thrIdx] *= 2;
}

__global__ void k2(float *data, int n) {
	int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thrIdx < n)
		data[thrIdx] += 1;
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t error, void *userData) {
	std::cout << "Stream callback print!" << std::endl;
}


int main() {
	float *hData, *dData;
	cudaStream_t stream1, stream2;
	cudaEvent_t event;
	std::cout << event << std::endl;

	size_t size = LEN * sizeof(float);

	cudaMallocHost(&hData, size);
	cudaMalloc(&dData, size);

	for (int i = 0; i < LEN; i++)
		hData[i] = i;

	int leastPrio, greatestPrio;
	cudaDeviceGetStreamPriorityRange(&leastPrio, &greatestPrio);
	cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPrio);
	cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPrio);

	cudaEventCreate(&event);

	cudaMemcpyAsync(dData, hData, size, cudaMemcpyHostToDevice, stream1);

	int gridSize = (LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
	k1<<<gridSize, BLOCK_SIZE, 0, stream1>>>(dData, LEN);
	
	cudaEventRecord(event, stream1);

	// simulate operation priority using events for streams
	cudaStreamWaitEvent(stream2, event, 0);

	k2<<<gridSize, BLOCK_SIZE, 0, stream2>>>(dData, LEN);

	cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0);

	
	cudaMemcpyAsync(hData, dData, size, cudaMemcpyDeviceToHost, stream2);

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);


	bool correct = true;
	for (int i = 0; i < LEN; i++) {
		float expected = i * 2 + 1;
		if (fabs(hData[i] - expected) > 1e-5) {
			correct = false;
			break;
		}
	}
	std::cout << "Results are ";
       	if (correct)
		std::cout << "correct";
	else
		std::cout << "incorrect";
	std::cout << std::endl;

	cudaFreeHost(hData); cudaFree(dData);
	cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
	cudaEventDestroy(event);

	return 0;
}
