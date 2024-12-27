#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <functional>
#include <numeric>

#define N 256
#define C 16
#define H 224
#define W 224

// CHECK_CUDA definition for detailed logging

#define CHECK_CUDNN(call) { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error in file %s, line %i: %s\n", __FILE__, __LINE__, cudnnGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA event-based timing function
float timeKernel(std::function<void()> kernel) {
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kernel();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
}

float benchmarkKernel(std::function<void()> kernel, int runsWarmup, int runsBenchmark) {
	for (int i = 0; i < runsWarmup; i++) kernel();

	std::vector<float> times;
	for (int i = 0; i < runsBenchmark; i++) times.push_back(timeKernel(kernel));

	return std::accumulate(times.begin(), times.end(), 0.0f) / runsBenchmark;
}


__global__ void tanhGpuNaive(float *i, float *o, int n) {
    int thrIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thrIdx < n)
        o[thrIdx] = tanhf(i[thrIdx]);
}

float tanhCpu(float i) {
    return tanhf(i);
}


int main() {
    const int tensorSize = N * C * H * W;

    float *hI, *hONaive, *hOCuDNN, *hOCpu;
    float *dI, *dONaive, *dOCuDNN;

    size_t sizeT = tensorSize * sizeof(float);

    hI = (float*)malloc(sizeT);
    hONaive = (float*)malloc(sizeT);
    hOCuDNN = (float*)malloc(sizeT);
    hOCpu = (float*)malloc(sizeT);

    for (int i = 0; i < tensorSize; i++) hI[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;  // [-1, 1]

    cudaMalloc(&dI, sizeT);
    cudaMalloc(&dONaive, sizeT);
    cudaMalloc(&dOCuDNN, sizeT);

    cudaMemcpy(dI, hI, sizeT, cudaMemcpyHostToDevice);

    // --- CPU Results ---
    for (int i = 0; i < tensorSize; i++) hOCpu[i] = tanhCpu(hI[i]);

    const int runsWarmup = 3;
    const int runsBenchmark = 20;

    // --- GPU Naive Results ---
    dim3 blockDim(256);
    dim3 gridDim((tensorSize + blockDim.x - 1) / blockDim.x);

    float naiveCUDATime = benchmarkKernel([&]() {
		tanhGpuNaive<<<gridDim, blockDim>>>(dI, dONaive, tensorSize);
	}, runsWarmup, runsBenchmark);
	printf("naive GPU tanh() average time: %lf ms\n", naiveCUDATime);
	cudaMemcpy(hONaive, dONaive, sizeT, cudaMemcpyDeviceToHost);

    // --- GPU cuDNN Results ---
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t inpD;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inpD));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inpD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    cudnnActivationDescriptor_t actD;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&actD));
    CHECK_CUDNN(cudnnSetActivationDescriptor(actD, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));

    const float alpha = 1.0, beta = 0.0;

    float cuDNNCUDATime = benchmarkKernel([&]() {
		cudnnActivationForward(handle, actD, &alpha, inpD, dI, &beta, inpD, dOCuDNN);
	}, runsWarmup, runsBenchmark);
	printf("cuDNN tanh() average time: %lf ms\n\n", cuDNNCUDATime);
	cudaMemcpy(hOCuDNN, dOCuDNN, sizeT, cudaMemcpyDeviceToHost);


    bool gpuNaiveCorrect = true, gpuCuDNNCorrect = true;
	for (int i = 0; i < tensorSize; i++) {
		if (fabs(hOCpu[i] - hONaive[i]) > 1e-5) {
			gpuNaiveCorrect = false;
			break;
		}
		if (fabs(hOCpu[i] - hOCuDNN[i]) > 1e-5) {
			gpuCuDNNCorrect = false;
			break;
		}
	}
	printf("naive GPU tanh() results are %s\n", gpuNaiveCorrect ? "correct" : "incorrect");
	printf("cuDNN tanh() results are %s\n", gpuCuDNNCorrect ? "correct" : "incorrect");

    CHECK_CUDNN(cudnnDestroy(handle));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inpD));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(actD));
    cudaFree(dI); cudaFree(dONaive); cudaFree(dOCuDNN);
    return 0;
}