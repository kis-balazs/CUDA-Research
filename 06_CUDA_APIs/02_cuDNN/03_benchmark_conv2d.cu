#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <functional>
#include <numeric>

#define N 4
#define CIn 1
#define H 32
#define W 32
#define COut 4
#define kernelSize 3

// CHECK_CUDA definition for detailed logging
// CHECK_CUDNN definition for detailed logging

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


// multi-channel 2D convolution kernel
__global__ void conv2DNaive(float *i, float *kernel, float *o, int w, int h, int cI, int cO, int kSize, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outC = blockIdx.z % cO;
    int batchIdx = blockIdx.z / cO;

    float sum;
    int kHalf = kSize / 2;

    // 3rd logical relation might be redundant, given the modulo op in the calculation
    if (x < w && y < h && outC < cO && batchIdx < n) {
        sum = 0.0;
        for (int inC = 0; inC < cI; inC++) {
            for (int kY = -kHalf; kY <= kHalf; kY += 1) { // += stride
                for (int kX = -kHalf; kX <= kHalf; kX += 1) { // see above
                    int iX = x + kX;
                    int iY = y + kY;

                    if (iY >= 0 && iY < h && iX >= 0 && iX < w) {
                        int inputIdx = ((batchIdx * cI + inC) * h + iY) * w + iX;
                        int kernelIdx = ((outC * cI + inC) * kSize + (kY + kHalf)) * kSize + (kX + kHalf);
                        sum += i[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }
        int outputIdx = ((batchIdx * cO + outC) * h + y) * w + x;
        o[outputIdx] = sum;
    }
}


void printMat(int n, int c, int h, int w, float* m) {
    for (int b = 0; b < n; b++) {
        for (int _c = 0; _c < c; _c++) {
            printf("Channel %d:\n", _c);
            for (int _h = 0; _h < h; _h++) {
                for (int _w = 0; _w < w; _w++) {
                    int idx = ((b * c + _c) * h + _h) * w + _w;
                    printf("%f ", m[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    const int inSize = N * CIn * W * H;
    const int outSize = N * COut * W * H;
    const int kernelElements = CIn * kernelSize * kernelSize * COut;

    printf("Image size: %d x kernel size %d --> output size %d\n", inSize, kernelElements, outSize);

    float *hI, *hK, *hONaive, *hOCuDNN;
    float *dI, *dK, *dONaive, *dOCuDNN;
    
    size_t sizeInput = inSize * sizeof(float);
    size_t sizeKernel = kernelElements * sizeof(float);
    size_t sizeOutput = outSize * sizeof(float);

    hI = (float*)malloc(sizeInput);
    hK = (float*)malloc(sizeKernel);
    hONaive = (float*)malloc(sizeOutput);
    hOCuDNN = (float*)malloc(sizeOutput);

    for (int i = 0; i < sizeInput; i++) hI[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;  // [-1, 1]
    for (int i = 0; i < sizeKernel; i++) hK[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;  // [-1, 1]

    cudaMalloc(&dI, sizeInput);
    cudaMalloc(&dK, sizeKernel);
    cudaMalloc(&dONaive, sizeOutput);
    cudaMalloc(&dOCuDNN, sizeOutput);

    cudaMemcpy(dI, hI, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, sizeKernel, cudaMemcpyHostToDevice);

    const int runsWarmup = 3;
    const int runsBenchmark = 20;

    // --- Naive GPU Results ---
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);
    
    float naiveCUDATime = benchmarkKernel([&]() {
		conv2DNaive<<<gridDim, blockDim>>>(dI, dK, dONaive, W, H, CIn, COut, kernelSize, N);
	}, runsWarmup, runsBenchmark);
	printf("naive conv2d() average time: %lf ms\n", naiveCUDATime);
	cudaMemcpy(hONaive, dONaive, sizeOutput, cudaMemcpyDeviceToHost);
    // printMat(N, COut, H, W, hONaive);

    // --- cuDNN GPU Results ---
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t inD, outD;
    cudnnFilterDescriptor_t kD;
    cudnnConvolutionDescriptor_t convD;

    cudnnCreateTensorDescriptor(&inD);
    cudnnCreateTensorDescriptor(&outD);
    cudnnCreateFilterDescriptor(&kD);
    cudnnCreateConvolutionDescriptor(&convD);

    cudnnSetTensor4dDescriptor(inD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, CIn, H, W);
    cudnnSetTensor4dDescriptor(outD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, COut, H, W);
    cudnnSetFilter4dDescriptor(kD, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, COut, CIn, kernelSize, kernelSize);
    cudnnSetConvolution2dDescriptor(convD, kernelSize / 2, kernelSize / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // ! find fastest cuDNN algorithm
    int reqAlgo = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int retAlgo;
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    cudnnGetConvolutionForwardAlgorithm_v7(handle, inD, kD, convD, outD, reqAlgo, &retAlgo, perfResults);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; // Default algorithm

    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(handle, inD, kD, convD, outD, algo, &workspaceSize);

    void* dWS;
    cudaMalloc(&dWS, workspaceSize);

    const float alpha = 1.0f, beta = 0.0f;

    float cuDNNCUDATime = benchmarkKernel([&]() {
		cudnnConvolutionForward(handle, &alpha, inD, dI, kD, dK, convD, algo, dWS, workspaceSize, &beta, outD, dOCuDNN);
	}, runsWarmup, runsBenchmark);
	printf("cuDNN conv2d() average time: %lf ms\n", cuDNNCUDATime);
	cudaMemcpy(hOCuDNN, dOCuDNN, sizeOutput, cudaMemcpyDeviceToHost);
    // printMat(N, COut, H, W, hOCuDNN);


    float maxDiff = 0.0f;
    for (int i = 0; i < outSize; i++) {
        float diff = fabs(hONaive[i] - hOCuDNN[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    printf("Max difference between cuDNN and naive kernel: %e\n", maxDiff);

    cudnnDestroyTensorDescriptor(inD);
    cudnnDestroyTensorDescriptor(outD);
    cudnnDestroyFilterDescriptor(kD);
    cudnnDestroyConvolutionDescriptor(convD);
    cudnnDestroy(handle);

    cudaFree(dI); cudaFree(dK); cudaFree(dONaive); cudaFree(dOCuDNN);
    cudaFree(dWS);
    return 0;
}