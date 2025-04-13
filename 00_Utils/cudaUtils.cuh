#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <functional>
#include <numeric>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA event-based timing function
float timeKernel(std::function<void()> kernel);
float benchmarkKernel(std::function<void()> kernel, int runsWarmup, int runsBenchmark);

// Usage example:
/*
float cudaTime = benchmarkKernel([&]() {
    kernel<<<gridDim, blockDim>>>(PARAMS);
}, runsWarmup, runsBenchmark);
printf("kernel() average time: %lf ms\n", cudaTime);
*/

void printCudaDeviceInfo();