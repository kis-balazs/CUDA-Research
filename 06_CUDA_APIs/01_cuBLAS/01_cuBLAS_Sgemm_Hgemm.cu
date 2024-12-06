#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define N 3
#define M 4
#define K 5

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

void mulMatsCpu(float *A, float *B, float *C, int n, int m, int k) {
	float sum;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			sum = 0;
			for (int l = 0; l < m; l++)
				sum += A[i * m + l] * B[l * k + j];
			C[i * k + j] = sum;
		}
	}
}

int main() {
	float A[N * M] = {1.0f};
	float B[M * K] = {2.0f};
	float CCpu[N * K], CCublasS[N * K], CCublasH[N * K];

	size_t sizeA = N * M * sizeof(float);
	size_t sizeB = M * K * sizeof(float);
	size_t sizeC = N * K * sizeof(float);

	size_t sizeAh = N * M * sizeof(half);
	size_t sizeBh = M * K * sizeof(half);
	size_t sizeCh = N * K * sizeof(half);

	mulMatsCpu(A, B, CCpu, N, M, K);

	cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

	float *dA, *dB, *dC;
	CHECK_CUDA(cudaMalloc(&dA, sizeA));
	cudaMalloc(&dB, sizeB);
	cudaMalloc(&dC, sizeC);

	CHECK_CUDA(cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice));
	cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

	// cuBLAS Sgemm --> Single Precision General Matrix Multiplication
	float alpha = 1.0f, beta = 0.0f;
	// the OP_SMTH is the matmul-required possible operations, such as identity, transpose, etc.
	// the general cublasSgemm setup looks awfully like a forward pass in a linear layer, which is pretty targeted
	// cuBLAS uses column major!
	CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, dB, K, dA, M, &beta, dC, K));
	cudaMemcpy(CCublasS, dC, sizeC, cudaMemcpyDeviceToHost);

	// cuBLAS Hgemm --> Half Precision General Matrix Multiplication
	half *dAh, *dBh, *dCh;
	CHECK_CUDA(cudaMalloc(&dAh, sizeAh));
	cudaMalloc(&dBh, sizeBh);
	cudaMalloc(&dCh, sizeCh);

	half Ah[N * M], Bh[M * K];
	for (int i = 0; i < N * M; i++) Ah[i] = __float2half(A[i]);
	for (int i = 0; i < M * K; i++) Bh[i] = __float2half(B[i]);

	CHECK_CUDA(cudaMemcpy(dAh, Ah, sizeAh, cudaMemcpyHostToDevice));
	cudaMemcpy(dBh, Bh, sizeBh, cudaMemcpyHostToDevice);

	__half alphah = __float2half(1.0f), betah = __float2half(0.0f);
	CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alphah, dBh, K, dAh, M, &betah, dCh, K));

	half Ch[N * K];
	cudaMemcpy(Ch, dCh, sizeCh, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N * K; i++) CCublasH[i] = __half2float(Ch[i]);


	bool cublasSgemmCorrect = true, cublasHgemmCorrect = true;
	for (int i = 0; i < N * K; i++) {
		if (fabs(CCpu[i] - CCublasS[i]) > 1e-4) {
			cublasSgemmCorrect = false;
			break;
		}
	}
	printf("Sgemm results are %s\n", cublasSgemmCorrect ? "correct" : "incorrect");

	for (int i = 0; i < N * K; i++) {
		if (fabs(CCpu[i] - CCublasH[i]) > 1e-4) {
			cublasHgemmCorrect = false;
			break;
		}
	}
	printf("Hgemm results are %s\n", cublasHgemmCorrect ? "correct" : "incorrect");

	cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dAh); cudaFree(dBh); cudaFree(dCh);
	CHECK_CUBLAS(cublasDestroy(handle));

	return 0;
}
