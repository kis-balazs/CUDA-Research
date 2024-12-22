#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>

#define N 4
#define M 4
#define K 4

// CHECK_CUDA_ERROR definition for detailed logging

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

	float CCpu[N * K], C_fp32[N * K], C_fp16[N * K];

	size_t sizeA = N * M * sizeof(float);
	size_t sizeB = M * K * sizeof(float);
	size_t sizeC = N * K * sizeof(float);
	
	size_t sizeAh = N * M * sizeof(half);
	size_t sizeBh = M * K * sizeof(half);
	size_t sizeCh = N * K * sizeof(half);

	mulMatsCpu(A, B, CCpu, N, M, K);

	float *dA, *dB, *dC;
	cudaMalloc(&dA, sizeA);
	cudaMalloc(&dB, sizeB);
	cudaMalloc(&dC, sizeC);

	half *dAh, *dBh, *dCh;
	cudaMalloc(&dAh, sizeAh);
	cudaMalloc(&dBh, sizeBh);
	cudaMalloc(&dCh, sizeCh);

	cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

	half Ah[N * M], Bh[M * K];
	for (int i = 0; i < N * M; i++) Ah[i] = __float2half(A[i]);
	for (int i = 0; i < M * K; i++) Bh[i] = __float2half(B[i]);

	cudaMemcpy(dAh, Ah, sizeAh, cudaMemcpyHostToDevice);
	cudaMemcpy(dBh, Bh, sizeBh, cudaMemcpyHostToDevice);

	cublasLtHandle_t handle;
	cublasLtCreate(&handle);

	// set up matrix & multiplication descriptors for float32
	cublasLtMatrixLayout_t lA, lB, lC;
	cublasLtMatrixLayoutCreate(&lA, CUDA_R_32F, M, N, M);
	cublasLtMatrixLayoutCreate(&lB, CUDA_R_32F, K, M, K);
	cublasLtMatrixLayoutCreate(&lC, CUDA_R_32F, M, N, M);

	cublasLtMatmulDesc_t mmDesc;
	cublasLtMatmulDescCreate(&mmDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

	// set up matrix & multiplication descriptors for float16
	cublasLtMatrixLayout_t lAh, lBh, lCh;
    cublasLtMatrixLayoutCreate(&lAh, CUDA_R_16F, M, N, M); // original NMM
    cublasLtMatrixLayoutCreate(&lBh, CUDA_R_16F, K, M, K); // see above
    cublasLtMatrixLayoutCreate(&lCh, CUDA_R_16F, M, N, M);

	cublasLtMatmulDesc_t mmDesch;
	cublasLtMatmulDescCreate(&mmDesch, CUBLAS_COMPUTE_16F, CUDA_R_16F);

	// set matrix operation for A and B
	cublasOperation_t transA = CUBLAS_OP_N;
	cublasOperation_t transB = CUBLAS_OP_N;
	cublasLtMatmulDescSetAttribute(mmDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(cublasOperation_t));
	cublasLtMatmulDescSetAttribute(mmDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(cublasOperation_t));
	cublasLtMatmulDescSetAttribute(mmDesch, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(cublasOperation_t));
	cublasLtMatmulDescSetAttribute(mmDesch, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(cublasOperation_t));

	const float alpha = 1.0, beta = 0.0f;  // comes from the generic matmul operation being constructed to mimic linear layer forward
	cublasLtMatmul(handle, mmDesc, &alpha, dB, lB, dA, lA, &beta, dC, lC, dC, lC, NULL, NULL, 0, 0);

	const half alphah = __float2half(1.0f);
	const half betah = __float2half(0.0f);
	cublasLtMatmul(handle, mmDesch, &alphah, dBh, lBh, dAh, lAh, &betah, dCh, lCh, dCh, lCh, NULL, NULL, 0, 0);

	
	cudaMemcpy(C_fp32, dC, sizeC, cudaMemcpyDeviceToHost);

	half Ch[N * K];
	cudaMemcpy(Ch, dCh, sizeCh, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N * K; i++) C_fp16[i] = __half2float(Ch[i]);

	bool cublasLtFp32Correct = true, cublasLtFp16Correct = true;
	for (int i = 0; i < N * K; i++) {
		if (fabs(CCpu[i] - C_fp32[i]) > 1e-4) {
			cublasLtFp32Correct = false;
			break;
		}
	}
	printf("cuBLAS-Lt FP32 results are %s\n", cublasLtFp32Correct ? "correct" : "incorrect");

	for (int i = 0; i < N * K; i++) {
		if (fabs(CCpu[i] - C_fp16[i]) > 1e-4) {
			cublasLtFp16Correct = false;
			break;
		}
	}
	printf("cuBLAS-Lt FP16 results are %s\n", cublasLtFp16Correct ? "correct" : "incorrect");

	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	cudaFree(dAh); cudaFree(dBh); cudaFree(dCh);
	cublasLtMatrixLayoutDestroy(lA);
	cublasLtMatrixLayoutDestroy(lAh);
	cublasLtMatrixLayoutDestroy(lB);
	cublasLtMatrixLayoutDestroy(lBh);
	cublasLtMatrixLayoutDestroy(lC);
	cublasLtMatrixLayoutDestroy(lCh);
	cublasLtMatmulDescDestroy(mmDesc);
	cublasLtMatmulDescDestroy(mmDesch);
	cublasLtDestroy(handle);

	return 0;

}	
