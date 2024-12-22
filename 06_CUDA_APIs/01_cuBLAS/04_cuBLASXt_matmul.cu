#include <cublasXt.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 512
#define M 512
#define K 512

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
	float CCpu[N * K], CCuBLASXt[N * K];

    mulMatsCpu(A, B, CCpu, N, M, K);

    float alpha = 1.0, beta = 0.0;

    cublasXtHandle_t handleXt;
    cublasXtCreate(&handleXt);

    int devices[1] = {0};
    cublasXtDeviceSelect(handleXt, 1, devices);

    cublasXtSgemm(handleXt, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, B, K, A, N, &beta, CCuBLASXt, K);

    bool cuBLASXtCorrect = true;
	for (int i = 0; i < N * K; i++) {
		if (fabs(CCpu[i] - CCuBLASXt[i]) > 1e-3) {
			cuBLASXtCorrect = false;
			break;
		}
    }
    printf("cuBLASXt results are %s\n", cuBLASXtCorrect ? "correct" : "incorrect");

    cublasXtDestroy(handleXt);

    return 0;
}