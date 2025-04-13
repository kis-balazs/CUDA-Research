#include "main.cuh"

void initMat(float *MAT, int r, int c) {
    for (int i = 0; i < r * c; i++) MAT[i] = (float)rand() / RAND_MAX;
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

int main(int argc, char **argv) {
    if (argc < 2 || argc > 2) {
        std::cerr << "<!> CLI must include one argument!" << std::endl;
        return -1;
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        std::cout << "<?> Help Menu\n  Available parameters:\n" << "\t- printCudaDeviceInfo" << std::endl;
        for (const auto& me : cudaFunctionMap) std::cout << "\t- " << me.first << std::endl;
        return 0;
    }

    // specific entry, to get GPU specs
    if (strcmp(argv[1], "printCudaDeviceInfo") == 0) {
        pcdi();
        return 0;
    }

    float *hA, *hB, *hC, *hC_ref;
	float *dA, *dB, *dC, *dC_ref;

    int sizes[5] = {64, 128, 256, 512, 1024};
    for (auto s: sizes) {
        int N = s, M = s, K = s;

        // ------
        int sizeA = N * M * sizeof(float);
        int sizeB = M * K * sizeof(float);
        int sizeC = N * K * sizeof(float);

        hA = (float*)malloc(sizeA);
        hB = (float*)malloc(sizeB);
        hC = (float*)malloc(sizeC);
        hC_ref = (float*)malloc(sizeC);

        srand(time(NULL));
        initMat(hA, N, M);
        initMat(hB, M, K);

        cudaMalloc(&dA, sizeA);
        cudaMalloc(&dB, sizeB);
        cudaMalloc(&dC, sizeC);
        cudaMalloc(&dC_ref, sizeC);

        cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

        // execute based on user input
        float benchmarkTime;
        float alpha = 1.0f, beta = 0.0f;
        auto it = cudaFunctionMap.find(argv[1]);
        if (it != cudaFunctionMap.end()) {
            
            // TODO replace with cuBLAS implementation -> the benchmark!
            mulMatsCpu(hA, hB, hC_ref, N, M, K);
            //refTime = it->second(N, M, K, alpha, dA, dB, beta, dC_ref, false);

            benchmarkTime = it->second(N, M, K, alpha, dA, dB, beta, dC, true);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

            // cudaMemcpy(hC_ref, dC_ref, sizeC, cudaMemcpyDeviceToHost);
            cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);
            if (!verify_matrix(hC_ref, hC, N * K)) {
                std::cout << "Failed to pass the correctness verification against NVIDIA cuBLAS. *(TBD)*" << std::endl;
            } else {
                long long flops = 2LL * N * M * K;
                std::cout << "SIZE: " << N <<  "; average elapsed time: " << benchmarkTime <<
                "s; performance: " << (flops * 1e-9) / benchmarkTime << " GFLOPS" << std::endl;
            }
        } else {
            std::cerr << "Entry not found!" << std::endl;
        }
        free(hA); free(hB); free(hC); free(hC_ref);
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
    }
    return 0;
}