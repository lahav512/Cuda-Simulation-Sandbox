#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int cublas_sgemm_main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int N = 1000000; // Number of matrices/vectors
    const int D = 21; // Dimension of each vector/matrix

    size_t size_A = D * D * N * sizeof(float);
    size_t size_X = D * N * sizeof(float);
    size_t size_B = D * N * sizeof(float);

    float *A, *X, *B;

    // Allocate memory on device
    cudaMalloc(&A, size_A);
    cudaMalloc(&X, size_X);
    cudaMalloc(&B, size_B);

    // Setup cuRAND
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123ULL); // Same seed for consistent results

    // Generate random numbers for A and X
    curandGenerateUniform(gen, A, D * D * N);
    curandGenerateUniform(gen, X, D * N);

    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Start timing
    cudaEventRecord(start);

    // Perform the matrix-matrix multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, N, D, &alpha, A, D, X, D, &beta, B, D);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time for matrix multiplication: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(A);
    cudaFree(X);
    cudaFree(B);
    curandDestroyGenerator(gen);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return 0;
}
