#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

#define N 21  // Size of the matrix and vector
#define BATCH_SIZE 1  // Number of matrix-vector multiplications

typedef struct {
    float *d_A;  // Device pointer to matrix A
    float *d_x;  // Device pointer to vector x
    float *d_y;  // Device pointer to output vector y
} Node;

// Error checking helpers
#define CHECK_CUDA(call) {\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

#define CHECK_CUBLAS(call) {\
    cublasStatus_t status = call;\
    if (status != CUBLAS_STATUS_SUCCESS) {\
        printf("CUBLAS error: %s:%d, ", __FILE__, __LINE__);\
        printf("status: %d\n", status);\
        exit(1);\
    }\
}

#define CHECK_CURAND(call) {\
    curandStatus_t status = call;\
    if (status != CURAND_STATUS_SUCCESS) {\
        printf("CURAND error: %s:%d, ", __FILE__, __LINE__);\
        printf("status: %d\n", status);\
        exit(1);\
    }\
}

int matrix_vector_cublas_example_main() {
    cublasHandle_t handle;
    curandGenerator_t gen;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set a fixed seed for reproducibility
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // Allocate memory for nodes
    Node *nodes = (Node *)malloc(BATCH_SIZE * sizeof(Node));

    // Allocate and initialize data on the device
    for (int i = 0; i < BATCH_SIZE; i++) {
        CHECK_CUDA(cudaMalloc(&(nodes[i].d_A), N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&(nodes[i].d_x), N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&(nodes[i].d_y), N * sizeof(float)));

        // Initialize matrices and vectors with random values
        CHECK_CURAND(curandGenerateUniform(gen, nodes[i].d_A, N * N));
        CHECK_CURAND(curandGenerateUniform(gen, nodes[i].d_x, N));
    }

    // Create arrays of pointers for batch processing
    float **array_A = (float **)malloc(BATCH_SIZE * sizeof(float *));
    float **array_x = (float **)malloc(BATCH_SIZE * sizeof(float *));
    float **array_y = (float **)malloc(BATCH_SIZE * sizeof(float *));

    for (int i = 0; i < BATCH_SIZE; i++) {
        array_A[i] = nodes[i].d_A;
        array_x[i] = nodes[i].d_x;
        array_y[i] = nodes[i].d_y;
    }

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;

    // Create CUDA events for timing
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Start the timer
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Perform batched matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemvBatched(handle, CUBLAS_OP_N, N, N, &alpha, (const float **)array_A, N,
                  (const float **)array_x, 1, &beta, array_y, 1, BATCH_SIZE));

    // Ensure all operations are complete before stopping the timer
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop)); // Ensure the stop event completes successfully

    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time for batched matrix-vector multiplication: %f ms\n", elapsedTime);

    // Cleanup resources
    for (int i = 0; i < BATCH_SIZE; i++) {
        CHECK_CUDA(cudaFree(nodes[i].d_A));
        CHECK_CUDA(cudaFree(nodes[i].d_x));
        CHECK_CUDA(cudaFree(nodes[i].d_y));
    }
    free(nodes);
    free(array_A);
    free(array_x);
    free(array_y);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CURAND(curandDestroyGenerator(gen));
    return 0;
}
