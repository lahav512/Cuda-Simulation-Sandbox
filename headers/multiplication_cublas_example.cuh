#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

const int MATRIX_SIZE = 21;  // Example size
const int VECTOR_SIZE = 21;  // Since it's a 21x21 matrix
const int BATCH_COUNT = 1000000;

int multiplication_cublas_example_main() {
    // Host and device pointers for matrices and vectors
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate pinned memory for host matrices and vectors
    cudaMallocHost((void**)&h_A, MATRIX_SIZE * MATRIX_SIZE * BATCH_COUNT * sizeof(float));
    cudaMallocHost((void**)&h_B, VECTOR_SIZE * BATCH_COUNT * sizeof(float));
    cudaMallocHost((void**)&h_C, VECTOR_SIZE * BATCH_COUNT * sizeof(float));

    // Initialize host matrices and vectors with some values
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE * BATCH_COUNT; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < VECTOR_SIZE * BATCH_COUNT; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory for device matrices and vectors
    cudaMalloc((void**)&d_A, MATRIX_SIZE * MATRIX_SIZE * BATCH_COUNT * sizeof(float));
    cudaMalloc((void**)&d_B, VECTOR_SIZE * BATCH_COUNT * sizeof(float));
    cudaMalloc((void**)&d_C, VECTOR_SIZE * BATCH_COUNT * sizeof(float));

    // Copy host matrices and vectors to device
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * BATCH_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, VECTOR_SIZE * BATCH_COUNT * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Device pointers for batched operation
    float **d_A_array, **d_B_array, **d_C_array;
    cudaMalloc((void ***)&d_A_array, BATCH_COUNT * sizeof(float *));
    cudaMalloc((void ***)&d_B_array, BATCH_COUNT * sizeof(float *));
    cudaMalloc((void ***)&d_C_array, BATCH_COUNT * sizeof(float *));

    // Array of pointers for each matrix and vector
    float **h_A_array = (float**)malloc(BATCH_COUNT * sizeof(float *));
    float **h_B_array = (float**)malloc(BATCH_COUNT * sizeof(float *));
    float **h_C_array = (float**)malloc(BATCH_COUNT * sizeof(float *));

    for (int i = 0; i < BATCH_COUNT; ++i) {
        h_A_array[i] = d_A + i * MATRIX_SIZE * MATRIX_SIZE;
        h_B_array[i] = d_B + i * VECTOR_SIZE;
        h_C_array[i] = d_C + i * VECTOR_SIZE;
    }

    // Copy arrays of pointers to device
    cudaMemcpy(d_A_array, h_A_array, BATCH_COUNT * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, h_B_array, BATCH_COUNT * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array, h_C_array, BATCH_COUNT * sizeof(float *), cudaMemcpyHostToDevice);

    // Warm up the GPU (optional, to avoid first call overhead)
    cublasSgemvBatched(handle,
                       CUBLAS_OP_N,
                       MATRIX_SIZE, MATRIX_SIZE,
                       &alpha,
                       (const float **)d_A_array, MATRIX_SIZE,
                       (const float **)d_B_array, 1,
                       &beta,
                       d_C_array, 1,
                       1);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);

    // Perform batched matrix-vector multiplication
    cublasSgemvBatched(handle,
                       CUBLAS_OP_N,
                       MATRIX_SIZE, MATRIX_SIZE,
                       &alpha,
                       (const float **)d_A_array, MATRIX_SIZE,
                       (const float **)d_B_array, 1,
                       &beta,
                       d_C_array, 1,
                       BATCH_COUNT);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalTime;
    // Calculate the elapsed time in milliseconds
    cudaEventElapsedTime(&totalTime, start, stop);

    // Print the time taken
    std::cout << "Time for batched matrix-vector multiplication: " << totalTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, VECTOR_SIZE * BATCH_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first resulting vector
    std::cout << "Result vector C (first vector):" << std::endl;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    free(h_A_array);
    free(h_B_array);
    free(h_C_array);

    return 0;
}
