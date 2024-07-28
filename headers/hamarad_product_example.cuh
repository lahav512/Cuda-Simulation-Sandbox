#include <cuda_runtime.h>
#include <iostream>
#include <vector>

void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = 3.0;
    }
}

__global__ void hadamardProductCUDA(float *A, float *B, int N, int opsPerThread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * opsPerThread;
    int end = start + opsPerThread;

    // if (idx == 0) {
    //     printf("ok\n");
    // }
  
    for (int i = start; i < end && i < N * N; ++i) {
        A[i] *= B[i];
    }
}

__global__ void copyFromShared(float *A, float *B, int N, int opsPerThread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * opsPerThread;
    int end = start + opsPerThread;

    // if (idx == 0) {
    //     printf("ok\n");
    // }

    __shared__ float res[16777216];
    
    for (int i = start; i < end && i < N * N; ++i) {
        res[i] = A[i] * B[i];
    }
}

void runHadamardProduct(int N, int blockSize, int opsPerThread) {
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);

    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int numBlocks = ((N * N) / opsPerThread + blockSize - 1) / blockSize;

    // // Warm up the GPU
    // hadamardProductCUDA<<<numBlocks, blockSize>>>(d_A, d_B, N, opsPerThread);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    hadamardProductCUDA<<<numBlocks, blockSize>>>(d_A, d_B, N, opsPerThread);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy the result back to the host
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    // Calculate TFLOPS
    float numOps = (float) N * N; // Number of operations (N^2 multiplications)
    float gflops = (numOps / (elapsedTime / 1000.0f)) / 1e9; // Convert ms to seconds and ops to GFLOPS

    // Print the results
    std::cout << "blockSize: " << blockSize << ", opsPerThread: " << opsPerThread << ", Time elapsed: " << elapsedTime << " ms, Achieved GFLOPS: " << gflops << " , result: " << h_A[0] << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int hamarad_product_example_main() {
    int N = 1024 * 4;  // Matrix size

    // std::vector<int> blockSizes = {64, 128, 256, 512, 1024};
    // std::vector<int> opsPerThreads = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    // for (int blockSize : blockSizes) {
    //     for (int opsPerThread : opsPerThreads) {
    //         runHadamardProduct(N, blockSize, opsPerThread);
    //     }
    // }

    int blockSize = 512;
    int opsPerThread = 8192;

    runHadamardProduct(N, blockSize, opsPerThread);


    return 0;
}
