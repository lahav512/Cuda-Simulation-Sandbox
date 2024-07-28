#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

using namespace nvcuda;

const int TILE_DIM = 16;

__global__ void matrixMulTensorCore(half *A, half *B, float *C, int M, int N, int K) {
    // Define the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> c_frag;

    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    nvcuda::wmma::load_matrix_sync(a_frag, A, TILE_DIM);
    nvcuda::wmma::load_matrix_sync(b_frag, B, TILE_DIM);

    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result
    nvcuda::wmma::store_matrix_sync(C, c_frag, TILE_DIM, nvcuda::wmma::mem_row_major);
}

void initializeMatrices(half *A, half *B, int M, int K, int N) {
    for (int i = 0; i < M * K; ++i) {
        A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
}

int tensor_cores_example_main() {
    // Matrix dimensions
    int M = 16;
    int N = 16;
    int K = 16;

    // Allocate and initialize matrices on the host
    half *h_A = (half *)malloc(M * K * sizeof(half));
    half *h_B = (half *)malloc(K * N * sizeof(half));
    float *h_C = (float *)malloc(M * N * sizeof(float));

    initializeMatrices(h_A, h_B, M, K, N);

    // Allocate and copy matrices to the device
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(half));
    cudaMalloc((void **)&d_B, K * N * sizeof(half));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // // Warm up tensor cores
    // matrixMulTensorCore<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the kernel
    dim3 gridDim(1);
    dim3 blockDim(32, 32);
    matrixMulTensorCore<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate and print GFLOPS
    float numOps = 2.0f * M * N * K; // 2 * M * N * K floating-point operations for matrix multiplication
    float gflops = numOps / (elapsedTime / 1000.0f) / 1e9f;

    std::cout << "Time elapsed: " << elapsedTime << " ms" << std::endl;
    std::cout << "Achieved GFLOPS: " << gflops << " GFLOPS" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
