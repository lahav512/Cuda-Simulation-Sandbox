#ifndef MODEL_CUH
#define MODEL_CUH

#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "utils.cuh"
#include <mma.h>
using namespace nvcuda;

// Define the dimensions
#define N_X 10
#define N_Y 10
#define N_XY 20
#define N_Z 1
#define N_IN 21
#define N_OUT 21
#define NUM_NODES 1000000
#define average_operations_per_node 966.0

// Define the struct for Node
struct Node {
    __half W[N_OUT][N_IN];
    __half b[N_OUT];
    __half alpha[N_OUT];
    __half beta[N_OUT];
    __half r_in[N_IN];
    __half r_out[N_OUT];
    __half g[N_OUT];
};

void printNodeROut(const Node& node) {
    std::cout << "Node r_out vector: ";
    for (int i = 0; i < N_OUT; ++i) {
        std::cout << std::fixed << std::setprecision(2) << (float) node.r_out[i] << " ";
    }
    std::cout << std::endl;
}

// Kernel to initialize random inputs for nodes
__global__ void initializeRandomInputs(Node* nodes, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_NODES) {
        curandState state;
        curand_init(seed, 0, 0, &state);

        for (int j = 0; j < N_OUT; ++j) {
            for (int k = 0; k < N_IN; ++k) {
                nodes[i].W[j][k] = __float2half(curand_uniform(&state));
            }
            nodes[i].r_in[j] = __float2half(curand_uniform(&state));
            nodes[i].b[j] = __float2half(curand_uniform(&state));
            nodes[i].alpha[j] = __float2half(curand_uniform(&state));
            nodes[i].beta[j] = __float2half(curand_uniform(&state));
            nodes[i].g[j] = 0;
        }
    }
}

// Kernel to calculate the stairs function
__device__ __half stairs(__half t) {
    return t >= (__half) 1.0f ? 1.0f : 0.0f;
}

// Kernel to calculate the stairs function
__device__ __half phi(__half t, int i) {
    if (i < N_X) {
        return t * t;
    } else if (i < N_XY) {
        return stairs(t);
    } else {
        return t * t;
    }
}

// Kernel to calculate g and f functions for all nodes
__global__ void calculate_model_2(Node* nodes, unsigned int* activeThreadsCount) {
    int i = blockIdx.x; // node index
    int j = threadIdx.x; // r_out index

    if (i < NUM_NODES && j < N_OUT) {
        // atomicAdd(activeThreadsCount, 1); // Increment the active thread counter

        Node* node = &nodes[i];
        __half g;
        // k: r_in index
        // j: r_out index

        // Calculate g
        g = 0;
        #pragma unroll
        for (int k = 0; k < N_IN; k++) {
            g = __hadd(g, __hmul(node->W[j][k], node->r_in[k]));
        }
        g = __hadd(g, node->b[j]);

        // Calculate f
        float phi_value = phi(g, j);
        float f_value = __hadd(__hmul(node->alpha[j], g), __hmul(node->beta[j], phi_value));
        node->r_out[j] = f_value;
    }
}

// Kernel to calculate g and f functions for all nodes
__global__ void calculate_model_3(Node* nodes, unsigned int* activeThreadsCount) {
    int i = blockIdx.x; // node index
    int j = threadIdx.x; // r_out index

    if (i < NUM_NODES && j < N_OUT) {
        // atomicAdd(activeThreadsCount, 1); // Increment the active thread counter

        Node* node = &nodes[i];
        __half g;
        // k: r_in index
        // j: r_out index

        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;

        // Initialize the output to zero
        wmma::fill_fragment(c_frag, __float2half(0.0f));

        // Ensure the dimensions match for wmma load functions
        int numTiles = (N_IN + 15) / 16; // Calculate the number of 16x16 tiles

        #pragma unroll
        for (int tile = 0; tile < numTiles; ++tile) {
            int k = tile * 16;

            // Use shared memory to store tiles
            __shared__ __half shared_A[16][16];
            __shared__ __half shared_B[16][16];

            if (k + 16 <= N_IN) {
                for (int row = 0; row < 16; ++row) {
                    for (int col = 0; col < 16; ++col) {
                        shared_A[row][col] = node->W[j][k + col];
                        shared_B[col][row] = node->r_in[k + col];
                    }
                }
            } else {
                // Handle the case where dimensions are not multiples of 16
                for (int row = 0; row < 16; ++row) {
                    for (int col = 0; col < 16; ++col) {
                        if (k + col < N_IN) {
                            shared_A[row][col] = node->W[j][k + col];
                            shared_B[col][row] = node->r_in[k + col];
                        } else {
                            shared_A[row][col] = __float2half(0.0f);
                            shared_B[col][row] = __float2half(0.0f);
                        }
                    }
                }
            }

            // Synchronize to make sure the shared memory is ready
            __syncthreads();

            // Load the shared memory data into the fragments
            wmma::load_matrix_sync(a_frag, &shared_A[0][0], 16);
            wmma::load_matrix_sync(b_frag, &shared_B[0][0], 16);

            // Perform the matrix-vector multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            // Synchronize again before next iteration
            __syncthreads();
        }

        // Store the result
        g = __hadd(c_frag.x[j], node->b[j]);

        // Calculate f
        float phi_value = phi(g, j);
        float f_value = __hadd(__hmul(node->alpha[j], g), __hmul(node->beta[j], phi_value));
        node->r_out[j] = f_value;
    }
}



template <typename KernelFunc>
void runKernelAndMeasureTime(KernelFunc kernel, Node* d_nodes, Node* h_nodes, unsigned int* d_activeThreadsCount, dim3 blocks, dim3 threadsPerBlock, const char* kernelName) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime;

    // Start the timer
    cudaEventRecord(start);

    // Calculate functions for all nodes
    kernel<<<blocks, threadsPerBlock>>>(d_nodes, d_activeThreadsCount);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    cudaEventElapsedTime(&totalTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_nodes, d_nodes, NUM_NODES * sizeof(Node), cudaMemcpyDeviceToHost);

    // Copy active threads count back to host
    unsigned int h_activeThreadsCount;
    cudaMemcpy(&h_activeThreadsCount, d_activeThreadsCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // // Print the active threads count
    // std::cout << "Active threads count: " << h_activeThreadsCount << std::endl;

    // Print the average time taken
    std::cout << "Time for " << kernelName << ": " << totalTime << " ms" << std::endl;

    float numOPS = (float) NUM_NODES * average_operations_per_node;
    float FLOPS = numOPS / (totalTime / 1000);
    float GFLOPS = FLOPS / 1e9;

    std::cout << "FLOPS:" << FLOPS << std::endl;
    std::cout << "GFLOPS:" << GFLOPS << std::endl;

    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int model_main() {
    std::cout << "Load variables" << std::endl;

    // Host memory allocation
    Node* h_nodes = new Node[NUM_NODES];

    // Calculate and print memory used by Node list
    size_t memoryUsed = NUM_NODES * sizeof(Node);
    std::cout << "Memory used by Node list: " << static_cast<float>(memoryUsed) / (1024 * 1024 * 1024) << " GB" << std::endl;

    // Device memory allocation
    Node* d_nodes;
    cudaMalloc(&d_nodes, NUM_NODES * sizeof(Node));
    cudaMemcpy(d_nodes, h_nodes, NUM_NODES * sizeof(Node), cudaMemcpyHostToDevice);

    // Allocate memory for counting active threads
    unsigned int* d_activeThreadsCount;
    cudaMalloc(&d_activeThreadsCount, sizeof(unsigned int));
    cudaMemset(d_activeThreadsCount, 0, sizeof(unsigned int));

    // Calculate functions for all nodes
    int threadsPerBlock = 256;
    int blocksPerNode = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blocks(CEIL_DIV(N_OUT, 32), CEIL_DIV(N_IN, 32));
    dim3 threads(N_OUT, N_IN);

    // Initialize random inputs
    initializeRandomInputs<<<blocksPerNode, threadsPerBlock>>>(d_nodes, 0);
    cudaDeviceSynchronize(); // Ensure initialization is complete

    // // Copy result back to host
    // cudaMemcpy(h_nodes, d_nodes, NUM_NODES * sizeof(Node), cudaMemcpyDeviceToHost);
    // printNode(h_nodes[0]);
    // printNode(h_nodes[7]);

    // // calculate_model_1
    // for (int i = 0; i < 1; i++) {
    //     runKernelAndMeasureTime(calculate_model_1, d_nodes, h_nodes, d_activeThreadsCount, NUM_NODES, 1, "calculate_model_1");
    //     cudaDeviceSynchronize();
    // }
    
    // for (int i = 0; i < 5; i++) {
    //     printNodeROut(h_nodes[i]);
    // }
    // std::cout << std::endl;

    // calculate_model_2
    for (int i = 0; i < 5; i++) {
        runKernelAndMeasureTime(calculate_model_3, d_nodes, h_nodes, d_activeThreadsCount, NUM_NODES, N_OUT, "calculate_model_3");
        cudaDeviceSynchronize();
    }
    
    for (int i = 0; i < 5; i++) {
        printNodeROut(h_nodes[i]);
    }
    std::cout << std::endl;

    // // calculate_model_3
    // for (int i = 0; i < 1; i++) {
    //     runKernelAndMeasureTime(calculate_model_3, d_nodes, h_nodes, d_activeThreadsCount, NUM_NODES, 2 * N_OUT, "calculate_model_3");
    //     cudaDeviceSynchronize();
    // }
    
    // for (int i = 0; i < 1; i++) {
    //     printNodeROut(h_nodes[i]);
    // }
    // std::cout << std::endl;

    // // calculate_model_4
    // for (int i = 0; i < 1; i++) {
    //     runKernelAndMeasureTime(calculate_model_4, d_nodes, h_nodes, d_activeThreadsCount, NUM_NODES, N_OUT, "calculate_model_4");
    //     cudaDeviceSynchronize();
    // }
    
    // for (int i = 0; i < 1; i++) {
    //     printNodeROut(h_nodes[i]);
    // }
    // std::cout << std::endl;

    // Cleanup
    delete[] h_nodes;
    cudaFree(d_nodes);
    cudaFree(d_activeThreadsCount);

    return 0;
}

#endif // MODEL_CUH
