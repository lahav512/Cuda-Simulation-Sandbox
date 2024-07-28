#ifndef MODEL_2_CUH
#define MODEL_2_CUH

#include <cassert>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "utils.cuh"
using namespace std; 

// Define the dimensions
#define N_X 10
#define N_Y 10
#define N_XY 20
#define N_Z 1
#define N_IN 21
#define N_OUT 21
#define DIM 21
#define NUM_NODES 10

// Define the struct for Node
struct Node {
    float W[N_OUT * N_IN];       // [N_OUT][N_IN];
    float b[N_OUT];       // [N_OUT];
    float alpha[N_OUT];   // [N_OUT];
    float beta[N_OUT];    // [N_OUT];
    float r_in[N_IN];    // [N_IN];
    float r_out[N_OUT];   // [N_OUT];
};

void printNodeROut(const Node& node) {
    std::cout << "Node r_out vector: ";
    for (int i = 0; i < N_OUT; ++i) {
        std::cout << std::fixed << std::setprecision(2) << node.r_out[i] << " ";
    }
    std::cout << std::endl;
}

// Function to print the contents of a Node
void printNode(const Node& node) {
    std::cout << "Node W matrix:" << std::endl;
    for (int i = 0; i < N_OUT; ++i) {
        for (int j = 0; j < N_IN; ++j) {
            std::cout << std::fixed << std::setprecision(2) << node.W[i + j * N_OUT] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Node b vector: ";
    for (int i = 0; i < N_OUT; ++i) {
        std::cout << std::fixed << std::setprecision(2) << node.b[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Node alpha vector: ";
    for (int i = 0; i < N_OUT; ++i) {
        std::cout << std::fixed << std::setprecision(2) << node.alpha[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Node beta vector: ";
    for (int i = 0; i < N_OUT; ++i) {
        std::cout << std::fixed << std::setprecision(2) << node.beta[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Node r_in vector: ";
    for (int i = 0; i < N_IN; ++i) {
        std::cout << std::fixed << std::setprecision(2) << node.r_in[i] << " ";
    }
    std::cout << std::endl;

    printNodeROut(node);
}

// Kernel to initialize random inputs for nodes
__global__ void initializeRandomInputs(Node* nodes, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_NODES) {
        curandState state;
        curand_init(seed, 0, 0, &state);

        for (int j = 0; j < N_OUT; ++j) {
            for (int k = 0; k < N_IN; ++k) {
                nodes[i].W[j + k * N_OUT] = curand_uniform(&state);
            }
            nodes[i].r_in[j] = curand_uniform(&state);
            nodes[i].b[j] = curand_uniform(&state);
            nodes[i].alpha[j] = curand_uniform(&state);
            nodes[i].beta[j] = curand_uniform(&state);
        }
    }
}

// Kernel to calculate the stairs function
__device__ float stairs(float t) {
    return t >= 1.0f ? 1.0f : 0.0f;
}

// Kernel to calculate the stairs function
__device__ float phi(float t, int i) {
    if (i < N_X) {
        return t * t;
    } else if (i < N_XY) {
        return stairs(t);
    } else {
        return t * t;
    }
}

__global__ void copyNodeData(Node* d_nodes, float** d_A, float** d_B, float** d_C, int batchCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchCount) {
        memcpy(d_A[idx], d_nodes[idx].W, N_OUT * N_IN * sizeof(float));
        memcpy(d_B[idx], d_nodes[idx].r_in, N_IN * sizeof(float));
        memcpy(d_C[idx], d_nodes[idx].b, N_OUT * sizeof(float));

        // if (idx == 0) {
        //     printf("d_B matrix: ");
        //     for (int i = 0; i < N_IN; ++i) {
        //         printf("%.2f ", d_B[idx][i]);
        //     }
        //     printf("\n");
        // }
    }
}

__global__ void printDeviceArray(float* a, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%2.f ", a[i]);
    }
    printf("\n");
}

int model_2_main() {
    std::cout << "Load variables" << std::endl;

    // Host memory allocation
    Node* h_nodes = new Node[NUM_NODES];

    // Calculate and print memory used by Node list
    size_t memoryUsed = NUM_NODES * sizeof(Node);
    std::cout << "Memory used by Node list: " << static_cast<float>(memoryUsed) / (1024 * 1024 * 1024) << " GB" << std::endl;

    bool *h_is_completed = new bool[NUM_NODES];

    for (int i = 0; i < NUM_NODES; ++i) {
        h_is_completed[i] = false;
    }

    // Device memory allocation
    Node* d_nodes;
    cudaMalloc(&d_nodes, NUM_NODES * sizeof(Node));
    cudaMemcpy(d_nodes, h_nodes, NUM_NODES * sizeof(Node), cudaMemcpyHostToDevice);

    // Allocate memory for counting active threads
    unsigned int* d_activeThreadsCount;
    cudaMalloc(&d_activeThreadsCount, sizeof(unsigned int));
    cudaMemset(d_activeThreadsCount, 0, sizeof(unsigned int));

    // Allocate memory to is_copmleted
    bool *d_is_completed;
    cudaMalloc(&d_is_completed, NUM_NODES * sizeof(Node));
    cudaMemcpy(d_is_completed, h_is_completed, NUM_NODES * sizeof(Node), cudaMemcpyHostToDevice);

    // Calculate functions for all nodes
    int threadsPerBlock = 256;
    int blocksPerNode = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;

    // dim3 blocks(CEIL_DIV(N_OUT, 32), CEIL_DIV(N_IN, 32));
    // dim3 threads(N_OUT, N_IN);

    // Initialize random inputs
    initializeRandomInputs<<<blocksPerNode, threadsPerBlock>>>(d_nodes, 0);
    cudaDeviceSynchronize(); // Ensure initialization is complete

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime;

    // Start the timer
    cudaEventRecord(start);

    // Calculate g using cublas (W*r_in + b)
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const int m = N_OUT, n = N_IN, k = 1; // Dimensions for each matrix
    const int batchCount = NUM_NODES;
    const float alpha = 1.0f;
    const float beta = 1.0f;

    float** d_A = (float**)malloc(batchCount * sizeof(float*));
    float** d_B = (float**)malloc(batchCount * sizeof(float*));
    float** d_C = (float**)malloc(batchCount * sizeof(float*));

    float** h_C = (float**)malloc(batchCount * sizeof(float*));

    for (int i = 0; i < batchCount; i++) {
        cudaMalloc(&d_A[i], N_OUT * N_IN * sizeof(float));
        cudaMalloc(&d_B[i], N_IN * sizeof(float));
        cudaMalloc(&d_C[i], N_OUT * sizeof(float));

        h_C[i] = (float*)malloc(N_OUT * sizeof(float));
    }

    float** d_A_dev;
    float** d_B_dev;
    float** d_C_dev;

    cudaMalloc(&d_A_dev, batchCount * sizeof(float*));
    cudaMalloc(&d_B_dev, batchCount * sizeof(float*));
    cudaMalloc(&d_C_dev, batchCount * sizeof(float*));

    cudaMemcpy(d_A_dev, d_A, batchCount * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_dev, d_B, batchCount * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_dev, d_C, batchCount * sizeof(float*), cudaMemcpyHostToDevice);

    // Launch kernel to copy data from d_nodes to d_A, d_B, and d_C
    int blockSize = 256;
    int numBlocks = (batchCount + blockSize - 1) / blockSize;
    copyNodeData<<<numBlocks, blockSize>>>(d_nodes, d_A_dev, d_B_dev, d_C_dev, batchCount);
    cudaDeviceSynchronize();

    cudaMemcpy(h_nodes, d_nodes, NUM_NODES * sizeof(Node), cudaMemcpyDeviceToHost);
    // printNode(h_nodes[0]);

    // for (int i = 0; i < batchCount; i++) {
    //     cudaMemcpy(h_C[i], d_C[i], m * k * sizeof(float), cudaMemcpyDeviceToHost);
    // }
    // std::cout << "h_C vector: ";
    // for (int i = 0; i < N_OUT; ++i) {
    //     std::cout << std::fixed << std::setprecision(2) << h_C[0][i] << " ";
    // }
    // std::cout << std::endl;

    int stat = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, n, &beta, d_C, m, batchCount);

    if(stat != CUBLAS_STATUS_SUCCESS){
	    cerr << "cublasSgemmBatched failed" << endl;
	    exit(1);
    }
    assert(!cudaGetLastError());

    for (int i = 0; i < batchCount; i++) {
        // Copy the result from the device to the host
        cublasGetVector(m*k, sizeof(float), d_C[i], 1, h_C[i], 1);

        // cudaMemcpy(h_C[i], d_C[i], m * k * sizeof(float), cudaMemcpyDeviceToHost);
    }
    std::cout << "h_C vector: ";
    for (int i = 0; i < N_OUT; ++i) {
        std::cout << std::fixed << std::setprecision(2) << h_C[0][i] << " ";
    }
    std::cout << std::endl;

    // Calculate f using a native kernel




    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    cudaEventElapsedTime(&totalTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_nodes, d_nodes, NUM_NODES * sizeof(Node), cudaMemcpyDeviceToHost);
    // printNodeROut(h_nodes[0]);

    // // Copy active threads count back to host
    // unsigned int h_activeThreadsCount;
    // cudaMemcpy(&h_activeThreadsCount, d_activeThreadsCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // // // Print the active threads count
    // // std::cout << "Active threads count: " << h_activeThreadsCount << std::endl;

    // Print the average time taken
    std::cout << "Time: " << totalTime << " [ms]" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 

    // Cleanup
    delete[] h_nodes;
    cudaFree(d_nodes);
    cudaFree(d_activeThreadsCount);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);

    return 0;
}

#endif // MODEL_CUH
