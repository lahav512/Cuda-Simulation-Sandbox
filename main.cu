#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Define the dimensions
#define N_X 10
#define N_Y 10
#define N_XY 20
#define N_Z 1
#define N_OUT 21
#define DIM 21
#define NUM_NODES 1000000

// Define the struct for Node
struct Node {
    float W[DIM][DIM];
    float b[DIM];
    float alpha[DIM];
    float beta[DIM];
    float r_in[DIM];
    float r_out[DIM];
};

// Kernel to initialize random inputs for nodes
__global__ void initializeRandomInputs(Node* nodes, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        for (int i = 0; i < DIM; ++i) {
            nodes[idx].r_in[i] = curand_uniform(&state);
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

// Kernel to calculate g and f functions for all nodes
__global__ void calculateFunctions(Node* nodes, bool* is_completed, unsigned int* activeThreadsCount) {
    __shared__ float partialSum[256]; // Assumes a maximum of 256 threads per block

    int i = blockIdx.x; // Use block index as i
    int nodeIdx = blockIdx.y * blockDim.x + threadIdx.x; // Use blockIdx.y for node index
    int j = threadIdx.x; // Use thread index as j

    if (nodeIdx < NUM_NODES) {
        atomicAdd(activeThreadsCount, 1); // Increment the active thread counter

        Node* node = &nodes[nodeIdx];
        float temp = 0.0f;

        // Each thread computes part of the sum
        if (j < DIM) {
            temp = node->W[i][j] * node->r_in[j];
        }

        // Store partial sum in shared memory
        partialSum[j] = temp;
        __syncthreads();

        // Perform reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (j < stride) {
                partialSum[j] += partialSum[j + stride];
            }
            __syncthreads();
        }

        // Thread 0 writes the final result
        if (!is_completed[i]) {
            float g_value = partialSum[0] + node->b[i];
            float phi_value = phi(g_value, i);
            float f_value = node->alpha[i] * g_value + node->beta[i] * phi_value;
            node->r_out[i] = f_value;
            is_completed[i] = true;
        }
    }
}


int main() {
    std::cout << "Load variables" << std::endl;

    // Host memory allocation
    Node* h_nodes = new Node[NUM_NODES];

    // Calculate and print memory used by Node list
    size_t memoryUsed = NUM_NODES * sizeof(Node);
    std::cout << "Memory used by Node list: " << static_cast<float>(memoryUsed) / (1024 * 1024 * 1024) << " GB" << std::endl;

    // Initialize host nodes with sample values
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < DIM; ++j) {
            for (int k = 0; k < DIM; ++k) {
                h_nodes[i].W[j][k] = static_cast<float>(rand()) / RAND_MAX;
            }
            h_nodes[i].b[j] = static_cast<float>(rand()) / RAND_MAX;
            h_nodes[i].alpha[j] = static_cast<float>(rand()) / RAND_MAX;
            h_nodes[i].beta[j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

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
    dim3 blocks(DIM, blocksPerNode); // 21 blocks for each element of node

    // Initialize random inputs
    initializeRandomInputs<<<blocksPerNode, threadsPerBlock>>>(d_nodes, time(NULL));
    cudaDeviceSynchronize(); // Ensure initialization is complete

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime;

    // Start the timer
    cudaEventRecord(start);

    // Calculate functions for all nodes
    calculateFunctions<<<blocks, threadsPerBlock>>>(d_nodes, d_is_completed, d_activeThreadsCount);

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

    // Print the active threads count
    std::cout << "Active threads count: " << h_activeThreadsCount << std::endl;

    // Print the average time taken
    std::cout << "Time taken: " << totalTime << " ms" << std::endl;

    // Cleanup
    delete[] h_nodes;
    cudaFree(d_nodes);
    cudaFree(d_activeThreadsCount);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
