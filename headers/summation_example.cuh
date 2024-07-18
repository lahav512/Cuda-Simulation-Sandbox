#include <cuda_runtime.h>
#include <iostream>

#define NUM_NODES 10000000
#define DIM 21
#define THREADS_PER_BLOCK 256

struct Node {
    float data[DIM];
    float sum;
};

void initialize_nodes(Node* nodes) {
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < DIM; ++j) {
            nodes[i].data[j] = static_cast<float>(j + 1);
        }
        nodes[i].sum = 0.0f;
    }
}

__global__ void calculate_sum_1(Node* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        Node* node = &nodes[idx];
        float sum = 0.0f;
        for (int i = 0; i < DIM; ++i) {
            sum += node->data[i];
        }
        node->sum = sum;
    }
}

__global__ void calculate_sum_2(Node* nodes) {
    int item_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (item_idx < NUM_NODES) {
        Node* node = &nodes[item_idx];

        // printf("Before[%d]: data: ", thread_idx);
        // for (int i = 0; i < DIM; i++) {
        //     printf("%d, ", (int) node->data[i]);
        // }
        // printf("\n");

        int start_index;

        // Perform reduction in shared memory
        for (int stride = 1; stride < DIM; stride <<= 1) {
            start_index = thread_idx * 2*stride;
            if (start_index + stride < DIM) {
                node->data[start_index] += node->data[start_index + stride];
            }
            __syncthreads();
        }

        // Thread 0 writes the final result
        if (thread_idx == 0) {
            node->sum = node->data[0];
        }
    }
}


template <typename KernelFunc>
void runKernelAndMeasureTime(KernelFunc kernel, Node* d_nodes, Node* h_nodes, dim3 blocks, int threadsPerBlock, const char* kernelName) {
    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run the kernel
    cudaEventRecord(start, 0);
    kernel<<<blocks, threadsPerBlock>>>(d_nodes);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for " << kernelName << ": " << elapsedTime << " ms" << std::endl;

    // Copy the results back to the host and check the results
    cudaMemcpy(h_nodes, d_nodes, NUM_NODES * sizeof(Node), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i) {
        std::cout << "Vector " << i << " sum (" << kernelName << "): " << h_nodes[i].sum << std::endl;
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int summation_example_main() {
    Node* h_nodes = new Node[NUM_NODES];
    Node* d_nodes;

    // Calculate and print memory used by Node list
    size_t memoryUsed = NUM_NODES * sizeof(Node);
    std::cout << "Memory used by Node list: " << static_cast<float>(memoryUsed) / (1024 * 1024 * 1024) << " GB" << std::endl;

    initialize_nodes(h_nodes);

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(Node));
    cudaMemcpy(d_nodes, h_nodes, NUM_NODES * sizeof(Node), cudaMemcpyHostToDevice);

    // Calculate functions for all nodes
    int threadsPerBlock = 256;
    int blocksPerNode = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(DIM, blocksPerNode); // 21 blocks for each element of node

    // // Run the kernels and measure time
    // for (int i = 0; i < 5; i++) {
    //     runKernelAndMeasureTime(calculate_sum_1, d_nodes, h_nodes, blocks, threadsPerBlock, "calculate_sum_1");
    //     cudaDeviceSynchronize();
    // }

    for (int i = 0; i < 1; i++) {
        runKernelAndMeasureTime(calculate_sum_2, d_nodes, h_nodes, (int) (NUM_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, "calculate_sum_2");
        cudaDeviceSynchronize();
    }

    cudaFree(d_nodes);
    delete[] h_nodes;

    return 0;
}
