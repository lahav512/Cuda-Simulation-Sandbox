#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void addIntsCUDA(int *a, int *b) {
    a[0] += b[0];
}

int main() {
    int a = 5;
    int b = 9;
    // int res;
    int *d_a, *d_b;

    d_a[0] = a;
    d_b[0] = b;

    // Allocate memory for the kernel function in the GPU
    cudaMallocManaged(&d_a, sizeof(int));
    cudaMallocManaged(&d_b, sizeof(int));

    // // Allocate memory for the kernel function in the GPU
    // cudaMalloc(&d_a, sizeof(int));
    // cudaMalloc(&d_b, sizeof(int));

    // // Copy the memory from the CPU to the GPU
    // cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // Execute the kernel function
    addIntsCUDA<<<1, 1>>>(d_a, d_b);

    // // Copy the result from the GPU to the CPU
    // cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Print the result
    cout<<"The result is "<< a <<endl;

    // Clear the GPU memory
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
