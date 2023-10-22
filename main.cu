#include <iostream>
#include <math.h>
#include <cuda.h>
 
// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y) {
    for (int i = index; i < n; i += 1) {
        y[i] += x[i];
    }
}
 
int main(void)
{
    int N = 1;
    float *x, *y;

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Print result
    for (int i = 0; i < N; i++) {
        std::cout << "x + y = " << y[i] << std::endl;
    }

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}