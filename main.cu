#include <iostream>
#include <math.h>
#include <cuda.h>
 
// CUDA kernel to add elements of two arrays
__global__ void AplusB(int *res, int a, int b) {
    res[threadIdx.x] = a + b + threadIdx.x;
    printf("GPU: %d: A+B = %d\n", threadIdx.x, res[threadIdx.x]);
}
 
int main(void)
{
    int N = 1000;
    int *res;
    int *x, *y;

    // Allocate memory
    cudaMallocManaged(&res, 1000 * sizeof(int));

    // Execute kernel function
    AplusB<<< 1, N >>>(res, 10, 100);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    // Wait until GPU completes calculations
    cudaDeviceSynchronize();

    // // Print result
    // for(int i = 0; i < N; i++) {
    //     printf("CPU: %d: A+B = %d\n", i, res[i]);
    // }

    // Clear memory
    cudaFree(res);

    return 0;
}