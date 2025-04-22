// Simplified CUDA Vector Addition using <ctime>

#include <iostream>
#include <cuda_runtime.h>
#include <ctime>

constexpr int VECTOR_SIZE = 1'000'000;
constexpr int THREADS_PER_BLOCK = 256;

// Kernel function for vector addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    size_t bytes = VECTOR_SIZE * sizeof(int);
    int *h_A, *h_B, *h_C; // Host memory pointers
    int *d_A, *d_B, *d_C; // Device memory pointers

    // Allocate pinned host memory
    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C, bytes);

    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Initialize host arrays
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Calculate grid size
    int blocks = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel and measure execution time
    clock_t start = clock();
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, VECTOR_SIZE);
    cudaDeviceSynchronize();
    clock_t end = clock();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Validate results
    bool valid = true;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            valid = false;
            break;
        }
    }

    // Print execution time and validation result
    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    std::cout << "CUDA execution time: " << ms << " ms\n";
    std::cout << (valid ? "Result is correct!" : "Result is incorrect.") << std::endl;

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
