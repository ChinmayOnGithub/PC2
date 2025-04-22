/*
Summary Points:
-------------
1. Program Purpose:
   - Performs matrix multiplication on GPU
   - Demonstrates parallel computation
   - Shows basic memory management

2. Program Flow:
   a) Get matrix size from user
   b) Allocate and initialize matrices
   c) Copy data to GPU
   d) Perform parallel multiplication
   e) Retrieve and display results
   f) Show execution timing

3. Key Components:
   - Matrix Operations: Row-column multiplication
   - Memory Management: Host and device allocation
   - Thread Organization: 2D block and grid
   - Performance: CUDA event timing

4. Technical Details:
   - Block Size: 16x16 threads (256 total)
   - Grid Size: Calculated based on matrix size
   - Memory Access: Global memory
   - Data Type: Single precision float

5. Memory Usage:
   - Three NxN matrices (A, B, C)
   - Both CPU and GPU memory allocation
   - Size per matrix: N*N*4 bytes
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/**
 * CUDA Kernel for matrix multiplication
 * Each thread computes one element of the output matrix
 */
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N;
    printf("Enter matrix size (N x N): ");
    scanf("%d", &N);

    // Allocate host memory
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 100);
        B[i] = (float)(rand() % 100);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input matrices to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (N + blockSize.y - 1) / blockSize.y);

    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate execution time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Matrix Multiplication Time: %.2f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(A);
    free(B);
    free(C);

    return 0;
}

/*
Compilation and Execution:
------------------------
Compilation:
nvcc matrix_multiplication_gpu.cu -o matrix_multiplication_gpu

Execution:
./matrix_multiplication_gpu
Then enter matrix size when prompted
*/
