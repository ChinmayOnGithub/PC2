#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

// Initialize matrix with random 0s and 1s
void initializeMatrix(int* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = rand() % 2;
    }
}

// Print first 15x15 elements of matrix for preview
void printMatrix(int* matrix, int N) {
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 15; j++) {
            cout << matrix[i * 15 + j] << " ";
        }
        cout << endl;
    }
}

// CUDA kernel: Each thread computes one element of C
__global__ void matrixMultiplyKernel(const int* A, const int* B, int* C, int N) {
    // Calculate row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        // Each thread computes dot product for its position
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Wrapper function to handle all GPU operations
void matrixMultiplyGPU(const int* h_A, const int* h_B, int* h_C, int N) {
    int size = N * N * sizeof(int);
    int *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices from CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,  // Ceiling division for grid size
                 (N + blockDim.y - 1) / blockDim.y);

    // Launch matrix multiplication kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy result from GPU back to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1024;  // Matrix dimension

    // Allocate matrices in CPU memory
    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C = new int[N * N];

    // Initialize input matrices with random values
    srand(time(0));
    initializeMatrix(A, N);
    initializeMatrix(B, N);

    // Print input matrices (15x15 preview)
    cout << "Matrix A:" << endl;
    printMatrix(A, N);
    
    cout << "Matrix B:" << endl;
    printMatrix(B, N);

    // Measure GPU execution time
    clock_t start = clock();
    matrixMultiplyGPU(A, B, C, N);
    clock_t end = clock();

    // Print result matrix (15x15 preview)
    cout << "Result Matrix C (A * B):" << endl;
    printMatrix(C, N);

    // Calculate and print execution time
    double elapsed = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "GPU Execution Time: " << elapsed << " ms" << endl;

    /* 
    // Uncomment to verify sample calculations (for a few specific elements)
    cout << "\nVerifying calculations for selected elements:" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int manualSum = 0;
            cout << "C[" << i << "][" << j << "] = ";
            for (int k = 0; k < 5; k++) {
                cout << A[i * N + k] << "*" << B[k * N + j];
                if (k < 4) cout << " + ";
                manualSum += A[i * N + k] * B[k * N + j];
            }
            // Calculate the rest of the sum for verification
            for (int k = 5; k < N; k++) {
                manualSum += A[i * N + k] * B[k * N + j];
            }
            cout << " + ... = " << C[i * N + j] << " (manually verified: " << manualSum << ")" << endl;
        }
    }
    */

    // Free CPU memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}