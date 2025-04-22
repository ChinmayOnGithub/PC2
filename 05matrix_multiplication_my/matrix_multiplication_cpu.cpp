#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int N;
    printf("Enter matrix size (N x N): ");
    scanf("%d", &N);

    // Allocate memory
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)(rand() % 100);
        B[i] = (float)(rand() % 100);
    }

    // Start timing
    clock_t start = clock();

    // Perform matrix multiplication
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < N; i++)
            {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    // Calculate execution time
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Matrix Multiplication Time: %.2f ms\n", cpu_time);

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}

/*
Compilation Command:
g++ matrix_multiplication_cpu.cpp -o matrix_multiplication_cpu
*/