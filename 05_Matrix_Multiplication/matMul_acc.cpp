#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

void initializeMatrix(int *matrix, int N)
{
#pragma acc parallel loop copyout(matrix[0 : N * N])
    for (int i = 0; i < N * N; i++)
    {
        matrix[i] = rand() % 2;
    }
}

void printMatrix(int *matrix, int N)
{
    for (int i = 0; i < 15; i++)
    {
        for (int j = 0; j < 15; j++)
        {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
}

void matrixMultiply(int *A, int *B, int *C, int N)
{
#pragma acc data copyin(A[0 : N * N], B[0 : N * N]) copyout(C[0 : N * N])
    {
#pragma acc parallel loop collapse(2)
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int sum = 0;
#pragma acc loop seq
                for (int k = 0; k < N; k++)
                {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

int main()
{
    int N = 1024;

    int *A = new int[N * N];
    int *B = new int[N * N];
    int *C = new int[N * N];

    srand(time(0));
    initializeMatrix(A, N);
    initializeMatrix(B, N);

    cout << "Matrix A:" << endl;
    printMatrix(A, N);

    cout << "Matrix B:" << endl;
    printMatrix(B, N);

    clock_t start = clock();
    matrixMultiply(A, B, C, N);
    clock_t end = clock();

    cout << "Result Matrix C (A * B):" << endl;
    printMatrix(C, N);

    double elapsed = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "Execution Time: " << elapsed << " ms" << endl;

    /*
    // Uncomment to verify sample calculations (for a few specific elements)
    cout << "\nVerifying calculations for selected elements:" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int expectedSum = 0;
            cout << "C[" << i << "][" << j << "] = ";
            for (int k = 0; k < N; k++) {
                if (k < 5) {
                    cout << A[i * N + k] << "*" << B[k * N + j];
                    if (k < 4) cout << " + ";
                }
                else if (k == 5) cout << "...";
                expectedSum += A[i * N + k] * B[k * N + j];
            }
            cout << " = " << expectedSum << " (calculated: " << C[i * N + j] << ")" << endl;
        }
    }
    */

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

// Compile : pgc++ -acc -ta=tesla:managed -fast matMul_acc.cpp -o matMul_acc
// Run : ./matMul_acc

// Alternative command
// compile : g++ -O3 -fopenacc -o matMul_acc matMul_acc.cpp
// Run : ./matMul_acc

/*
Theory :
Key OpenACC Optimizations:
Added #pragma acc parallel loop for matrix initialization

Used #pragma acc data region to manage data movement for the multiplication

Added #pragma acc parallel loop collapse(2) to parallelize the outer two loops

Used #pragma acc loop seq for the innermost reduction loop
*/