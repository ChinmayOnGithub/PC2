#include <iostream>
#include <ctime>

constexpr int VECTOR_SIZE = 1'000'000;

int main()
{
    size_t bytes = VECTOR_SIZE * sizeof(int);
    int *h_A = new int[VECTOR_SIZE];
    int *h_B = new int[VECTOR_SIZE];
    int *h_C = new int[VECTOR_SIZE];

// Initialize host arrays
#pragma acc parallel loop copyout(h_A[0 : VECTOR_SIZE], h_B[0 : VECTOR_SIZE])
    for (int i = 0; i < VECTOR_SIZE; ++i)
    {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // Vector addition on device
    clock_t start = clock();
#pragma acc parallel loop copyin(h_A[0 : VECTOR_SIZE], h_B[0 : VECTOR_SIZE]) copyout(h_C[0 : VECTOR_SIZE])
    for (int i = 0; i < VECTOR_SIZE; ++i)
    {
        h_C[i] = h_A[i] + h_B[i];
    }
    clock_t end = clock();

    // Validate results
    bool valid = true;
    for (int i = 0; i < VECTOR_SIZE; ++i)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            valid = false;
            break;
        }
    }

    // Print execution time and validation result
    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    std::cout << "OpenACC execution time: " << ms << " ms\n";
    std::cout << (valid ? "Result is correct!" : "Result is incorrect.") << std::endl;

    /*
    // Uncomment to print the first few elements of the result vector
    std::cout << "\nSample results (first 10 elements):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << " (A[" << i << "] + B[" << i << "] = "
                  << h_A[i] << " + " << h_B[i] << " = " << h_A[i] + h_B[i] << ")\n";
    }
    */

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

// Compile : pgc++ -acc -ta=tesla:managed -fast vectorAdditionGPU_acc.cpp -o vectorAdditionGPU_acc
// Run : ./vectorAdditionGPU_acc

// alternative command
// g++ -O3 -fopenacc -o vectorAdditionGPU_acc vectorAdditionGPU_acc.cpp

/*
Theory :
OpenACC Optimizations:
#pragma acc parallel loop for both initialization and addition

copyin/copyout clauses to manage data movement

-ta=tesla:managed enables NVIDIA GPU acceleration with unified memory

-fast flag enables general optimizations
*/