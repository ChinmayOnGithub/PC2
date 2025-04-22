/*
Summary Points:
-------------
1. Program Purpose:
   - Enhances image contrast using histogram equalization
   - Processes images on GPU for faster execution
   - Demonstrates atomic operations in CUDA

2. Program Flow:
   a) Load grayscale input image
   b) Calculate histogram (frequency of each intensity)
   c) Compute cumulative distribution function (CDF)
   d) Apply equalization using CDF
   e) Save enhanced image and timing data

3. Key Components:
   - Image Processing: Using OpenCV
   - Histogram Calculation: Atomic operations
   - Memory Handling: Image data transfers
   - Timing: CUDA events for performance measurement

4. Technical Details:
   - Input: Grayscale image (0-255 intensity)
   - Atomic Add: Thread-safe counting
   - Output: Enhanced image with better contrast
   - Results: Saved in results/gpu/ directory

5. Required Files:
   - Input image: images/input.jpg
   - Output directory: results/gpu/
   - Timing file: results/speedup.txt
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <fstream>

using namespace std;
using namespace cv;

// Kernel to compute histogram of image
__global__ void computeHistogram(unsigned char* input, int* histogram, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(&histogram[input[tid]], 1);
    }
}

// Kernel to apply histogram equalization using pre-computed CDF
__global__ void equalizeHistogram(unsigned char* input, unsigned char* output, float* cdf, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        output[tid] = (unsigned char)(cdf[input[tid]] * 255.0f);
    }
}

int main() {
    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 1. Load input image
    Mat image = imread("images/input.jpg", IMREAD_GRAYSCALE);
    if(image.empty()) {
        cout << "Could not read input image" << endl;
        return -1;
    }

    int size = image.rows * image.cols;
    
    // 2. Allocate memory on GPU
    unsigned char *d_input, *d_output;
    int *d_histogram;
    float *d_cdf;
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_histogram, 256 * sizeof(int));
    cudaMalloc(&d_cdf, 256 * sizeof(float));
    
    // 3. Copy image to GPU and initialize histogram
    cudaMemcpy(d_input, image.data, size, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, 256 * sizeof(int));

    // 4. Compute histogram on GPU
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    computeHistogram<<<numBlocks, blockSize>>>(d_input, d_histogram, size);
    
    // 5. Get histogram and compute CDF on CPU (small data, not worth GPU overhead)
    int histogram[256];
    cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    
    float cdf[256];
    float sum = 0;
    for(int i = 0; i < 256; i++) {
        sum += histogram[i];
        cdf[i] = sum / size;
    }
    
    // 6. Copy CDF back to GPU and equalize image
    cudaMemcpy(d_cdf, cdf, 256 * sizeof(float), cudaMemcpyHostToDevice);
    equalizeHistogram<<<numBlocks, blockSize>>>(d_input, d_output, d_cdf, size);
    
    // 7. Get result back and save
    Mat result(image.rows, image.cols, CV_8UC1);
    cudaMemcpy(result.data, d_output, size, cudaMemcpyDeviceToHost);
    
    system("mkdir -p results/gpu");
    imwrite("results/gpu/equalized.jpg", result);
    
    // After all GPU operations, record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "GPU Execution Time: " << milliseconds << " ms" << endl;
    
    // Record execution time
    ofstream results("results/speedup.txt", ios::app);
    results << "GPU " << milliseconds << " ms\n";
    results.close();
    
    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 8. Cleanup GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histogram);
    cudaFree(d_cdf);
    
    return 0;
}

/*
Compilation and Execution:
------------------------
Compilation:
nvcc gpu_histogram.cu -o gpu `pkg-config --cflags --libs opencv4`

Alternative (if pkg-config not available):
nvcc gpu_histogram.cu -o gpu -I/usr/include/opencv4 -lopencv_core -lopencv_imgcodecs -lopencv_highgui

Execution:
./gpu
*/
