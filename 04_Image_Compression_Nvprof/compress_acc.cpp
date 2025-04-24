#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024

void compressImage(const cv::Mat &inputImg, cv::Mat &outputImg, int factor)
{
    int newSize = WIDTH / factor;
    outputImg.create(newSize, newSize, CV_8UC1);

#pragma acc data copyin(inputImg.data[0 : WIDTH * HEIGHT]) copyout(outputImg.data[0 : newSize * newSize])
    {
#pragma acc parallel loop collapse(2)
        for (int y = 0; y < newSize; y++)
        {
            for (int x = 0; x < newSize; x++)
            {
                int sum = 0;
#pragma acc loop reduction(+ : sum)
                for (int j = 0; j < factor; j++)
                {
                    for (int i = 0; i < factor; i++)
                    {
                        sum += inputImg.data[(y * factor + j) * WIDTH + (x * factor + i)];
                    }
                }
                outputImg.data[y * newSize + x] = sum / (factor * factor);
            }
        }
    }
}

int main()
{
    // Load input image
    cv::Mat inputImg = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImg.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Resize input image to fixed width and height
    cv::resize(inputImg, inputImg, cv::Size(WIDTH, HEIGHT));

    // Start measuring time
    clock_t start = clock();

    // Compress the image with different factors
    cv::Mat outputImg2, outputImg4;
    compressImage(inputImg, outputImg2, 2);
    compressImage(inputImg, outputImg4, 4);

    // Save the compressed images
    cv::imwrite("compressed_2x_OpenACC.jpg", outputImg2);
    cv::imwrite("compressed_4x_OpenACC.jpg", outputImg4);

    // End measuring time
    clock_t end = clock();

    // Calculate and print the elapsed time
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Compression complete! Time taken: " << elapsed_time << " seconds" << std::endl;

    return 0;
}

// Compile : pgc++ -acc -ta=tesla:managed -fast compress_acc.cu -o compress_acc `pkg-config --cflags --libs opencv4`
// Run : ./compress_acc

// Alternative command
// g++ -O3 -fopenacc -o compress_acc compress_acc.cpp `pkg-config --cflags --libs opencv4`
// Run : ./compress_acc

/*
Theory :
OpenACC Optimizations:
#pragma acc data region to manage all data movement

#pragma acc parallel loop collapse(2) to parallelize the outer two loops

#pragma acc loop reduction(+:sum) for the pixel summation reduction

-ta=tesla:managed enables NVIDIA GPU acceleration with unified memory
*/