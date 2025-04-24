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
    std::cout << "Original size: " << WIDTH << "x" << HEIGHT << " pixels" << std::endl;
    std::cout << "2x compressed: " << WIDTH / 2 << "x" << HEIGHT / 2 << " pixels" << std::endl;
    std::cout << "4x compressed: " << WIDTH / 4 << "x" << HEIGHT / 4 << " pixels" << std::endl;

    /*
    // Uncomment to print detailed compression information
    std::cout << "\n----- Detailed Compression Information -----\n";

    // Calculate data reduction
    int originalBytes = WIDTH * HEIGHT;
    int compressed2xBytes = (WIDTH/2) * (HEIGHT/2);
    int compressed4xBytes = (WIDTH/4) * (HEIGHT/4);

    double reduction2x = 100.0 * (1.0 - (double)compressed2xBytes / originalBytes);
    double reduction4x = 100.0 * (1.0 - (double)compressed4xBytes / originalBytes);

    std::cout << "Original data size: " << originalBytes << " bytes\n";
    std::cout << "2x compressed data size: " << compressed2xBytes << " bytes ("
              << reduction2x << "% reduction)\n";
    std::cout << "4x compressed data size: " << compressed4xBytes << " bytes ("
              << reduction4x << "% reduction)\n";

    // Print sample pixel values to demonstrate the averaging
    std::cout << "\nSample compression for 2x (first 2x2 block):\n";
    std::cout << "Original pixels (2x2 block):\n";
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            std::cout << (int)inputImg.data[j * WIDTH + i] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "Averaged value: " << (int)outputImg2.data[0] << "\n";

    std::cout << "\nSample compression for 4x (first 4x4 block):\n";
    std::cout << "Original pixels (4x4 block):\n";
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            std::cout << (int)inputImg.data[j * WIDTH + i] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "Averaged value: " << (int)outputImg4.data[0] << "\n";
    */

    return 0;
}

// Compile : pgc++ -acc -ta=tesla:managed -fast compress_acc.cpp -o compress_acc `pkg-config --cflags --libs opencv4`
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