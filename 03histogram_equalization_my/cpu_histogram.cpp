// Summary:
// This program performs histogram equalization on a grayscale image using OpenCV.
// It measures the execution time for the operation, saves the equalized image,
// and logs the execution time to a file. The results are stored in the "results/cpu" directory.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

// Function to create directory if it doesn't exist
void createDirectory(const string &path)
{
    string command = "mkdir -p " + path;
    system(command.c_str());
}

int main()
{
    createDirectory("results/cpu");
    auto start = high_resolution_clock::now();

    // Load grayscale image
    Mat image = imread("images/input.jpg", IMREAD_GRAYSCALE);
    if (image.empty())
    {
        cerr << "Error: Could not load input image at images/input.jpg" << endl;
        return -1;
    }

    // Apply histogram equalization
    Mat equalizedImage;
    equalizeHist(image, equalizedImage);

    // Save the equalized image
    imwrite("results/cpu/equalized.jpg", equalizedImage);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "CPU Execution Time: " << duration.count() << " ms" << endl;

    // Record execution time
    ofstream results("results/speedup.txt", ios::app);
    results << "CPU " << duration.count() << " ms\n";
    results.close();

    return 0;
}

/*
Execution Command:
g++ -o cpu_histogram cpu_histogram.cpp `pkg-config --cflags --libs opencv4`
./cpu_histogram
 */