#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <ctime>

using namespace std;

#define SIZE 256

int main()
{
    // Load grayscale image
    cv::Mat img = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Error: Could not load image!" << endl;
        return -1;
    }

    // Resize image to 1024x1024
    cv::resize(img, img, cv::Size(1024, 1024));
    int w = img.cols, h = img.rows, total = w * h;

    // Allocate host memory
    unsigned char *h_in = img.data;
    unsigned char *h_out = new unsigned char[total];
    int h_hist[SIZE] = {0};
    int h_cdf[SIZE] = {0};

    // Start timing
    clock_t start = clock();

// Compute histogram on device
#pragma acc data copyin(h_in[0 : total]) copyout(h_out[0 : total]) create(h_hist[0 : SIZE], h_cdf[0 : SIZE])
    {
// Initialize histogram
#pragma acc parallel loop
        for (int i = 0; i < SIZE; i++)
        {
            h_hist[i] = 0;
        }

// Compute histogram
#pragma acc parallel loop reduction(+ : h_hist[ : SIZE])
        for (int i = 0; i < total; i++)
        {
            int val = h_in[i];
            h_hist[val]++;
        }

        // Compute CDF
        int sum = 0;
#pragma acc parallel loop seq
        for (int i = 0; i < SIZE; i++)
        {
            sum += h_hist[i];
            h_cdf[i] = (sum * 255) / total;
        }

// Apply histogram equalization
#pragma acc parallel loop
        for (int i = 0; i < total; i++)
        {
            h_out[i] = h_cdf[h_in[i]];
        }
    }

    // End timing
    clock_t end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC * 1000;
    cout << "\n<<<<<< ..... Histogram Equalization (OpenACC) ..... >>>>>>\n";
    cout << "\nTime taken: " << duration << " ms" << endl;

    // Create result image
    cv::Mat result(h, w, CV_8U, h_out);

    // Calculate old & new image pixel intensities
    map<int, int> oldIntensities, newIntensities;
    for (int i = 0; i < total; i++)
        oldIntensities[h_in[i]]++;
    for (int i = 0; i < total; i++)
        newIntensities[h_out[i]]++;

    // Compute average pixels per intensity
    int old_total_used_intensities = oldIntensities.size();
    int new_total_used_intensities = newIntensities.size();
    double old_avg_pixels_per_intensity = old_total_used_intensities > 0 ? (double)total / old_total_used_intensities : 0;
    double new_avg_pixels_per_intensity = new_total_used_intensities > 0 ? (double)total / new_total_used_intensities : 0;

    // Display old image pixel intensities
    cout << "\n\n<<<<<< ..... Old Image Pixel Intensities ..... >>>>>>\n"
         << endl;
    cout << "Average Pixels Per Intensity: " << old_avg_pixels_per_intensity << " pixels\n"
         << endl;
    for (auto &pair : oldIntensities)
    {
        cout << "Intensity " << pair.first << " : " << pair.second << " pixels" << endl;
    }

    // Display new image pixel intensities
    cout << "\n\n<<<<<< ..... New Image Pixel Intensities ..... >>>>>>\n"
         << endl;
    cout << "Average Pixels Per Intensity: " << new_avg_pixels_per_intensity << " pixels\n"
         << endl;
    for (auto &pair : newIntensities)
    {
        cout << "Intensity " << pair.first << " : " << pair.second << " pixels" << endl;
    }

    // Save the equalized image
    cv::imwrite("result_OpenACC.jpg", result);

    // Free host memory
    delete[] h_out;

    return 0;
}

/*
Theory :
OpenACC Optimizations:
#pragma acc data region to manage all data movement

#pragma acc parallel loop for parallelizing all compute-intensive loops

reduction(+:h_hist[:SIZE]) for the histogram computation

#pragma acc parallel loop seq for the sequential CDF computation
*/

// Compile : pgc++ -acc -ta=tesla:managed -fast histogramEqualization_acc.cu -o histogramEqualization_acc `pkg-config --cflags --libs opencv4`
// Run : ./histogramEqualization_acc

// alternative command
// g++ -O3 -fopenacc -o histogramEqualization_acc histogramEqualization_acc.cpp `pkg-config --cflags --libs opencv4`
// Run : ./histogramEqualization_acc
