#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

void displayComparison()
{
    Mat img_cpu = imread("results/cpu/equalized.jpg", IMREAD_GRAYSCALE);
    Mat img_gpu = imread("results/gpu/equalized.jpg", IMREAD_GRAYSCALE);

    if (img_cpu.empty() || img_gpu.empty())
    {
        cerr << "Error: Could not load equalized images! Check results/cpu/ and results/gpu/ folders." << endl;
        return;
    }

    int display_size = 512;
    resize(img_cpu, img_cpu, Size(display_size, display_size));
    resize(img_gpu, img_gpu, Size(display_size, display_size));

    putText(img_cpu, "CPU Equalized", Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);
    putText(img_gpu, "GPU Equalized", Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    Mat final_comparison;
    hconcat(img_cpu, img_gpu, final_comparison);
    imwrite("results/comparison.jpg", final_comparison);
}

void compareExecutionTime()
{
    ifstream file("results/speedup.txt");
    if (!file)
    {
        cerr << "Error: Could not open execution time file!" << endl;
        return;
    }

    double cpu_time = -1, gpu_time = -1;
    string line, label;

    while (getline(file, line))
    {
        istringstream iss(line);
        string unit;
        double time;

        if (!(iss >> label >> time >> unit))
        {
            continue;
        }

        if (label == "CPU")
            cpu_time = time;
        else if (label == "GPU")
            gpu_time = time;
    }

    file.close();

    cout << "===================================" << endl;
    cout << " CPU Execution Time: " << (cpu_time >= 0 ? to_string(cpu_time) + " ms" : "Not recorded") << endl;
    cout << " GPU Execution Time: " << (gpu_time >= 0 ? to_string(gpu_time) + " ms" : "Not recorded") << endl;

    if (cpu_time >= 0 && gpu_time >= 0)
    {
        double speedup = cpu_time / gpu_time;
        cout << " Speedup Factor: " << speedup << "x" << endl;
    }
    else
    {
        cout << " Speedup Factor: Cannot calculate - missing timing data" << endl;
    }
    cout << "===================================" << endl;
}

int main()
{
    compareExecutionTime();
    displayComparison();
    return 0;
}
