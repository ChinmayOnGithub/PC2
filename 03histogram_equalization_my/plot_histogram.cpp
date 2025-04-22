#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void plotHistogram(const Mat &image, const string &windowName)
{
  // Calculate histogram
  vector<Mat> bgr_planes;
  split(image, bgr_planes);

  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange = {range};

  Mat hist;
  calcHist(&bgr_planes[0], 1, 0, Mat(), hist, 1, &histSize, &histRange);

  // Create histogram visualization
  int hist_w = 512, hist_h = 400;
  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

  // Normalize histogram for visualization
  normalize(hist, hist, 0, hist_h, NORM_MINMAX);

  // Draw histogram
  for (int i = 0; i < histSize; i++)
  {
    line(histImage,
         Point(i * 2, hist_h),
         Point(i * 2, hist_h - cvRound(hist.at<float>(i))),
         Scalar(255, 255, 255),
         2);
  }

  // Save histogram image
  imwrite("results/" + windowName + "_histogram.jpg", histImage);
  cout << "Saved histogram for " << windowName << endl;
}

int main()
{
  // Read input and output images
  Mat input_img = imread("images/input.jpg", IMREAD_GRAYSCALE);
  Mat output_img = imread("results/gpu/equalized.jpg", IMREAD_GRAYSCALE); // Using GPU output

  if (input_img.empty() || output_img.empty())
  {
    cerr << "Error: Could not read images!" << endl;
    cerr << "Make sure images/input.jpg and results/gpu/equalized.jpg exist." << endl;
    return -1;
  }

  // Plot and save histograms
  plotHistogram(input_img, "input");
  plotHistogram(output_img, "equalized");

  cout << "Histograms have been saved in the results folder:" << endl;
  cout << "- results/input_histogram.jpg (Original image histogram)" << endl;
  cout << "- results/equalized_histogram.jpg (After equalization)" << endl;
  return 0;
}