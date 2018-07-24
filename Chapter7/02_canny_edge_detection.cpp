#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;


int main()
{
    Mat h_image = imread("images/drawing.JPG",0);
    if (h_image.empty())
    {
        cout << "can not open image"<< endl;
        return -1;
    }
	GpuMat d_edge,d_image;
    Mat h_edge;
	d_image.upload(h_image);
	cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edge = cv::cuda::createCannyEdgeDetector(2.0, 100.0, 3, false);
	canny_edge->detect(d_image, d_edge);
    d_edge.download(h_edge);
imshow("source", h_image);
    imshow("detected edges", h_edge);
	 waitKey(0);

    return 0;
}