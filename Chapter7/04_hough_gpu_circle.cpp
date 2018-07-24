#include "opencv2/opencv.hpp"

#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    Mat h_image = imread("images/eight.tif", IMREAD_COLOR);
    Mat h_gray;
    cvtColor(h_image, h_gray, COLOR_BGR2GRAY);
	cuda::GpuMat d_gray,d_result;
	std::vector<cv::Vec3f> d_Circles;
medianBlur(h_gray, h_gray, 5);
cv::Ptr<cv::cuda::HoughCirclesDetector> detector = cv::cuda::createHoughCirclesDetector(1, 100, 122, 50, 1, max(h_image.size().width, h_image.size().height));
  d_gray.upload(h_gray);
  detector->detect(d_gray, d_result);
  d_Circles.resize(d_result.size().width);
  if (!d_Circles.empty())
    d_result.row(0).download(cv::Mat(d_Circles).reshape(3, 1));

cout<<"No of circles: " <<d_Circles.size() <<endl;
    for( size_t i = 0; i < d_Circles.size(); i++ )
    {
        Vec3i cir = d_Circles[i];
        circle( h_image, Point(cir[0], cir[1]), cir[2], Scalar(255,0,0), 2, LINE_AA);
    }
    imshow("detected circles", h_image);
    waitKey(0);

    return 0;
}
