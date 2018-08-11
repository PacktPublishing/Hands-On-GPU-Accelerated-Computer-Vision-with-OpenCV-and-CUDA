#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main()
{
	
    Mat h_img1;
	cv::cuda::GpuMat d_img1,d_blur,d_result3x3;
    h_img1 = imread("images/blobs.png",1);
	int64 start = cv::getTickCount();
	d_img1.upload(h_img1);
	cv::cuda::cvtColor(d_img1,d_img1,cv::COLOR_BGR2GRAY);
	cv::Ptr<cv::cuda::Filter> filter3x3;
   	filter3x3 = cv::cuda::createGaussianFilter(CV_8UC1,CV_8UC1,cv::Size(3,3),1);
	filter3x3->apply(d_img1, d_blur);
    cv::Ptr<cv::cuda::Filter> filter1;
   	filter1 = cv::cuda::createLaplacianFilter(CV_8UC1,CV_8UC1,1);
	filter1->apply(d_blur, d_result3x3);
    cv::Mat h_result3x3,h_blur;
    d_result3x3.download(h_result3x3);
	d_blur.download(h_blur);
    double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
    std::cout << "FPS : " << fps << std::endl;
    imshow("Laplacian", h_result3x3);
	imshow("Blurred", h_blur);
    cv::waitKey();
    return 0;
}
