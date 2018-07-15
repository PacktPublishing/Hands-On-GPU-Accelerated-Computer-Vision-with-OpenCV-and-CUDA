#include <iostream>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"

int main ()
{
    cv::Mat h_img1 = cv::imread("images/cameraman.tif",0);
    cv::cuda::GpuMat d_img1,d_result1,d_result2;
    d_img1.upload(h_img1);
	int width= d_img1.cols;
	int height = d_img1.size().height;
    cv::cuda::resize(d_img1,d_result1,cv::Size(200, 200), cv::INTER_CUBIC);
    cv::cuda::resize(d_img1,d_result2,cv::Size(0.5*width, 0.5*height), cv::INTER_LINEAR);    
    cv::Mat h_result1,h_result2;
    d_result1.download(h_result1);
	d_result2.download(h_result2);
    cv::imshow("Original Image ", h_img1);
	cv::imshow("Resized Image", h_result1);
	cv::imshow("Resized Image 2", h_result2);
	cv::imwrite("Resized1.png", h_result1);
	cv::imwrite("Resized2.png", h_result2);
	cv::waitKey();
    return 0;
}



















