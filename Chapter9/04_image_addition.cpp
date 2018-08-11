#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
//#include <cuda_runtime.h>

int main (int argc, char* argv[])
{
    //Read Two Images 
    cv::Mat h_img1 = cv::imread("images/cameraman.tif");
    cv::Mat h_img2 = cv::imread("images/circles.png");
	int64 work_begin = cv::getTickCount(); 
    //Create Memory for storing Images on device
    cv::cuda::GpuMat d_result1,d_img1, d_img2;
    cv::Mat h_result1;
    //Upload Images to device     
    d_img1.upload(h_img1);
	d_img2.upload(h_img2);

    cv::cuda::add(d_img1,d_img2, d_result1);
    //Download Result back to host
    d_result1.download(h_result1);
	int64 delta = cv::getTickCount() - work_begin;
	double freq = cv::getTickFrequency();
	double work_fps = freq / delta;
	std::cout<<"Performance of Addition on Jetson TX1: " <<std::endl;
	std::cout <<"Time: " << (1/work_fps) <<std::endl;
	std::cout <<"FPS: " <<work_fps <<std::endl;
	cv::imshow("Image1 ", h_img1);
	cv::imshow("Image2 ", h_img2);
	cv::imshow("Result addition ", h_result1);
    cv::imwrite("result_add.png", h_result1);
    cv::waitKey();
    return 0;
}
