#include <iostream>
#include "opencv2/opencv.hpp"


int main ()
{
    cv::Mat h_img1 = cv::imread("images/blobs.png",0);
    cv::cuda::GpuMat d_img1,d_result1,d_result3;
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filter1,filter3;
   	filter1 = cv::cuda::createLaplacianFilter(CV_8UC1,CV_8UC1,1);
	filter1->apply(d_img1, d_result1);
	filter3 = cv::cuda::createLaplacianFilter(CV_8UC1,CV_8UC1,3);
	filter3->apply(d_img1, d_result3);
	cv::Mat h_result1,h_result3;
    d_result1.download(h_result1);
	d_result3.download(h_result3);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Laplacian Filter 1", h_result1);
	cv::imshow("Laplacian Filter 3", h_result3);
	cv::imwrite("laplacian1.png", h_result1);
	cv::imwrite("laplacian3.png", h_result3);
	cv::waitKey();
    return 0;
}



















