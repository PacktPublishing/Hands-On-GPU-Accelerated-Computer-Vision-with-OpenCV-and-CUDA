#include <iostream>
#include "opencv2/opencv.hpp"


int main ()
{
    cv::Mat h_img1 = cv::imread("images/cameraman.tif",0);
    cv::cuda::GpuMat d_img1,d_result3x3,d_result5x5,d_result7x7;
 
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filter3x3,filter5x5,filter7x7;
   	filter3x3 = cv::cuda::createBoxFilter(CV_8UC1,CV_8UC1,cv::Size(3,3));
	filter3x3->apply(d_img1, d_result3x3);
	filter5x5 = cv::cuda::createBoxFilter(CV_8UC1,CV_8UC1,cv::Size(5,5));
	filter5x5->apply(d_img1, d_result5x5);
	filter7x7 = cv::cuda::createBoxFilter(CV_8UC1,CV_8UC1,cv::Size(7,7));
	filter7x7->apply(d_img1, d_result7x7);

    cv::Mat h_result3x3,h_result5x5,h_result7x7;
    d_result3x3.download(h_result3x3);
	d_result5x5.download(h_result5x5);
	d_result7x7.download(h_result7x7);
	

    cv::imshow("Original Image ", h_img1);
	cv::imshow("Blurred with kernel size 3x3", h_result3x3);
	cv::imshow("Blurred with kernel size 5x5", h_result5x5);
	cv::imshow("Blurred with kernel size 7x7", h_result7x7);
	cv::imwrite("Blurred3x3.png", h_result3x3);
	cv::imwrite("Blurred5x5.png", h_result5x5);
	cv::imwrite("Blurred7x7.png", h_result7x7);

	cv::waitKey();
    return 0;
}



















