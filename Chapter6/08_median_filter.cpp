#include <iostream>
#include "opencv2/opencv.hpp"

int main ()
{
    cv::Mat h_img1 = cv::imread("images/saltpepper.png",0);
	cv::Mat h_result;
	cv::medianBlur(h_img1,h_result,3);
    cv::imshow("Original Image ", h_img1);
	cv::imshow("Median Blur Result", h_result);
	cv::waitKey();
    return 0;
}



















