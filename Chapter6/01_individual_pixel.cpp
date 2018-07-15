#include <iostream>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"

int main ()
{
    cv::Mat h_img1 = cv::imread("images/cameraman.tif",0);
	cv::Scalar intensity = h_img1.at<uchar>(cv::Point(100, 50));
	std::cout<<"Pixel Intensity of gray scale Image at (100,50) is:"<<intensity.val[0]<<std::endl;
    cv::Mat h_img2 = cv::imread("images/autumn.tif",1);
	cv::Vec3b intensity1 = h_img1.at<cv::Vec3b>(cv::Point(100, 50));
	std::cout<<"Pixel Intensity of color Image at (100,50) is:"<<intensity1<<std::endl;
    return 0;
}



















