#include <iostream>
#include "opencv2/opencv.hpp"


int main ()
{
    cv::Mat h_img1 = cv::imread("images/autumn.tif");
    cv::Mat h_img2,h_result1;
    cvtColor(h_img1, h_img2, cv::COLOR_BGR2HSV);
    //Split the image into 3 channels; H, S and V channels respectively and store it in a std::vector
    std::vector< cv::Mat > vec_channels;
    cv::split(h_img2, vec_channels); 
    //Equalize the histogram of only the V channel 
    cv::equalizeHist(vec_channels[2], vec_channels[2]);
    //Merge 3 channels in the vector to form the color image in HSV color space.
    cv::merge(vec_channels, h_img2); 
      
    //Convert the histogram equalized image from HSV to BGR color space again
    cv::cvtColor(h_img2,h_result1, cv::COLOR_HSV2BGR);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Histogram Equalized Image", h_result1);
    cv::waitKey();
    return 0;
}
