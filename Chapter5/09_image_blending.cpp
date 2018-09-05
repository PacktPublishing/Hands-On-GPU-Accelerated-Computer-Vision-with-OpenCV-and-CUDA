#include <iostream>
#include "opencv2/opencv.hpp"

int main (int argc, char* argv[])
{
    //Read Two Images 
    cv::Mat h_img1 = cv::imread("images/cameraman.tif");
    cv::Mat h_img2 = cv::imread("images/circles.png");
    //Create Memory for storing Images on device
    cv::cuda::GpuMat d_result1,d_img1, d_img2;
    cv::Mat h_result1;
    //Upload Images to device     
    d_img1.upload(h_img1);
    d_img2.upload(h_img2);

    cv::cuda::addWeighted(d_img1,0.7,d_img2,0.3,0,d_result1);
    //Download Result back to host
    d_result1.download(h_result1);
    cv::imshow("Image1 ", h_img1);
    cv::imshow("Image2 ", h_img2);
    cv::imshow("Result blending ", h_result1);
    cv::imwrite("images/result_add.png", h_result1);
    cv::waitKey();
    return 0;
}
