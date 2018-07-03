#include <iostream>
#include "opencv2/opencv.hpp"

int main (int argc, char* argv[])
{
    cv::Mat h_img1 = cv::imread("images/circles.png");
    //Create Device variables
    cv::cuda::GpuMat d_result1,d_img1;
    cv::Mat h_result1;     
    //Upload Image to device
    d_img1.upload(h_img1);

    cv::cuda::bitwise_not(d_img1,d_result1);
    
    //Download result back  to host
    d_result1.download(h_result1);
    cv::imshow("Result inversion ", h_result1);
    cv::imwrite("images/result_inversion.png", h_result1);
    cv::waitKey();
    return 0;
}