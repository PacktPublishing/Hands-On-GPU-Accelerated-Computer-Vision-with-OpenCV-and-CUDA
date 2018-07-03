#include <iostream>
#include "opencv2/opencv.hpp"

int main (int argc, char* argv[])
{
        cv::Mat h_img1 = cv::imread("images/cameraman.tif", 0);
        //Define device variables
        cv::cuda::GpuMat d_result1,d_result2,d_result3,d_result4,d_result5, d_img1;
        //Upload image on device
        d_img1.upload(h_img1);

        //Perform different thresholding techniques on device
        cv::cuda::threshold(d_img1, d_result1, 128.0, 255.0, cv::THRESH_BINARY);
        cv::cuda::threshold(d_img1, d_result2, 128.0, 255.0, cv::THRESH_BINARY_INV);
        cv::cuda::threshold(d_img1, d_result3, 128.0, 255.0, cv::THRESH_TRUNC);
        cv::cuda::threshold(d_img1, d_result4, 128.0, 255.0, cv::THRESH_TOZERO);
        cv::cuda::threshold(d_img1, d_result5, 128.0, 255.0, cv::THRESH_TOZERO_INV);

        cv::Mat h_result1,h_result2,h_result3,h_result4,h_result5;
        //Copy results back to host
        d_result1.download(h_result1);
        d_result2.download(h_result2);
        d_result3.download(h_result3);
        d_result4.download(h_result4);
        d_result5.download(h_result5);
        cv::imshow("Result Threshhold binary ", h_result1);
        cv::imshow("Result Threshhold binary inverse ", h_result2);
        cv::imshow("Result Threshhold truncated ", h_result3);
        cv::imshow("Result Threshhold truncated to zero ", h_result4);
        cv::imshow("Result Threshhold truncated to zero inverse ", h_result5);
        cv::waitKey();

    return 0;
}
