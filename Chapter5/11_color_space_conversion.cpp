#include <iostream>
#include "opencv2/opencv.hpp"

int main (int argc, char* argv[])
{
        cv::Mat h_img1 = cv::imread("images/autumn.tif");
        //Define device variables
        cv::cuda::GpuMat d_result1,d_result2,d_result3,d_result4,d_img1;
        //Upload Image to device
        d_img1.upload(h_img1);

        //Convert image to different color spaces
        cv::cuda::cvtColor(d_img1, d_result1,cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(d_img1, d_result2,cv::COLOR_BGR2RGB);
        cv::cuda::cvtColor(d_img1, d_result3,cv::COLOR_BGR2HSV);
        cv::cuda::cvtColor(d_img1, d_result4,cv::COLOR_BGR2YCrCb);
        
        cv::Mat h_result1,h_result2,h_result3,h_result4;
        //Download results back to host
        d_result1.download(h_result1);
        d_result2.download(h_result2);
        d_result3.download(h_result3);
        d_result4.download(h_result4);
 
        cv::imshow("Result in Gray ", h_result1);
        cv::imshow("Result in RGB", h_result2);
        cv::imshow("Result in HSV ", h_result3);
        cv::imshow("Result in YCrCb ", h_result4);
        
        cv::waitKey();
        return 0;
}