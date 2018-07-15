#include <iostream>
#include "opencv2/opencv.hpp"


int main ()
{
    cv::Mat h_img1 = cv::imread("images/blobs.png",0);
    cv::cuda::GpuMat d_img1,d_resultx,d_resulty,d_resultxy;
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filterx,filtery;
   	filterx = cv::cuda::createScharrFilter(CV_8UC1,CV_8UC1,1,0);
	filterx->apply(d_img1, d_resultx);
	filtery = cv::cuda::createScharrFilter(CV_8UC1,CV_8UC1,0,1);
	filtery->apply(d_img1, d_resulty);
	cv::cuda::add(d_resultx,d_resulty,d_resultxy);        
	cv::Mat h_resultx,h_resulty,h_resultxy;
    d_resultx.download(h_resultx);
	d_resulty.download(h_resulty);
	d_resultxy.download(h_resultxy);
	cv::imshow("Original Image ", h_img1);
	cv::imshow("Scharr-x derivative", h_resultx);
	cv::imshow("Scharr-y derivative", h_resulty);
	cv::imshow("Scharr-xy derivative", h_resultxy);
	cv::imwrite("scharrx.png", h_resultx);
	cv::imwrite("scharry.png", h_resulty);
	cv::imwrite("scharrxy.png", h_resultxy);
	cv::waitKey();
    return 0;
}



















