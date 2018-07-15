#include <iostream>
#include "opencv2/opencv.hpp"


int main ()
{
    cv::Mat h_img1 = cv::imread("images/blobs.png",0);
    cv::cuda::GpuMat d_img1,d_resulte,d_resultd,d_resulto, d_resultc;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));	
	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filtere,filterd,filtero,filterc;
   	filtere = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,CV_8UC1,element);
	filtere->apply(d_img1, d_resulte);
	filterd = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,CV_8UC1,element);
	filterd->apply(d_img1, d_resultd);
	filtero = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN,CV_8UC1,element);
	filtero->apply(d_img1, d_resulto);
	filterc = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE,CV_8UC1,element);
	filterc->apply(d_img1, d_resultc);
	        
	cv::Mat h_resulte,h_resultd,h_resulto,h_resultc;
    d_resulte.download(h_resulte);
	d_resultd.download(h_resultd);
	d_resulto.download(h_resulto);
	d_resultc.download(h_resultc);
    cv::imshow("Original Image ", h_img1);
	cv::imshow("Erosion", h_resulte);
	cv::imshow("Dilation", h_resultd);
	cv::imshow("Opening", h_resulto);
	cv::imshow("closing", h_resultc);
	cv::imwrite("erosion7.png", h_resulte);
	cv::imwrite("dilation7.png", h_resultd);
	cv::imwrite("opening7.png", h_resulto);
	cv::imwrite("closing7.png", h_resultc);
	cv::waitKey();
    return 0;
}



















