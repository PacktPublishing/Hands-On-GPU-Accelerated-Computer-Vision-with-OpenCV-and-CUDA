#include <iostream>
#include "opencv2/opencv.hpp"

int main ()
{
        cv::Mat h_img1 = cv::imread("images/cameraman.tif",0);
        cv::cuda::GpuMat d_img1,d_result1,d_result2;
        d_img1.upload(h_img1);
		int cols= d_img1.cols;
		int rows = d_img1.size().height;
		//Translation
		cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, 70, 0, 1, 50);
		cv::cuda::warpAffine(d_img1,d_result1,trans_mat,d_img1.size());
		//Rotation
		cv::Point2f pt(d_img1.cols/2., d_img1.rows/2.);    
		cv::Mat r = cv::getRotationMatrix2D(pt, 45, 1.0);
		cv::cuda::warpAffine(d_img1, d_result2, r, cv::Size(d_img1.cols, d_img1.rows));
        cv::Mat h_result1,h_result2;
        d_result1.download(h_result1);
		d_result2.download(h_result2);
        cv::imshow("Original Image ", h_img1);
		cv::imshow("Translated Image", h_result1);
		cv::imshow("Rotated Image", h_result2);
		cv::imwrite("Translated.png", h_result1);
		cv::imwrite("Rotated.png", h_result2);
		cv::waitKey();
		return 0;
}



















