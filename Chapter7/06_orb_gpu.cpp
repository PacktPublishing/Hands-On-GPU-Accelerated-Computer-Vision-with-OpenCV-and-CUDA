#include <iostream>
#include "opencv2/opencv.hpp"
 
using namespace cv;
using namespace std;
 
int main()
{
Mat h_image = imread( "images/drawing.JPG", 0 );
cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create();
std::vector<cv::KeyPoint> keypoints;
cv::cuda::GpuMat d_image;
d_image.upload(h_image);
detector->detect(d_image, keypoints);
cv::drawKeypoints(h_image,keypoints,h_image);
imshow("Final Result", h_image );
waitKey(0);
return 0;
}
