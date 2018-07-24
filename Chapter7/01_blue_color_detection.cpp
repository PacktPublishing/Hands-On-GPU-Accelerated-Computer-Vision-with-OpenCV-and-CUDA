#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

 int main( int argc, char** argv )
 {
    VideoCapture cap(0); //capture the video from web cam
	// if webcam is not available then exit the program
    if ( !cap.isOpened() )  
    {
         cout << "Cannot open the web cam" << endl;
         return -1;
    }
    while (true)
    {
        Mat frame;
// read a new frame from webcam
        bool flag = cap.read(frame); 
         if (!flag) 
        {
             cout << "Cannot read a frame from webcam" << endl;
             break;
        }
  
cuda::GpuMat d_frame, d_frame_hsv,d_intermediate,d_result;
cuda::GpuMat d_frame_shsv[3];
cuda::GpuMat d_thresc[3];
Mat h_result;
d_frame.upload(frame);
//Transform image to HSV
cuda::cvtColor(d_frame, d_frame_hsv, COLOR_BGR2HSV);

//Split HSV 3 channels
cuda::split(d_frame_hsv, d_frame_shsv);

//Threshold HSV channels
cuda::threshold(d_frame_shsv[0], d_thresc[0], 110, 130, THRESH_BINARY);
cuda::threshold(d_frame_shsv[1], d_thresc[1], 50, 255, THRESH_BINARY);
cuda::threshold(d_frame_shsv[2], d_thresc[2], 50, 255, THRESH_BINARY);

//Bitwise AND the channels
cv::cuda::bitwise_and(d_thresc[0], d_thresc[1],d_intermediate);
cv::cuda::bitwise_and(d_intermediate, d_thresc[2], d_result);

d_result.download(h_result);
imshow("Thresholded Image", h_result); 
imshow("Original", frame); 

        if (waitKey(1) == 'q') 
       {
            break; 
       }
    }
   return 0;
}
