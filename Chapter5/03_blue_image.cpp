#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
 
 //Create blank blue color Image with size 256x256
 Mat img(256, 256, CV_8UC3, Scalar(255,0,0)); 
 
 String win_name = "Blank Blue Color Image"; 

 namedWindow(win_name); 

 imshow(win_name, img); 

 waitKey(0); 

 destroyWindow(win_name); 

 return 0;
}
