#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  Mat img = imread("images/cameraman.tif",0);
 if (img.empty()) 
 {
  cout << "Could not open an image" << endl;
  return -1;
 }
imshow("Image Read on Jetson TX1"; , img); 
waitKey(0); 
return 0;
}
