#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
 // Read the image 
 Mat img = imread("images/cameraman.tif",0);

 // Check for failure in reading an Image
 if (img.empty()) 
 {
  cout << "Could not open an image" << endl;
  return -1;
 }
//Name of the window
 String win_name = "My First Opencv Program"; 

 // Create a window
 namedWindow(win_name); 

 // Show our image inside the created window.
imshow(win_name, img); 

// Wait for any keystroke in the window 
waitKey(0); 

//destroy the created window
 destroyWindow(win_name); 

 return 0;
}