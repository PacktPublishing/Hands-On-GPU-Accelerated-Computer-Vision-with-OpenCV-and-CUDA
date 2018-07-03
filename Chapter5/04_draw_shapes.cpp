#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
 //create a new image which consists of 
 //3 channels 
 //image depth of 8 bits 
 //800 x 600 of resolution (800 wide and 600 high)
 //each pixels initialized to the value of (100, 250, 30) for Blue, Green and Red planes respectively.
 Mat img(512, 512, CV_8UC3, Scalar(0,0,0)); 
 line(img,Point(0,0),Point(511,511),Scalar(0,255,0),7);
rectangle(img,Point(384,0),Point(510,128),Scalar(255,255,0),5);
circle(img,Point(447,63), 63, Scalar(0,0,255), -1);
ellipse(img,Point(256,256),Point(100,100),0,0,180,255,-1);
    putText( img, "OpenCV!", Point(10,500), FONT_HERSHEY_SIMPLEX, 3,
           Scalar(255, 255, 255), 5, 8 );
 String win_name = "Blank Blue Color Image"; //Name of the window

 namedWindow(win_name); // Create a window

 imshow(win_name, img); // Show our image inside the created window.

 waitKey(0); // Wait for any keystroke in the window

 destroyWindow(win_name); //destroy the created window

 return 0;
}
