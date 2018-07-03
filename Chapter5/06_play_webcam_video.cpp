#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
 //open the Webcam
 VideoCapture cap(0); 
 // if not success, exit program
 if (cap.isOpened() == false)  
 {
  cout << "Cannot open Webcam" << endl;
  return -1;
 }
 //get the frames rate of the video
 double fps = cap.get(CAP_PROP_FPS); 
 cout << "Frames per seconds : " << fps << endl;
cout<<"Press Q to Quit" <<endl;
 String win_name = "Webcam Video";
 namedWindow(win_name); //create a window
 while (true)
 {
  Mat frame;
  bool flag = cap.read(frame); // read a new frame from video 
  //show the frame in the created window
  imshow(win_name, frame);
  if (waitKey(1) == 'q')
  {
      break;
  }
 }
return 0;
}
