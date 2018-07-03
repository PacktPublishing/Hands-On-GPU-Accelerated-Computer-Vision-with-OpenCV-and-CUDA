#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main(int argc, char* argv[])
{
 //open the video file from PC
 VideoCapture cap("images/rhinos.avi"); 
 // if not success, exit program
 if (cap.isOpened() == false) 
 {
  cout << "Cannot open the video file" << endl;
  return -1;
 }
cout<<"Press Q to Quit" << endl;
String win_name = "First Video";
namedWindow(win_name); 
 while (true)
 {
  Mat frame;
  // read a frame
  bool flag = cap.read(frame); 

  //Breaking the while loop at the end of the video
  if (flag == false) 
  {
   break;
  }
  //display the frame 
  imshow(win_name, frame);
  //Wait for 100 ms and key 'q' for exit
  if (waitKey(100) == 'q')
  {
    break;
  }
 }
destroyWindow(win_name);
return 0;
}