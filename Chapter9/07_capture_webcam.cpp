#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int, char**)
{
    Mat frame;
    VideoCapture cap("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"); 
    if (!cap.isOpened()) {
        cout << "Unable to open camera\n";
        return -1;
    }
    while (1)
    {

		int64 start = cv::getTickCount();
        cap.read(frame);

        if (frame.empty()) {
            cout << "Cannot read frame\n";
            break;
        }
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;

        imshow("Live", frame);
		        if (waitKey(30) == 'q')
            break;
    }
    return 0;
}
