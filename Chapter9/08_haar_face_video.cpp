#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    
	VideoCapture cap("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");    
	if (!cap.isOpened()) {
        cout << "Can not open video source";
        return -1;
    }
	std::vector<cv::Rect> h_found;
    cv::Ptr<cv::cuda::CascadeClassifier> cascade = cv::cuda::CascadeClassifier::create("haarcascade_frontalface_alt2.xml");
    cv::cuda::GpuMat d_frame, d_gray, d_found;
    while(1)
    {
        Mat frame;
        if ( !cap.read(frame) ) {
            cout << "Can not read frame from webcam";
            return -1;
        }
		int64 start = cv::getTickCount();
        d_frame.upload(frame);
        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);

        cascade->detectMultiScale(d_gray, d_found);
        cascade->convert(d_found, h_found);
        
		for(int i = 0; i < h_found.size(); ++i)
		{
              rectangle(frame, h_found[i], Scalar(0,255,255), 5);
		}
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
        imshow("Result", frame);
        if (waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
