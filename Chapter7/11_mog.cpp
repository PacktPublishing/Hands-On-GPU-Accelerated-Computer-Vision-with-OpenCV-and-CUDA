#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;
int main()
{
    VideoCapture cap("abc.avi");
    if (!cap.isOpened())
    {
        cerr << "can not open camera or video file" << endl;
        return -1;
    }
    Mat frame;
    cap.read(frame);
    GpuMat d_frame;
	d_frame.upload(frame);
    Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG();
    GpuMat d_fgmask,d_fgimage,d_bgimage;
    Mat h_fgmask,h_fgimage,h_bgimage;
    mog->apply(d_frame, d_fgmask, 0.01);
    while(1)
    {
        cap.read(frame);
        if (frame.empty())
            break;
        d_frame.upload(frame);
        int64 start = cv::getTickCount();
        mog->apply(d_frame, d_fgmask, 0.01);
        mog->getBackgroundImage(d_bgimage);
        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
        d_fgimage.create(d_frame.size(), d_frame.type());
        d_fgimage.setTo(Scalar::all(0));
        d_frame.copyTo(d_fgimage, d_fgmask);
        d_fgmask.download(h_fgmask);
        d_fgimage.download(h_fgimage);
        d_bgimage.download(h_bgimage);
        imshow("image", frame);
        imshow("foreground mask", h_fgmask);
        imshow("foreground image", h_fgimage);
        imshow("mean background image", h_bgimage);
        if (waitKey(1) == 'q')
            break;
    }

    return 0;
}
