#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudalegacy.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(
)
{
    VideoCapture cap("abc.avi");
    if (!cap.isOpened())
    {
        cerr << "can not open video file" << endl;
        return -1;
    }
    Mat frame;
    cap.read(frame);
    GpuMat d_frame;
	d_frame.upload(frame);
    Ptr<BackgroundSubtractor> gmg = cuda::createBackgroundSubtractorGMG(40);
   GpuMat d_fgmask,d_fgimage,d_bgimage;
    Mat h_fgmask,h_fgimage,h_bgimage;
    gmg->apply(d_frame, d_fgmask);
    while(1)
    {
        cap.read(frame);
        if (frame.empty())
            break;
        d_frame.upload(frame);
        int64 start = cv::getTickCount();
        gmg->apply(d_frame, d_fgmask, 0.01);
        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
        d_fgimage.create(d_frame.size(), d_frame.type());
        d_fgimage.setTo(Scalar::all(0));
        d_frame.copyTo(d_fgimage, d_fgmask);
        d_fgmask.download(h_fgmask);
        d_fgimage.download(h_fgimage);
        imshow("image", frame);
        imshow("foreground mask", h_fgmask);
        imshow("foreground image", h_fgimage);
        if (waitKey(30) == 'q')
            break;
    }
    return 0;
}
