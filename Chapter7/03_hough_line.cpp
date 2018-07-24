#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;


int main()
{
    Mat h_image = imread("images/drawing.JPG",0);
    if (h_image.empty())
    {
        cout << "can not open image"<< endl;
        return -1;
    }

    Mat h_edge;
    cv::Canny(h_image, h_edge, 100, 200, 3);

    Mat h_imagec;
    cv::cvtColor(h_edge, h_imagec, COLOR_GRAY2BGR);
    Mat h_imageg = h_imagec.clone();
    vector<Vec4i> h_lines;
    {
        const int64 start = getTickCount();
        HoughLinesP(h_edge, h_lines, 1, CV_PI / 180, 50, 60, 5);
        const double time_elapsed = (getTickCount() - start) / getTickFrequency();
        cout << "CPU Time : " << time_elapsed * 1000 << " ms" << endl;
        cout << "CPU FPS : " << (1/time_elapsed) << endl;
    }

    for (size_t i = 0; i < h_lines.size(); ++i)
    {
        Vec4i line_point = h_lines[i];
        line(h_imagec, Point(line_point[0], line_point[1]), Point(line_point[2], line_point[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }

    GpuMat d_edge, d_lines;
	d_edge.upload(h_edge);
    {
		const int64 start = getTickCount();
        Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 50, 5);
        hough->detect(d_edge, d_lines);

        const double time_elapsed = (getTickCount() - start) / getTickFrequency();
        cout << "GPU Time : " << time_elapsed * 1000 << " ms" << endl;
       cout << "GPU FPS : " << (1/time_elapsed) << endl;
    }
    vector<Vec4i> lines_g;
    if (!d_lines.empty())
    {
        lines_g.resize(d_lines.cols);
        Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_g[0]);
        d_lines.download(h_lines);
    }
    for (size_t i = 0; i < lines_g.size(); ++i)
    {
        Vec4i line_point = lines_g[i];
        line(h_imageg, Point(line_point[0], line_point[1]), Point(line_point[2], line_point[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }

    imshow("source", h_image);
    imshow("detected lines [CPU]", h_imagec);
    imshow("detected lines [GPU]", h_imageg);
    imwrite("hough_source.png", h_image);
    imwrite("hough_cpu_line.png", h_imagec);
    imwrite("hough_gpu_line.png", h_imageg);
    
    waitKey(0);

    return 0;
}
