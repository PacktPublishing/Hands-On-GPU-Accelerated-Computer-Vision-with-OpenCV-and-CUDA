#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main( int argc, char** argv )
{
	Mat h_object_image = imread( "images/object1.jpg", 0 ); 
	Mat h_scene_image = imread( "images/scene1.jpg", 0 );
	
	cuda::GpuMat d_object_image;
	cuda::GpuMat d_scene_image;
	
	cuda::GpuMat d_keypoints_scene, d_keypoints_object; 
	vector< KeyPoint > h_keypoints_scene, h_keypoints_object;
	cuda::GpuMat d_descriptors_scene, d_descriptors_object;
	
	d_object_image.upload(h_object_image);
	d_scene_image.upload(h_scene_image);

	
	cuda::SURF_CUDA surf(100);
	surf( d_object_image, cuda::GpuMat(), d_keypoints_object, d_descriptors_object );
	surf( d_scene_image, cuda::GpuMat(), d_keypoints_scene, d_descriptors_scene );

	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();
	vector< vector< DMatch> > d_matches;
	matcher->knnMatch(d_descriptors_object, d_descriptors_scene, d_matches, 2);

	
	surf.downloadKeypoints(d_keypoints_scene, h_keypoints_scene);
	surf.downloadKeypoints(d_keypoints_object, h_keypoints_object);
	
	
	std::vector< DMatch > good_matches;
	for (int k = 0; k < std::min(h_keypoints_object.size()-1, d_matches.size()); k++)
	{
		if ( (d_matches[k][0].distance < 0.6*(d_matches[k][1].distance)) &&
				((int)d_matches[k].size() <= 2 && (int)d_matches[k].size()>0) )
		{
			good_matches.push_back(d_matches[k][0]);
		}
	}
std::cout << "size:" <<good_matches.size();
	Mat h_image_result;
	drawMatches( h_object_image, h_keypoints_object, h_scene_image, h_keypoints_scene,
			good_matches, h_image_result, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::DEFAULT );

	
	std::vector<Point2f> object;
	std::vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++) {
		object.push_back(h_keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(h_keypoints_scene[good_matches[i].trainIdx].pt);
	}
		Mat Homo = findHomography(object, scene, RANSAC);
		std::vector<Point2f> corners(4);
		std::vector<Point2f> scene_corners(4);
		corners[0] = Point(0, 0);
		corners[1] = Point(h_object_image.cols, 0);
		corners[2] = Point(h_object_image.cols, h_object_image.rows);
		corners[3] = Point(0, h_object_image.rows);
		perspectiveTransform(corners, scene_corners, Homo);
		line(h_image_result, scene_corners[0] + Point2f(h_object_image.cols, 0),scene_corners[1] + Point2f(h_object_image.cols, 0),	Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[1] + Point2f(h_object_image.cols, 0),scene_corners[2] + Point2f(h_object_image.cols, 0),Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[2] + Point2f(h_object_image.cols, 0),scene_corners[3] + Point2f(h_object_image.cols, 0),Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[3] + Point2f(h_object_image.cols, 0),scene_corners[0] + Point2f(h_object_image.cols, 0),Scalar(255, 0, 0), 4);
	imshow("Good Matches & Object detection", h_image_result);
	waitKey(0);

	
	return 0;
}

