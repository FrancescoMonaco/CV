#ifndef PROCESSING
#define PROCESSING
#include "panoramic_utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"


/**
* Function to perform SIFT
* @param in1 first image
* @param in2 second image
*/
std::vector<cv::KeyPoint> compute_SIFT(cv::Mat& in1);

/**
* Function to cyclindrical projection
* @param in1 first image
* @param in2 second image
*/
void stitch(std::vector<cv::Mat> in);

/**
* Functions that matches the keypoints and finds the homography matrix
* @param img1, img2 the images to match
* @param vec1, vec2 the vectors of keypoints
*/
void matcher(std::vector<cv::Mat> images, std::vector<std::vector<cv::KeyPoint>> vec);

#endif
