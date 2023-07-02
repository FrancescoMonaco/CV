#ifndef PROCESSING
#define PROCESSING
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include <iostream>


void apply_ORB(cv::Mat& templ, cv::Mat& image, bool drawBox=true);

#endif
