#ifndef PROCESSING
#define PROCESSING
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>
#include "opencv2/calib3d.hpp"


/**
* The function finds the bounding boxes of the bread and then creates the segmentation
* @param images vector of images
* @param bounding_boxes vectors that contains the bounding boxes
* @param segments vector that contains the segmentation till now
*/
void breadFinder(std::vector<cv::Mat> images, std::vector<cv::Mat> bounding_boxes, std::vector<cv::Mat> segments);

cv::Mat breadBox(cv::Mat image);

#endif
