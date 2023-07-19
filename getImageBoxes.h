#ifndef IMAGEBOXES_H
#define IMAGEBOXES_H

#include <vector>
#include <opencv2/highgui.hpp>
/// @brief Extract the plates using the Hough Transform
/// @param image, image to be processed
/// @return a vector of the extracted images
std::vector<cv::Rect> getImageBoxes(cv::Mat& image);
std::vector<cv::Rect> getRectanglesBoxes(cv::Mat& image);

#endif
