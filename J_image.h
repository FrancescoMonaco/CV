#ifndef J_IMAGE_H
#define J_IMAGE_H

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Mat> J_images(cv::Mat quantized_images, std::vector<int> windows, int num_classes);

void J_image_calc(cv::Mat quantized_image, cv::Mat J_image, int size, int numClasses);

double J_valueForSinglePixel(cv::Mat quantized_image, int row, int col, int neighborSize, int num_classes);

#endif
