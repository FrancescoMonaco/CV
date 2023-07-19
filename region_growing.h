#ifndef REG_GROWING_H
#define REG_GROWING_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat region_growing(std::vector<cv::Mat> J_images, std::vector<int> windowSizes, int& numRegions);
std::vector<double> computeThresholdForEachRegion(cv::Mat J_image, cv::Mat& regions, int windowSizes, int& label, int& numberOfRegions, int minimum_size, std::vector<double>& reg);
void seed_area_growing(cv::Mat Jimage, cv::Mat& regions);
void complete_expansion(cv::Mat& regions, const cv::Mat Jimage);
void connect_starting_seeds(cv::Mat& currentRegions, std::vector<cv::Point> starting_points, cv::Mat prev_regions, double prev);
#endif