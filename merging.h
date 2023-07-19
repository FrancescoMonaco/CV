#ifndef MERGING_H
#define MERGING_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>


double calculateColorHistogramDistance(const cv::Mat& hist1, const cv::Mat& hist2);

cv::Mat createHistogram(const cv::Mat& image);

void mergeRegions(cv::Mat & regions, cv::Mat & colorQuantization, int numRegions, double threshold);
#endif
