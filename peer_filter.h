#ifndef PEER_FILTER_H
#define PEER_FILTER_H
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<double> peer_group_filter(cv::Mat img, cv::Mat weighted_image, int size);

std::vector<std::pair<cv::Point, double>> neighbourhood(cv::Mat img_in, int row, int col, int kernel_size);

double euclideanDistance(const cv::Vec3b& vec1, const cv::Vec3b& vec2);

int Fisher_discriminant(std::vector<std::pair<cv::Point, double>> vec, double& Tn);

void weight_peergroup(int optIndex, std::vector<std::pair<cv::Point, double>> neighbours, cv::Mat weigthedImg, int row, int cols);

#endif

