#ifndef COLOR_QUANTIZATION_H
#define COLOR_QUANTIZATION_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <opencv2/imgproc.hpp>

double euclideanDistance(const cv::Vec3b& vec1, const cv::Vec3b& vec2);
int quantization(cv::Mat peer_image, std::vector<double> peergroup_vec, cv::Mat clusterAssignments);

//LLoyd's clustering
void assignLabels(cv::Mat dataset, std::vector<cv::Vec3b>& centroids, cv::Mat assignments);
void calculateNewCentroids(cv::Mat dataset, cv::Mat weights, cv::Mat assignments, std::vector<cv::Vec3b>& centroids);
void generalizedLloydAlgorithm(cv::Mat dataset, int k, cv::Mat weights, std::vector<cv::Vec3b>& centroids, cv::Mat clusterAssignments);


//Agglomerative clustering
void mergeClusters(cv::Mat peergroupImage, std::vector<cv::Vec3b>& centroids, cv::Mat clusterAssignments, double threshold);

#endif