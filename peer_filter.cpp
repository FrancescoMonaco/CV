#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "peer_filter.h"

/**
* Function that implements the peer group filtering.
* @param img, original image
* @param weighted_image, image that will be updated with the new values found after the application of the function
* @param size, indicates the size of the neighbourhood to consider for each pixel
* @return a vector containing the threshold (max distance found) for each pixel of the image
*/
std::vector<double> peer_group_filter(cv::Mat img, cv::Mat weighted_image, int size) {
    //this vector will contain the threshold for each pixel of the image
    std::vector<double> Tn;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            //define the neighbours of the central pixel, storing also their distance from it
            std::vector<std::pair<cv::Point, double>> neighbours;

            neighbours = neighbourhood(img, i, j, size);


            double currMaxDist = 0;

            //find the optimal index for the threshold
            int optimalIndex = Fisher_discriminant(neighbours, currMaxDist);

            Tn.push_back(currMaxDist);


            weight_peergroup(optimalIndex, neighbours, weighted_image, i, j);
        }
    }

    return Tn;
}

/**
* Function that computes the neighbours of the central pixel, ordered based off the distance from it
* @param img_in input image
* @param row, row index of the central pixel
* @param col, column index of the central pixel
* @param kernel_size size of the neighbourhood of the central pixel
* @return a vector containing the neighbors of the central pixel and their distance from it, in non decreasing order
*/
std::vector<std::pair<cv::Point, double>> neighbourhood(cv::Mat img_in, int row, int col, int kernel_size) {
    int movement = (kernel_size - 1) / 2;
    std::vector<std::pair<cv::Point, double>> neighbour;

    for (int rowOff = -movement; rowOff <= movement; rowOff++) {
        const int newRow = row + rowOff;
        if (newRow >= 0 && newRow < img_in.rows) {
            for (int colOff = -movement; colOff <= movement; colOff++) {
                const int newCol = col + colOff;
                if (newCol >= 0 && newCol < img_in.cols) {
                    double dist = euclideanDistance(img_in.at<cv::Vec3b>(newRow, newCol), img_in.at<cv::Vec3b>(row, col));

                    neighbour.push_back(std::make_pair(cv::Point(newRow, newCol), dist));
                }
            }
        }
    }
    
    //sort the vector in ascending order, so that the minimum distance neighbor is at the first position
    std::sort(neighbour.begin(), neighbour.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    return neighbour;
}

/**
* Function used to compute the euclidean distance (intensity-wise) of two points
* @param p1 first point
* @param p2 second point
* @return the euclidean distance between p1 and p2
*/
double euclideanDistance(const cv::Vec3b& p1, const cv::Vec3b& p2) {
    double sum = 0.0;
    for (int i = 0; i < p1.channels; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

/**
* Function that implements the Fisher discriminant criterion. The goal is to find the index for which J is maximized
* @param vec vector containing the neighbours of the central pixel with the respective distances
* @param Tn value of the maximum distance
* @return cut off position where the criterion J is maximized
*/
int Fisher_discriminant(std::vector<std::pair<cv::Point, double>> vec, double& Tn) {
    //store all the J_i values and then take the index of the maximum one
    std::vector<double> J;
    for (int i = 0; i < vec.size(); i++) {

        double a1 = 0;
        double a2 = 0;

        // don't consider the first one since it is the one with distance zero, so the center point
        for (int j = 1; j < vec.size(); j++) {

            double diff = vec[j].second;

            if (j == i - 1) {
                a1 = a2;
            }
            a2 += diff;
        }

        a1 /= (i + 1);
        a2 /= (vec.size() + 1 - i);

        double s1 = 0;
        double s2 = 0;

        for (int j = 0; j < vec.size(); j++) {

            double diff = vec[j].second;
            if (j <= j - 1) {
                s1 += abs(diff - a1);
            }
            s2 += abs(diff - a2);
        }
        double J_i = 0;
        double numj = abs(a1 - a2);
        numj = pow(numj, 2);

        if (s1 + s2 != 0)
            J_i = numj / (s1 + s2);

        J.push_back(J_i);
    }

    int optIndex = -1;

    // find the maximum value
    if (!J.empty()) {
        optIndex = std::distance(J.begin(), std::max_element(J.begin(), J.end()));
    }

    Tn = J[optIndex];

    return optIndex;
}

/**
* Function that replaces the value of the central pixel with the weighted average of its neighbors
* @param optIndex cut-off optimal index
* @param neighbors vector of the neighbours of the central pixel with corresponding distance from it
* @param weightedImg image in which the new value for the central pixel is stored
* @param row, row index of the central pixel
* @param col, column index of the central pixel
*/
void weight_peergroup(int optIndex, std::vector<std::pair<cv::Point, double>> neighbours, cv::Mat weigthedImg, int row, int cols) {

    // Perform peer group classification and replacement
    cv::Vec3b weightedAverage(0, 0, 0);
    double weightSum = 0.0;
    double sigma = 0.5;

    for (int a = 0; a < optIndex; a++) {
        //the weights considered are Gaussian weights
        double weight = exp(-0.5 * (neighbours[a].second * neighbours[a].second) / (sigma * sigma));

        weightedAverage += weight * weigthedImg.at<cv::Vec3b>((neighbours[a].first).x, (neighbours[a].first).y);
        weightSum += weight;
    }

    if (weightSum > 0)
        weightedAverage /= weightSum;
    else
        weightedAverage = 0;

    // Replace the central pixel with the weighted average
    weigthedImg.at<cv::Vec3b>(row, cols) = weightedAverage;
}