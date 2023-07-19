#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "visualize_image.h"

// Function to calculate the Euclidean distance between two color histograms
double calculateColorHistogramDistance(const cv::Mat& hist1,
    const cv::Mat& hist2) {
    cv::Mat diff;
    cv::absdiff(hist1, hist2, diff);
    diff = diff.mul(diff);
    double distance = cv::sum(diff)[0];
    return std::sqrt(distance);
}

cv::Mat createHistogram(const cv::Mat& image) {    
    int channels[] = { 0 }; //the grayscale image has a single channel, which is indexed as 0
    int histSize[] = { 256 };
    float range[] = { 0,256 }; //greylevel of each pixel (from 0 to 255, the upper value is excluded)
    const float* histRange[] = { range };

    cv::Mat hist;

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, histRange);
    cv::normalize(hist, hist, 0, hist.rows, cv::NORM_MINMAX, -1, cv::Mat());

    return hist;
}

// Function to merge regions based on color similarity
void mergeRegions(cv::Mat& regions, cv::Mat& colorQuantization, int numRegions, double threshold) {
    std::vector<double> labels;

    // Create a distance table to store distances between region histograms
    cv::Mat distanceTable(numRegions, numRegions, CV_64F, cv::Scalar(std::numeric_limits<double>::max()));

    // vector of histograms
    std::vector<cv::Mat> hist;
    // we are not considering pixels not classified
    for (int z = 0; z < numRegions; z++) {
        cv::Mat temp = (regions == z);

        cv::Mat result; // 3d matrix

        cv::multiply(colorQuantization, temp, result);
                
        // create histogram for each region
        hist.push_back(createHistogram(result));
    }

    // Calculate and store distances between color histograms of neighboring
    // regions
    for (int i = 0; i < hist.size(); i++) {
        for (int j = 0; j < hist.size(); j++) {
            if (i != j) {
                double dist = cv::norm(hist[i], hist[j], cv::NORM_L2SQR);
                //double dist = calculateColorHistogramDistance(hist[i], hist[j]);
                distanceTable.at<double>(i, j) = dist;
                distanceTable.at<double>(j, i) = dist;
            }
            else {
                distanceTable.at<double>(j, i) = 0;
            }
        }
    }

    // Merge regions based on their color similarity
    while (numRegions > 1) {
        // Find the pair of regions with the minimum distance
        double minDistance = std::numeric_limits<double>::max();
        int minRegion1 = -1;
        int minRegion2 = -1;

        for (int i = 0; i < distanceTable.rows; i++) {
            for (int j = 0; j < distanceTable.cols; j++) {
                double distance = distanceTable.at<double>(i, j);
                if (i != j && distance < minDistance) {
                    minDistance = distance;
                    minRegion1 = i;
                    minRegion2 = j;
                }
            }
        }

        if (minDistance > threshold) {
            break;
        }

        if (minRegion1 != -1 && minRegion2 != -1) {
            //Merge the two regions and remove minRegion2
            for (int i = 0; i < regions.rows; i++) {
                for (int j = 0; j < regions.cols; j++) {
                    int region = regions.at<double>(i, j);

                    //if the region is the same as minRegion2, simply replace its value with minRegion1
                    if (region == minRegion2) {
                        regions.at<double>(i, j) = minRegion1;
                    }
                }
            }

            // Set the distance to max for minRegion2 in distanceTable
            for (int i = 0; i < distanceTable.rows; i++) {
                
                distanceTable.at<double>(i, minRegion2) = std::numeric_limits<double>::max();
                distanceTable.at<double>(minRegion2, i) = std::numeric_limits<double>::max();
            }
            
            numRegions--;

            /*
            //replace all occourences of minRegion2 with minRegion1 (or with the minimum distance?) in the distanceTable matrix
            for (int i = 0; i < distanceTable.rows; i++) {
                if (i == minRegion2) {
                    for (int j = 0; j < distanceTable.cols; j++) {
                        if(distanceTable.at<double>(i, j) != std::numeric_limits<double>::max())
                            distanceTable.at<double>(i, j) = std::min(distanceTable.at<double>(minRegion1, j), distanceTable.at<double>(i, j));
                    }
                }
                else {
                    for (int j = 0; j < distanceTable.cols; j++) {
                        if (j == minRegion2) {
                            if (distanceTable.at<double>(i, j) != std::numeric_limits<double>::max())
                                distanceTable.at<double>(i, j) = std::min(distanceTable.at<double>(i, minRegion1), distanceTable.at<double>(i, j));
                        }
                    }
                }
            }

            //and ensure that the entries relative to the regions just merged can't be selected again
            distanceTable.at<double>(minRegion1, minRegion2) = std::numeric_limits<double>::max();
            distanceTable.at<double>(minRegion2, minRegion1) = std::numeric_limits<double>::max();*/
        }
        else {
            break;
        }
    }

    visualize_image(regions, "fine");
}