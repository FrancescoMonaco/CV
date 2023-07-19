#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <time.h>
#include <opencv2/highgui.hpp>

#include "color_quantization.h"


int quantization(cv::Mat peergroup_image, std::vector<double> peergroup_vec, cv::Mat clusterAssignments) {

    cv::Mat changeColor;

    // Define the type of greylevel such as a double
    cv::Mat weights(peergroup_image.rows, peergroup_image.cols, CV_64FC1);

    for (int i = 0; i < peergroup_image.rows; i++) {
        for (int j = 0; j < peergroup_image.cols; j++) {
            //the weight of each pixel is computed as e^(-T)
            weights.at<double>(i, j) =
                exp(-peergroup_vec[i * peergroup_image.rows + j]);
        }
    }

    //compute the average of T(n), which represents the smoothness of the image
    double sum = std::accumulate(peergroup_vec.begin(), peergroup_vec.end(), 0.0);
    double average = sum / peergroup_vec.size();

    // number of clusters
    int N = 2 * average;

    //vector containing the coordinates of the centroids
    std::vector<cv::Vec3b> centroids;

    generalizedLloydAlgorithm(peergroup_image, N, weights, centroids, clusterAssignments);

    //This snippet of code is used to show the clusterized image (it associates to each label a distinct BGR color)
    cv::Mat clusterizedImage = peergroup_image.clone();

    std::vector<cv::Vec3b> clusterColors;

    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < N; i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        clusterColors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    for (int i = 0; i < peergroup_image.rows; i++) {
        for (int j = 0; j < peergroup_image.cols; j++) {
            int index = static_cast<int>(clusterAssignments.at<uchar>(i, j));

            clusterizedImage.at<cv::Vec3b>(i, j) = clusterColors[index];
        }
    }

    cv::namedWindow("Clustering?");
    cv::imshow("Clustering?", clusterizedImage);
    cv::waitKey(0);
    

    //to obtain the true pixel clusters, recompute the centroids without the pixel weights
    calculateNewCentroids(peergroup_image, cv::Mat(peergroup_image.rows, peergroup_image.cols, CV_64FC1, 1.0), clusterAssignments, centroids);

    //merge the clusters which have centroids whose distance is smaller than a set threshold
    mergeClusters(peergroup_image, centroids, clusterAssignments, 75);

    //finally, assign to each pixel the true cluster it belongs to
    assignLabels(peergroup_image, centroids, clusterAssignments);

    cv::Mat clusterizedAggImage = peergroup_image.clone();

    for (int i = 0; i < peergroup_image.rows; i++) {
        for (int j = 0; j < peergroup_image.cols; j++) {
            int index = static_cast<int>(clusterAssignments.at<uchar>(i, j));

            clusterizedAggImage.at<cv::Vec3b>(i, j) = clusterColors[index];
        }
    }

    //std::cout << "New num of clusters" << centroids.size() << std::endl;

    cv::namedWindow("AggClustering?");
    cv::imshow("AggClustering?", clusterizedAggImage);
    cv::waitKey(0);

    return centroids.size();
}

/**
* Function that implements the LLoyd's algorithm
* @param dataset image containing the values of each pixel
* @param k number of clusters
* @param weights image containing the weights assigned to each pixel
* @param centroids vector that will contain the final centroids computed by the clustering
* @param labels image that contains as value of each pixel the cluster they are assigned to
*/
void generalizedLloydAlgorithm(cv::Mat dataset, int k, cv::Mat weights, std::vector<cv::Vec3b>& centroids, cv::Mat labels) {

    srand(time(NULL));
    //Choose k random pixels in the image as initial centroids
    for (int i = 0; i < k; i++) {
        int row = ((double)rand() / (RAND_MAX)) + rand() % dataset.rows;
        int col = ((double)rand() / (RAND_MAX)) + rand() % dataset.cols;

        centroids.push_back(dataset.at<cv::Vec3b>(row, col));
    }

    bool converged = false;
    int maxIter = 500;
    int currIter = 0;

    double previousDistortion = std::numeric_limits<double>::max();

    //the algorithm ends either if convergence is reached or a maximum number of iterations have been computed
    while (currIter < maxIter) {
        double currentDistortion = 0;
        cv::Mat previousAssignment = labels.clone();
        std::vector<cv::Vec3b> oldCentroids = centroids;

        //assign labels to each pixel
        assignLabels(dataset, centroids, labels);

        //compute the new centroids
        calculateNewCentroids(dataset, weights, labels, centroids);

        for (int i = 0; i < labels.rows; i++) {
            for (int j = 0; j < labels.cols; j++) {
                currentDistortion += weights.at<double>(i, j) * pow(euclideanDistance(dataset.at<cv::Vec3b>(i, j),
                    centroids[labels.at<uchar>(i, j)]), 2);
            }
        }

        std::cout << "Previous: " << previousDistortion << " current: " << currentDistortion << std::endl;

        //check if the current iteration didn't change the centroids
        if (currentDistortion >= previousDistortion) {
            labels = previousAssignment;
            break;
        }

        previousDistortion = currentDistortion;
        currIter++;
    }
}

//assign to each pixel the closest centroid
void assignLabels(cv::Mat dataset, std::vector<cv::Vec3b>& centroids, cv::Mat labels) {

    //iterate through all pixels of the image
    for (int i = 0; i < dataset.rows; i++) {
        for (int j = 0; j < dataset.cols; j++) {
            double minDistance = std::numeric_limits<double>::max();
            int closestCentroidIndex = -1;

            //compute the minimum distance and assign the centroid
            for (int k = 0; k < centroids.size(); k++) {
                double dist = euclideanDistance(dataset.at<cv::Vec3b>(i, j), centroids[k]);

                if (dist < minDistance) {
                    minDistance = dist;
                    closestCentroidIndex = k;
                }
            }

            //assign to the pixel its closest centroid (number between 0 and k-1)
            labels.at<uchar>(i, j) = static_cast<uchar>(closestCentroidIndex);
        }
    }
}

//calculate the new centroids
void calculateNewCentroids(cv::Mat dataset, cv::Mat weights, cv::Mat labels, std::vector<cv::Vec3b>& centroids) {
    //for each centroid, define the sum of weights and intensity for each channel
    std::vector<double> clusterWeights(centroids.size(), 0.0);
    std::vector<double> sumB(centroids.size(), 0.0);
    std::vector<double> sumG(centroids.size(), 0.0);
    std::vector<double> sumR(centroids.size(), 0.0);

    for (int i = 0; i < dataset.rows; i++) {
        for (int j = 0; j < dataset.cols; j++) {
            double weight = weights.at<double>(i, j);

            for (int k = 0; k < centroids.size(); k++) {
                if (labels.at<uchar>(i, j) == k) {
                    clusterWeights[k] += weight;
                    sumB[k] += weight * dataset.at<cv::Vec3b>(i, j)[0];
                    sumG[k] += weight * dataset.at<cv::Vec3b>(i, j)[1];
                    sumR[k] += weight * dataset.at<cv::Vec3b>(i, j)[2];
                }
            }
        }
    }

    for (int k = 0; k < centroids.size(); k++) {
        if (clusterWeights[k] > 0) {
            centroids[k][0] = sumB[k] / clusterWeights[k];
            centroids[k][1] = sumG[k] / clusterWeights[k];
            centroids[k][2] = sumR[k] / clusterWeights[k];
        }
    }
}


void mergeClusters(cv::Mat peerImage, std::vector<cv::Vec3b>& centroids, cv::Mat clusterAssignments, double threshold) {    
    int numLabels = centroids.size();

    // Create a distance table to store distances between region histograms
    cv::Mat distanceTable(numLabels, numLabels, CV_64F, cv::Scalar(std::numeric_limits<double>::max()));
    
    for (int i = 0; i < numLabels; i++) {
        for (int j = 0; j < numLabels; j++) {
            if (i != j) {
                double distance = euclideanDistance(centroids[i], centroids[j]);

                distanceTable.at<double>(i, j) = distance;
                distanceTable.at<double>(j, i) = distance;
            }
            else {
                distanceTable.at<double>(j, i) = 0;
            }
        }
    }

    std::vector<bool> indicesToRemove(centroids.size(), false);
    int regionsLeft = centroids.size();

    while (regionsLeft > 2) {
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

        std::cout << minDistance << std::endl;
        if (minDistance > threshold) {
            break;
        }

        if (minRegion1 != -1 && minRegion2 != -1) {
            //Merge the two regions and remove minRegion2
            bool found = false;
            int index = -1;
            for (int i = 0; i < clusterAssignments.rows; i++) {
                for (int j = 0; j < clusterAssignments.cols; j++) {
                    int region = static_cast<int>(clusterAssignments.at<uchar>(i, j));

                    //if the region is the same as minRegion2, simply replace its value with minRegion1
                    if (region == minRegion2) {
                        clusterAssignments.at<uchar>(i, j) = minRegion1;
                        found = true;
                        index = region;
                    }
                }
            }

            if(found) {
                //centroids.erase(centroids.begin() + index);
                indicesToRemove[index] = true;
                regionsLeft--;
                // Set the distance to max for minRegion2 in distanceTable
                for (int i = 0; i < distanceTable.rows; i++) {
                    
                    distanceTable.at<double>(i, minRegion2) = std::numeric_limits<double>::max();
                    distanceTable.at<double>(minRegion2, i) = std::numeric_limits<double>::max();
                }
            }
        }
        else {
            break;
        }

    }
    std::vector<cv::Vec3b> newCentroids;

    for (int i = 0; i < centroids.size(); i++) {
        if (!indicesToRemove[i])
            newCentroids.push_back(centroids[i]);
    }

    centroids = newCentroids;
}