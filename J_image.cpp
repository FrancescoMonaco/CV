#include "J_image.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <omp.h>

/**
* Function that returns an array of J_images, one for the each window specified
* @param quantized_images image where each pixel contains the label of the cluster it was assigned to
* @param windows array that specify each window (number of neighbours to consider) for each J_image
* @param num_classes number of clusters
*/
std::vector<cv::Mat> J_images(cv::Mat quantized_image, std::vector<int> windows, int num_classes) {
    std::vector<cv::Mat> J_images;
    J_images.reserve(windows.size());

    for (int i = 0; i < windows.size(); ++i) {
        const int windowSize = windows[i];
        cv::Mat J_values(quantized_image.size(), CV_64FC1);

        J_image_calc(quantized_image, J_values, windowSize, num_classes);

        J_images.push_back(J_values);
    }

    return J_images;
}


void J_image_calc(cv::Mat quantized_image, cv::Mat J_image, int size, int num_classes) {
    const int kernel_size = size;
    const int movement = (kernel_size - 1) / 2;
    const int rows = quantized_image.rows;
    const int cols = quantized_image.cols;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double J_value = 0.0;
            double S_T = 0.0;
            double S_W = 0.0;

            int totalPoints = 0;
            int sumX = 0;
            int sumY = 0;

            for (int rowOff = -movement; rowOff <= movement; ++rowOff) {
                const int newRow = row + rowOff;
                if (newRow >= 0 && newRow < rows) {
                    for (int colOff = -movement; colOff <= movement; ++colOff) {
                        const int newCol = col + colOff;
                        if (newCol >= 0 && newCol < cols) {
                            ++totalPoints;
                            sumX += newRow;
                            sumY += newCol;
                        }
                    }
                }
            }

            const cv::Point globalMean(sumX / totalPoints, sumY / totalPoints);

            for (int rowOff = -movement; rowOff <= movement; ++rowOff) {
                const int newRow = row + rowOff;
                if (newRow >= 0 && newRow < rows) {
                    for (int colOff = -movement; colOff <= movement; ++colOff) {
                        const int newCol = col + colOff;
                        if (newCol >= 0 && newCol < cols) {
                            const cv::Point diff(newRow - globalMean.x, newCol - globalMean.y);
                            S_T += diff.x * diff.x + diff.y * diff.y;
                        }
                    }
                }
            }
            
            std::vector<cv::Point> classMeans(num_classes);
            std::vector<int> classPoints(num_classes);

            for (int k = 0; k < num_classes; ++k) {
                for (int rowOff = -movement; rowOff <= movement; ++rowOff) {
                    const int newRow = row + rowOff;
                    if (newRow >= 0 && newRow < rows) {
                        for (int colOff = -movement; colOff <= movement; ++colOff) {
                            const int newCol = col + colOff;
                            if (newCol >= 0 && newCol < cols) {
                                const uchar label = quantized_image.at<uchar>(newRow, newCol);
                                if (label == k) {
                                    ++classPoints[k];
                                    classMeans[k].x += newRow;
                                    classMeans[k].y += newCol;
                                }
                            }
                        }
                    }
                }
            }

            for (int k = 0; k < num_classes; ++k) {
                if (classPoints[k] != 0) {
                    classMeans[k].x /= classPoints[k];
                    classMeans[k].y /= classPoints[k];
                }
                else {
                    classMeans[k].x = 0;
                    classMeans[k].y = 0;
                }
            }

            for (int k = 0; k < num_classes; ++k) {
                for (int rowOff = -movement; rowOff <= movement; ++rowOff) {
                    const int newRow = row + rowOff;
                    if (newRow >= 0 && newRow < rows) {
                        for (int colOff = -movement; colOff <= movement; ++colOff) {
                            const int newCol = col + colOff;
                            if (newCol >= 0 && newCol < cols) {
                                const uchar label = quantized_image.at<uchar>(newRow, newCol);
                                if (label == k) {
                                    const cv::Point diff(newRow - classMeans[k].x, newCol - classMeans[k].y);
                                    S_W += diff.x * diff.x + diff.y * diff.y;
                                }
                            }
                        }
                    }
                }
            }

            if (S_W != 0.0)
                J_value = (S_T - S_W) / S_W;

            #pragma omp critical
            J_image.at<double>(row, col) = J_value;
        }
    }
}