#ifndef Eval
#define Eval
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

struct BoundingBox {
    int file;
    int id;
    int x1;
    int y1;
    int width;
    int height;
};

/// @brief Returns a vector of bounding boxes from the given file
/// @param filePath, path to the file containing the bounding box data
/// @return vector of bounding boxes
std::vector<BoundingBox> loadBoundingBoxData(const std::string& filePath);

/// @brief 
/// @param filePath 
/// @return 
cv::Mat loadSemanticSegmentationData(const std::string& filePath);

/// @brief Computes the mAP over the predictions and the golden data
/// @param resultData, vector of bounding boxes from the result file
/// @param predData , vector of bounding boxes from the prediction file
/// @return mAP
float processBoxPreds(const std::vector<std::vector<BoundingBox>>& resultData, const std::vector<std::vector<BoundingBox>>& predData);

/// @brief 
/// @param relativePath 
/// @param folder 
/// @return 
std::vector<cv::Mat> processSemanticSegmentation(const std::string& relativePath, const std::string& folder);


#endif