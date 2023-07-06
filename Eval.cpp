#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Eval_func.h"

//****Variables, namespaces definitions***
const std::string GOLD_BB = "bounding_boxes";
const std::string PREDS_BB = "box_preds";
const std::string GOLD_SS = "masks";
const std::string PREDS_SS = "mask_preds";
namespace fs = std::filesystem;

//****Main****
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./Eval <relative_path>\n";
        return 1;
    }

    std::string relativePath = argv[1];
    std::vector<std::vector<BoundingBox>> boundingBoxes_gold;
    std::vector<std::vector<BoundingBox>> boundingBoxes_pred;
    // Bounding Boxes evaluation
        // Load ground truth data
        for(const auto& entry : fs::directory_iterator(relativePath+ GOLD_BB)){
        if(entry.is_regular_file() && entry.path().extension() == ".txt"){
            //if the path contains a 3 skip it (no evaluation for this file)
            if(entry.path().string().find("3") != std::string::npos){
                continue;
            }
            std::vector<BoundingBox> data = loadBoundingBoxData(entry.path().string());
            if(!data.empty()){
                boundingBoxes_gold.push_back(data);
            }
        }
    }
        // Load prediction data
        for(const auto& entry : fs::directory_iterator(relativePath+ PREDS_BB)){
        if(entry.is_regular_file() && entry.path().extension() == ".txt"){
            //if the path contains a 3 skip it (no evaluation for this file)
            if(entry.path().string().find("3") != std::string::npos){
                continue;
            }
            std::vector<BoundingBox> data = loadBoundingBoxData(entry.path().string());
            if(!data.empty()){
                boundingBoxes_pred.push_back(data);
            }
        }
    }


    float mAP = processBoxPreds(boundingBoxes_gold, boundingBoxes_pred);

    // Print the result
    std::cout << "mAP: " << mAP << std::endl;

    // Semantic Segmentation evaluation

    std::vector<cv::Mat> images_gold;
    std::vector<cv::Mat> images_pred;
        // Load ground truth data
        for(const auto& entry : fs::directory_iterator(relativePath+ GOLD_SS)){
        if(entry.is_regular_file() && entry.path().extension() == ".png"){
            //if the path contains a 3 skip it (no evaluation for this file)
            if(entry.path().string().find("3") != std::string::npos){
                continue;
            }
            cv::Mat data = loadSemanticSegmentationData(entry.path().string());
            if(!data.empty()){
                images_gold.push_back(data);
            }
        }
    }
        // Load prediction data
        for(const auto& entry : fs::directory_iterator(relativePath+ PREDS_SS)){
        if(entry.is_regular_file() && entry.path().extension() == ".png"){
            //if the path contains a 3 skip it (no evaluation for this file)
            if(entry.path().string().find("3") != std::string::npos){
                continue;
            }
            cv::Mat data = loadSemanticSegmentationData(entry.path().string());
            if(!data.empty()){
                images_pred.push_back(data);
            }
        }
    }

    // Compute IoU
    float IoU = processSemanticSegmentation(images_gold, images_pred);

    // Print results
    std::cout << "IoU: " << IoU << std::endl;

    return 0;
}
