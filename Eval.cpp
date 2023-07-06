#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Eval_func.h"

const std::string GOLD_BB = "bounding_boxes";
const std::string PREDS_BB = "box_preds";
const std::string GOLD_SS = "masks";
const std::string PREDS_SS = "mask_preds";
namespace fs = std::filesystem;

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
            std::vector<BoundingBox> data = loadBoundingBoxData(entry.path().string());
            if(!data.empty()){
                boundingBoxes_gold.push_back(data);
            }
        }
    }
        // Load prediction data
        for(const auto& entry : fs::directory_iterator(relativePath+ PREDS_BB)){
        if(entry.is_regular_file() && entry.path().extension() == ".txt"){
            std::vector<BoundingBox> data = loadBoundingBoxData(entry.path().string());
            if(!data.empty()){
                boundingBoxes_pred.push_back(data);
            }
        }


    float mAP = processBoxPreds(boundingBoxes_gold, boundingBoxes_pred);

    // Print the result
    std::cout << "mAP: " << mAP << std::endl;

    // Semantic Segmentation evaluation
    std::vector<cv::Mat> resultImages = processSemanticSegmentation(relativePath, GOLD_SS);
    std::vector<cv::Mat> predImages = processSemanticSegmentation(relativePath, PREDS_SS);

    // Compute IoU

    // Print results

    return 0;
}
