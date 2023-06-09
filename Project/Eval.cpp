#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Eval_func.h"

//****Variables, namespaces definitions***
const std::string GOLD_BB = "/bounding_boxes";
const std::string PREDS_BB = "/box_preds";
const std::string GOLD_SS = "/masks";
const std::string PREDS_SS = "/mask_preds";
namespace fs = std::filesystem;

//****Main****
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./Eval <relative_path>\n";
        return 1;
    }

    int fileNum = 0;
    std::string relativePath = argv[1];
    std::vector<std::vector<BoundingBox>> boundingBoxes_gold;
    std::vector<std::vector<BoundingBox>> boundingBoxes_pred;
    // Bounding Boxes evaluation
    std::cout << "***Bounding Boxes evaluation started***" << std::endl;
        // Load ground truth data
        for(const auto& entry : fs::directory_iterator(relativePath)){
            //if entry is a folder
            if(entry.is_directory()){
                for(const auto& sub : fs::directory_iterator(entry.path().string() + GOLD_BB)){
                    if(sub.is_regular_file() && sub.path().extension() == ".txt"){
                        //if the path contains a 3 skip it (no evaluation for this file)
                        if(sub.path().string().find("3") != std::string::npos){
                            continue;
                        }
                        std::vector<BoundingBox> data = loadBoundingBoxData(sub.path().string(), fileNum);
                        if(!data.empty()){
                            boundingBoxes_gold.push_back(data);
                            fileNum++;
                        }
                    }
                }
            }
    }
        // Load prediction data
        fileNum = 0;
        for(const auto& entry : fs::directory_iterator(relativePath)){
            //if entry is a folder
            if(entry.is_directory()){
                for(const auto& sub : fs::directory_iterator(entry.path().string() + GOLD_BB)){
                    if(sub.is_regular_file() && sub.path().extension() == ".txt"){
                        //if the path contains a 3 skip it (no evaluation for this file)
                        if(sub.path().string().find("3") != std::string::npos){
                            continue;
                        }
                        std::vector<BoundingBox> data = loadBoundingBoxData(sub.path().string(), fileNum);
                        if(!data.empty()){
                            boundingBoxes_pred.push_back(data);
                            fileNum++;
                        }
                    }
                }
            }
    }


    float mAP = processBoxPreds(boundingBoxes_gold, boundingBoxes_pred);

        // Print the result
    std::cout << "mAP: " << mAP << std::endl;
    
    // Semantic Segmentation evaluation
    std::cout << std::endl << "***Semantic Segmentation evaluation started***" << std::endl;
    std::vector<cv::Mat> images_gold;
    std::vector<cv::Mat> images_pred;
        // Load ground truth data
        for(const auto& entry : fs::directory_iterator(relativePath)){
            if(entry.is_directory()){
                for(const auto& sub : fs::directory_iterator(entry.path().string() + GOLD_SS)){
                    if(sub.is_regular_file() && sub.path().extension() == ".png"){
                        //if the path contains a 3 skip it (no evaluation for this file)
                        if(sub.path().string().find("3") != std::string::npos){
                            continue;
                        }
                        cv::Mat data = loadSemanticSegmentationData(sub.path().string());
                        if(!data.empty()){
                            images_gold.push_back(data);
                            std::cout << "*";
                        }
                    }
                }
            }
    }
        // Load prediction data
        for(const auto& entry : fs::directory_iterator(relativePath)){
            if(entry.is_directory()){
                for(const auto& sub : fs::directory_iterator(entry.path().string() + GOLD_SS)){
                    if(sub.is_regular_file() && sub.path().extension() == ".png"){
                        //if the path contains a 3 skip it (no evaluation for this file)
                        if(sub.path().string().find("3") != std::string::npos){
                            continue;
                        }
                        cv::Mat data = loadSemanticSegmentationData(sub.path().string());
                        if(!data.empty()){
                            images_pred.push_back(data);
                            std::cout << "*";
                        }
                    }
                }
            }
    }
        // Compute IoU
    float IoU = processSemanticSegmentation(images_gold, images_pred);

        // Print results
    std::cout << std::endl << "IoU: " << IoU << std::endl;

    // Food Leftover estimation
    std::cout << std::endl << "***Food Leftover estimation started***" << std::endl;
    for(const auto& entry: fs::directory_iterator(relativePath)){
        if(entry.is_directory()){
            foodLeftoverEstimation(entry.path().string() + GOLD_SS,\
             entry.path().string() + GOLD_SS);
        }
        std::cout << std::endl;
    }
    std::cout << "***Food Leftover estimation completed***" << std::endl;
    return 0;
}
