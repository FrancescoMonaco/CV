#include "Eval_func.h"
#include <math.h>
const float IOU_THRESH = 0.5;

namespace fs = std::filesystem;



std::vector<BoundingBox> loadBoundingBoxData(const std::string& filePath) {
    std::ifstream file(filePath);
    std::vector<BoundingBox> data;
    int fileCounter = 0;

    if (file) {
        std::string line;
        while (std::getline(file, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            if (!line.empty()) {
                size_t delimiterPos = line.find(';');
                std::string idString = line.substr(0, delimiterPos);
                std::string coordinatesString = line.substr(delimiterPos + 1);
                int id = std::stoi(idString.substr(3));
                std::vector<int> coordinates;
                size_t start = 1;
                while (start < coordinatesString.size()) {
                    size_t end = coordinatesString.find(',', start);
                    if (end == std::string::npos) {
                        end = coordinatesString.size() - 1;
                    }
                    std::string coordinate = coordinatesString.substr(start, end - start);
                    coordinates.push_back(std::stoi(coordinate));
                    start = end + 1;
                }
                BoundingBox box;
                box.file = fileCounter;
                box.id = id;
                box.x1 = coordinates[0];
                box.y1 = coordinates[1];
                box.width = coordinates[2];
                box.height = coordinates[3];
                data.push_back(box);
            }
        }
    }

    return data;
}

cv::Mat loadSemanticSegmentationData(const std::string& filePath) {
    return cv::imread(filePath, cv::IMREAD_GRAYSCALE);
}

float processBoxPreds(const std::vector<std::vector<BoundingBox>>& resultData, const std::vector<std::vector<BoundingBox>>& predData) {
    // Confusion matrix data
    std::vector<int> TP(14, 0);
    std::vector<int> FP(14, 0);
    std::vector<int> FN(14, 0);
    // Recall, precision vectors, updated at each iteration for later use in the average precision calculation
    std::vector<std::vector<float>> recall(14, std::vector<float>(0, 0));
    std::vector<std::vector<float>> precision(14, std::vector<float>(0, 0));


    // For each file in both resultData and predData
    for(int i = 0; i < resultData.size(); i++) {
        for(int j = 0; j < predData.size(); j++) {
                // Compare each result with each pred
            for(int k = 0; k < resultData[i].size(); k++) {

                // False negative case, check in all predData if exist a box with the same id
                bool found = false;
                for(int l = 0; l < predData[j].size(); l++) {
                    if(resultData[i][k].file == predData[j][l].file && resultData[i][k].id == predData[j][l].id) {
                        found = true;
                    }
                }
                if(!found) {
                    FN[resultData[i][k].id]++;
                    // Compute precision and recall at the time point and update the vectors
                    precision[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FP[resultData[i][k].id]));
                    recall[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FN[resultData[i][k].id]));
                }

                for(int l = 0; l < predData[j].size(); l++) {
                        // If the result and pred have the same file and id
                        // Calculate the iou
                    if(resultData[i][k].file == predData[j][l].file && resultData[i][k].id == predData[j][l].id) {
                        int xA = std::max(resultData[i][k].x1, predData[j][l].x1);
                        int yA = std::max(resultData[i][k].y1, predData[j][l].y1);
                        int xB = std::min(resultData[i][k].x1 + resultData[i][k].width, predData[j][l].x1 + predData[j][l].width);
                        int yB = std::min(resultData[i][k].y1 + resultData[i][k].height, predData[j][l].y1 + predData[j][l].height);

                        float interArea = std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1);

                        float boxAArea = (resultData[i][k].width + 1) * (resultData[i][k].height + 1);
                        float boxBArea = (predData[j][l].width + 1) * (predData[j][l].height + 1);

                        float iou = interArea / (boxAArea + boxBArea - interArea);

        // If the IOU is greater than the threshold, it is a true positive
        // If the IOU is less than the threshold, it is a false positive
                        if(iou > IOU_THRESH) {
                            TP[resultData[i][k].id]++;
                             // Compute precision and recall at the time point and update the vectors
                            precision[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FP[resultData[i][k].id]));
                            recall[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FN[resultData[i][k].id]));
                        } else {
                            FP[resultData[i][k].id]++;
                            // Compute precision and recall at the time point and update the vectors
                            precision[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FP[resultData[i][k].id]));
                            recall[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FN[resultData[i][k].id]));
                        }
                    }

                }
            }
        

        }
    }

    // Compute average precision for class
    std::vector<float> AP(14, 0);



    // Compute mAP
    float mAP = 0.0f;
    for (float value : AP) {
        mAP += value;
    }
    mAP = mAP / AP.size();
    return mAP;
}

std::vector<cv::Mat> processSemanticSegmentation(const std::string& relativePath, const std::string& folder) {
    std::vector<cv::Mat> images;
    std::string boundingBoxDir;

    for (const auto& entry : fs::recursive_directory_iterator(relativePath)) {
        if (entry.is_directory() && entry.path().filename() == folder) {
            boundingBoxDir = entry.path().string();
        }
    }

    if (!boundingBoxDir.empty()) {
        for (const auto& entry : fs::directory_iterator(boundingBoxDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                cv::Mat data = loadSemanticSegmentationData(entry.path().string());
                if (!data.empty()) {
                    images.push_back(data);
                }
            }
        }
    }

    return images;
}