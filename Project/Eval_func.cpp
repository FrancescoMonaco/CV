#include "Eval_func.h"
#include <math.h>
#include <numeric>
#include <iomanip>

//***Variables, namespaces definitions***
const std::string GOLD_BB = "/bounding_boxes";
const std::string PREDS_BB = "/box_preds";
const std::string GOLD_SS = "/masks";
const std::string PREDS_SS = "/mask_preds";
const float IOU_THRESH = 0.5;
const int NUM_CLASSES = 14;
const std::vector<std::string> classes = {"background", "pasta with pesto",\
 "pasta with tomato sauce", "pasta with meat sauce", "pasta with clams and mussels", "pilaw rice with peppers and peas",\
  "grilled pork cutlet", "fish cutlet", "rabbit", "seafood salad",\
   "beans", "basil potatoes", "salad", "bread"};
std::vector<cv::Vec3b> customColormap = {
    cv::Vec3b(255, 0, 0),     // Red
    cv::Vec3b(0, 255, 0),     // Green
    cv::Vec3b(0, 0, 255),     // Blue
    cv::Vec3b(255, 255, 0),   // Yellow
    cv::Vec3b(255, 0, 255),   // Magenta
    cv::Vec3b(0, 255, 255),   // Cyan
    cv::Vec3b(255, 128, 0),   // Orange
    cv::Vec3b(128, 0, 255),   // Purple
    cv::Vec3b(0, 255, 128),   // Lime
    cv::Vec3b(255, 128, 128), // Pink
    cv::Vec3b(128, 255, 128), // Light Green
    cv::Vec3b(128, 128, 255), // Light Blue
    cv::Vec3b(255, 255, 128), // Light Yellow
    cv::Vec3b(128, 255, 255)  // Light Cyan
};

namespace fs = std::filesystem;

//***Functions implementations***

std::vector<BoundingBox> loadBoundingBoxData(const std::string& filePath, const int fileNum) {
    std::ifstream file(filePath);
    std::vector<BoundingBox> data;

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
                box.file = fileNum;
                box.id = id;
                box.x1 = coordinates[0];
                box.y1 = coordinates[1];
                box.width = coordinates[2];
                box.height = coordinates[3];
                data.push_back(box);
                // print the data as a test
                //std::cout << "File: " << box.file << " ID: " << box.id << " x1: " << box.x1 << " y1: " << box.y1 << " width: " << box.width << " height: " << box.height << std::endl;
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
            //if the file is the same
            if(resultData[i][0].file == predData[j][0].file) {

                // Compare each result with each pred
            for(int k = 0; k < resultData[i].size(); k++) {

                // False negative case, check in all predData if exist a box with the same id
                bool found = false;
                int l = 0;
                for(l = 0; l < predData[j].size(); l++) {
                    if(resultData[i][k].id == predData[j][l].id) {
                        found = true;
                        break;
                    }
                }
                if(!found) {
                    FN[resultData[i][k].id]++;
                    // Compute precision and recall at the time point and update the vectors
                    precision[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FP[resultData[i][k].id]));
                    recall[resultData[i][k].id].push_back((float)TP[resultData[i][k].id] / (TP[resultData[i][k].id] + FN[resultData[i][k].id]));
                }

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

    // Compute average precision for class
    std::vector<float> AP = computeAP(recall, precision);

    // Compute mAP
    float mAP = 0.0f;
    for (float value : AP) {
        mAP += value;
    }
    mAP = mAP / AP.size();
    return mAP;
}

float processSemanticSegmentation(const std::vector<cv::Mat>& resultData, const std::vector<cv::Mat>& predData)
{
    //for each couple of allineated images, compute IoU and push it into a vector
    std::vector<float> IoU;
    for (int i = 0; i < resultData.size(); i++) {
            float iou = computeIoU(resultData[i], predData[i]);
            IoU.push_back(iou);
            std::cout << "*";
    }

    // return mean IoU
    float mIoU = 0.0f;
    for (float value : IoU) {
        mIoU += value;
    }
    mIoU = mIoU / IoU.size();
    return mIoU;
}

float computeIoU(const cv::Mat & result, const cv::Mat & pred)
{
    //compute intersection and union for each class
    std::vector<float> intersection (NUM_CLASSES, 0.0f);
    std::vector<float> unionArea (NUM_CLASSES, 0.0f);
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            if (result.at<uchar>(i, j) == pred.at<uchar>(i, j)) {
                intersection[result.at<uchar>(i, j)]++;
            }
            unionArea[result.at<uchar>(i, j)]++;
            unionArea[pred.at<uchar>(i, j)]++;
        }
    }
    // union is the sum of the two areas minus the intersection
    for (int i = 0; i < NUM_CLASSES; i++) {
        unionArea[i] -= intersection[i];
    }
    // compute IoU for each class
    std::vector<float> IoU (NUM_CLASSES, 0.0f);
    for (int i = 0; i < NUM_CLASSES; i++) {
        //don't compute if for classes that have 0 in result
        if (unionArea[i] == 0) {
            continue;
        }
        IoU[i] = intersection[i] / unionArea[i];
    }
    // return mean IoU
    float mIoU = 0.0f;
    for (float value : IoU) {
        mIoU += value;
    }
    // divide by number of classes that do not have 0 in IoU
    int count = std::count_if(IoU.begin(), IoU.end(), [](float i){return i > 0.0f;});
    mIoU = mIoU / count;
    return mIoU;
}

std::vector<float> computeAP(const std::vector<std::vector<float>>& precision, const std::vector<std::vector<float>>& recall) {
    std::vector<float> ap;
    // we start from 1 to skip the background class
    for (size_t i = 1; i < precision.size(); ++i) {
        const std::vector<float>& prec = precision[i];
        const std::vector<float>& rec = recall[i];

        // Sort precision and recall vectors in descending order of recall
        std::vector<size_t> indices(prec.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&rec](size_t a, size_t b) { return rec[a] > rec[b]; });

        // Compute interpolated precision, 
        std::vector<float> interpolatedPrec(indices.size());
        float maxPrec = 0.0f;
        for (size_t j = 0; j < indices.size(); ++j) {
            size_t idx = indices[j];
            maxPrec = std::max(maxPrec, prec[idx]);
            interpolatedPrec[j] = maxPrec;
        }

        // Compute average precision
        float sumPrec = std::accumulate(interpolatedPrec.begin(), interpolatedPrec.end(), 0.0f);
        ap.push_back(sumPrec / static_cast<float>(interpolatedPrec.size()));
    }

    return ap;
}

void foodLeftoverEstimation(const std::string& goldPath, const std::string& predPath)
{
    // Compute on the ground truth
        // Open all the segmentation images and put them in a vector
    std::vector<cv::Mat> goldData;
    for (const auto& entry : std::filesystem::directory_iterator(goldPath)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        goldData.push_back(img);
    }

        // Compare the first image (before) with all the others (after), compute for each category the ratio between the
        // number of pixels with that label in the after image and the number of pixels with that label in the before image
        // if there's no pixel with that label put -1 as a flag
    std::vector<std::vector<float>> goldRatios;

    for (int i = 1; i < goldData.size(); i++) {
        std::vector<float> ratios;
        for (int j = 0; j < NUM_CLASSES; j++) {
            float ratio = 0.0f;
            if (cv::countNonZero(goldData[0] == j) != 0) {
                ratio = (float)cv::countNonZero(goldData[i] == j) / (float)cv::countNonZero(goldData[0] == j);
            } else {
                ratio = -1.0f;
            }
            ratios.push_back(ratio);
        }
        goldRatios.push_back(ratios);
    }

    // Compute on the prediction
        // Open all the segmentation images and put them in a vector
    std::vector<cv::Mat> predData;

    for (const auto& entry : std::filesystem::directory_iterator(predPath)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        predData.push_back(img);
    }

        // Compare the first image (before) with all the others (after), compute for each category the ratio between the
        // number of pixels with that label in the after image and the number of pixels with that label in the before image
        // if there's no pixel with that label put -1 as a flag
    std::vector<std::vector<float>> predRatios;

    for (int i = 1; i < predData.size(); i++) {
        std::vector<float> ratios;
        for (int j = 0; j < NUM_CLASSES; j++) {
            float ratio = 0.0f;
            if (cv::countNonZero(predData[0] == j) != 0) {
                ratio = (float)cv::countNonZero(predData[i] == j) / (float)cv::countNonZero(predData[0] == j);
            } else {
                ratio = -1.0f;
            }
            ratios.push_back(ratio);
        }
        predRatios.push_back(ratios);
    }

    // Print tray, category, food id and value

    std::string numTray = goldPath.substr(goldPath.find("tray") + 4, 1);
    std::cout << "Tray " << numTray << std::endl;
    std::cout << "Gold ratios:" << std::endl;

    for(int i = 0; i < goldRatios.size(); i++) {
        std::cout << "Image " << i << ": ";
        for (int j = 1; j < goldRatios[i].size(); j++) {
            if (goldRatios[i][j] != -1.0f){
                std::cout << std::setw(3) << j << " ";
                std::cout << std::fixed << std::setprecision(2) << std::setw(5) << goldRatios[i][j] << " ";
            }
        }
        std::cout << std::endl;
    }



    std::cout << "Pred ratios:" << std::endl;
    for (int i = 0; i < predRatios.size(); i++) {
        std::cout << "Image " << i << ": ";

        for (int j = 1; j < predRatios[i].size(); j++) {
            if (predRatios[i][j] != -1.0f) {
                std::cout << std::setw(3) << j << " ";
                std::cout << std::fixed << std::setprecision(2) << std::setw(5) << predRatios[i][j] << " ";
            }
        }

        std::cout << std::endl;
    }
}

void showResults(const std::string & path)
{
    //save in a vector all the jpg in the folder
    std::vector<cv::String> fn;
    cv::glob(path + "/*.jpg", fn, false);

    //for each image show the corresponding bounding box
    for (size_t k = 0; k < fn.size(); ++k) {
        //read the image
        cv::Mat im = cv::imread(fn[k]);

        //read the corresponding txt, that is the subfolder path+GOLD_BB and has the same substring of the image
        std::string filename = fn[k].substr(fn[k].find_last_of('/\\') + 1);
        std::string nameWithoutExtension = filename.substr(0, filename.find_last_of('.'));
        std::string txtPath = path + GOLD_BB +"/" + nameWithoutExtension + "_bounding_box.txt";

        std::ifstream file(txtPath);
        //for each line in the txt
        std::string line;
        while (std::getline(file, line)) {

            //split the line
            std::istringstream iss(line);
            std::vector<std::string> results((std::istream_iterator<std::string>(iss)),
                                             std::istream_iterator<std::string>());

            //get the class and the coordinates, a line is like ID: 4; [247, 152, 367, 342]
            int classId = std::stoi(results[1]);
            int x = std::stoi(results[2].substr(1, results[2].size() - 2));
            int y = std::stoi(results[3].substr(0, results[3].size() - 1));
            int w = std::stoi(results[4].substr(0, results[4].size() - 1));
            int h = std::stoi(results[5].substr(0, results[5].size() - 1));

            //draw the bounding box
            cv::rectangle(im, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0));

            //put the class name
            std::string className = classes[classId];
            int baseLine;
            cv::Size labelSize = cv::getTextSize(className, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(im, className, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0));
        }

        //show the image with the bounding boxes
        cv::namedWindow("image", cv::WINDOW_GUI_NORMAL);
        cv::imshow("image", im);
        cv::waitKey(0);
    }

    //show the segmentation
    std::vector<cv::String> fn2;
    cv::glob(path + GOLD_SS +"/*.png", fn2, false);

    for (size_t k = 0; k < fn2.size(); ++k) {
        cv::Mat im = cv::imread(fn2[k], cv::IMREAD_GRAYSCALE);
        //Show the regions colored in overlay with the original image
        cv::Mat im_color(im.size(), CV_8UC3);
        for (int i = 0; i < im.rows; i++) {
            for (int j = 0; j < im.cols; j++) {
                int index = im.at<uchar>(i, j);
                im_color.at<cv::Vec3b>(i, j) = customColormap[index];
            }
        }

        double alpha = 0.1;
        cv::Mat blended;
        cv::Mat original_image = cv::imread(fn[k].substr(0, fn[k].find_last_of('.')) + ".jpg");
        cv::addWeighted(original_image, alpha, im_color, 1 - alpha, 0, blended);
        cv::namedWindow("Segmentation Overlay", cv::WINDOW_GUI_NORMAL);
        cv::imshow("Segmentation Overlay", blended);

        cv::waitKey(0);
    }

}