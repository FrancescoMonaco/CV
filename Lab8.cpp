#include "processing_8.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // check argc
    vector<cv::Mat> images;

    string folder("/Users/franc/Downloads/calibr/*.png");
    vector<cv::String> filenames;
    glob(folder, filenames, false);

    for (auto& str : filenames) {
        Mat img = imread(str);
        images.push_back(img);
    }

    calibrate(images);

    waitKey(0);
    return 0;
}