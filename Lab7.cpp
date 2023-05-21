#include "processing_7.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // check argc
    vector<Mat> images;

    string folder("/Users/franc/Downloads/kitchen/*.bmp");
    vector<cv::String> filenames;
    cv::glob(folder, filenames, false);

    for (auto& str : filenames) {
        Mat img = imread(str);
        images.push_back(img);
    }

    cout << filenames.size() << endl << images.size() << endl;
    stitch(images);

    waitKey(0);
    return 0;
}
