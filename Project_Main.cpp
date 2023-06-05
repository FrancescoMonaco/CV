#include <stdexcept>
#include "processing.h"
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // check argc
    vector<cv::Mat> images;

    string folder("/Users/franc/Downloads/Food_leftover_dataset/tray8/*.jpg");
    vector<cv::String> filenames;
    glob(folder, filenames, false);

    for (auto& str : filenames) {
        Mat img = imread(str);
        images.push_back(img);
    }

    for (auto& image : images) {
        //imshow("img", image);
        std::vector<Vec3f> circles;
        Mat grayscale;
        Mat mask = Mat::zeros(image.size(), CV_8UC1);
        cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
        HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, grayscale.rows/4, 120, 10, 250, 10);
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the outline
            //circle(image, center, radius, Scalar(0, 30, 205), 2, LINE_AA);
            circle(mask, center, radius, Scalar(255), -1);
        }
        Mat result;
        image.copyTo(result, mask);
        imshow("Masked", result);

       // cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
        //imshow("gray", grayscale);
        waitKey(0);
    }

    
    waitKey(0);
    return 0;
}
