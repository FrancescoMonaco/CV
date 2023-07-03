#include <stdexcept>
#include "processing.h"
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

const int bigRadius = 295;
const int smallRadius = 287;
double lowerBound = 0.1;

void breadFinder(Mat& image, int radius);
bool apply_ORB(cv::Mat& in1, cv::Mat& in2);

int main(int argc, char** argv) {
    // check argc
    vector<cv::Mat> images;

    string folder("/Users/franc/Downloads/Food_leftover_dataset/tray5/*.jpg");
    vector<cv::String> filenames;
    glob(folder, filenames, false);

    for (auto& str : filenames) {
        Mat img = imread(str);
        images.push_back(img);
    }

    for (auto& image : images) {
        //imshow("img", image);
        GaussianBlur(image, image, Size(3, 3), 0.5);
        std::vector<Vec3f> circles;
        Mat grayscale;
        Mat mask = Mat::zeros(image.size(), CV_8UC1); // Mask initialization
        cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
        int rad;
        HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, grayscale.rows/3, 180, 60, 230, 350);
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the outline
            //circle(image, center, radius, Scalar(0, 30, 205), 2, LINE_AA);
            circle(mask, center, radius, Scalar(255), -1);
            cout << radius << endl;
            if (i == 0) rad = radius;
        }
        // Code for the inverse mask
        Mat inverse_mask, mask2, mask3, bread_image2;
        bitwise_not(mask, inverse_mask);
        Mat result;
        image.copyTo(result, inverse_mask);
        breadFinder(result, rad);
       

        waitKey(0);
    }

    
    waitKey(0);
    return 0;
}

void breadFinder(Mat& result, int radius) {
    Mat hsv_image, mask2, mask3, bread_image2;

    // Load the templates and check for their existence
    Mat bread_template = imread("/Users/franc/Downloads/pan_br.jpeg");
    Mat crumbs_template = imread("/Users/franc/Downloads/pan_left.jpg");

    if (bread_template.data == NULL || crumbs_template.data == NULL)
        throw invalid_argument("Data does not exist");

    // Convert to HSV
    cvtColor(result, hsv_image, COLOR_BGR2HSV);
    std::vector<Mat> hsv_channels;
    split(hsv_image, hsv_channels);
    Mat hue_channel = hsv_channels[0];
    Mat sat_channel = hsv_channels[1];
    Mat value_channel = hsv_channels[2];

    // Mask the colors that are not near the browns
    // this removes the tag, the plastic cup and the yogurt
    inRange(sat_channel, 40, 200, mask3); // only for tray 5
    inRange(hue_channel, 5, 95, mask2); //threshold for bread hue
    Mat bread_image;
    bitwise_and(result, result, bread_image, mask2);
    bitwise_and(bread_image, bread_image, bread_image2, mask3);
    
    // Initialization for TM values
    double minVal, maxVal;
    Point minLoc, maxLoc;
    double minVal2, maxVal2;
    Point minLoc2, maxLoc2;
    Mat mask, temp1, temp2;
    Rect rec;

    // Bread matching
    if (1) { // the if is used to easily delete the local variables

            imshow("Hue", bread_image2);


            Mat matchOut;
            matchTemplate(bread_image2, bread_template, matchOut, TM_SQDIFF_NORMED);

            minMaxLoc(matchOut, &minVal, &maxVal, &minLoc, &maxLoc);

            cout << "Bread " << minVal << " ";

    }
    if(1) {

            Mat matchOut2;
            matchTemplate(bread_image, crumbs_template, matchOut2, TM_SQDIFF_NORMED);


            minMaxLoc(matchOut2, &minVal2, &maxVal2, &minLoc2, &maxLoc2);
            cout << "crumbs " << minVal2 << endl;

        }

    // Create the bounding box
     if (minVal2 < lowerBound) { 
            int offsetX = bread_template.cols / 2;  // Half the width of the template
            int offsetY = bread_template.rows / 2;  // Half the height of the template
            Point center(minLoc2.x + offsetX, minLoc2.y + offsetY);  // Calculate the center position

            // Define the top-left and bottom-right corners of the rectangle
            Point topLeft(center.x - offsetX, center.y - offsetY);
            Point bottomRight(center.x + offsetX, center.y + offsetY);
            //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            //imshow("Rec", result);
            rec = Rect(topLeft, bottomRight);
        }
        // Choose window size based on the radius of the plates
     else if (radius <= smallRadius) {
            int offsetX = bread_template.cols / 2;  // Half the width of the template
            int offsetY = bread_template.rows / 2;  // Half the height of the template
            Point center(minLoc.x + offsetX, minLoc.y + offsetY);  // Calculate the center position

            // Define the top-left and bottom-right corners of the rectangle
            Point topLeft(center.x - offsetX, center.y - offsetY);
            Point bottomRight(center.x + offsetX, center.y + offsetY);
            //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            //imshow("Rec", result);
            rec = Rect(topLeft, bottomRight);
        }
     else if (radius > bigRadius) {
            int offsetX = bread_template.cols / 2;  // Half the width of the template
            int offsetY = bread_template.rows / 2;  // Half the height of the template
            Point center(minLoc.x + offsetX, minLoc.y + offsetY);  // Calculate the center position

            // Define the top-left and bottom-right corners of the rectangle
            // as a bigger box due to perspective
            Point topLeft(center.x - 1.5*offsetX, center.y - 1.5*offsetY);
            Point bottomRight(center.x + 1.5*offsetX, center.y + 1.5*offsetY);
            //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            //imshow("Rec", result);
            rec = Rect(topLeft, bottomRight);
        }
     else {
            int offsetX = bread_template.cols / 2;  // Half the width of the template
            int offsetY = bread_template.rows / 2;  // Half the height of the template
            Point center(minLoc.x + offsetX, minLoc.y + offsetY);  // Calculate the center position

            // Define the top-left and bottom-right corners of the rectangle
            // as a slightly bigger box due to perspective
            Point topLeft(center.x - 0.4*offsetX, center.y - 1.5 * offsetY);
            Point bottomRight(center.x + 1.1*offsetX, center.y + 1.2 * offsetY);
            //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            //imshow("Rec", result);
            rec = Rect(topLeft, bottomRight);
        }

    

    mask = Mat::ones(result.size(), CV_32FC1);
    //mask(rec) = 0;
    Mat roiMask = mask(rec);
    roiMask = 0;
    mask.at<float>(minLoc) = 2;

    watershed(result, mask);
    vector<Vec3b> colors;
    colors.push_back(Vec3b(0, 0, 255));
    colors.push_back(Vec3b(0, 255, 255));
    colors.push_back(Vec3b(255, 255, 255));

    Mat dst = Mat::zeros(mask.size(), CV_8UC3);
    result.copyTo(dst);
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            int index = mask.at<int>(i, j);
                dst.at<Vec3b>(i, j) = colors[index - 1];
      
        }
    }
    imshow("Result", dst);
    waitKey(0);
}


bool apply_ORB(cv::Mat& in1, cv::Mat& in2)
{
    // Variables initialization
    int tresh = 4;
    double ratio = 0.75;
    Mat out, out_match;
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> vec, vec2;

    // Find the keypoints in the two images
    (*orb).detect(in1, vec);
    (*orb).detect(in2, vec2);

    // Show them
    drawKeypoints(in1, vec, out);
    //namedWindow("ORB Keypoints", WINDOW_GUI_NORMAL);
    //imshow("ORB Keypoints", out);

    Mat descriptors1, descriptors2;
    orb->compute(in1, vec, descriptors1);
    orb->compute(in2, vec2, descriptors2);

    // Create a brute force matcher and match the features
    Ptr<BFMatcher> matcher = BFMatcher::create();
    vector<vector<DMatch>> matches;
    vector<vector<DMatch>> pruned_matches;

    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

    for (int i = 0; i < matches.size(); i++) {
        // check the similar matches
        if (matches[i][0].distance < ratio * matches[i][1].distance) {
            pruned_matches.push_back(matches[i]);
        }
    }


    // show the matches
    drawMatches(in1, vec, in2, vec2, pruned_matches, out_match);
    namedWindow("ORB Matches", WINDOW_GUI_NORMAL);
    imshow("ORB Matches", out_match);

    // print the result
    if (pruned_matches.size() > tresh)
        return true;
    return false;
}