#include <stdexcept>
#include "processing.h"
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

void breadFinder(Mat image);
bool apply_ORB(cv::Mat& in1, cv::Mat& in2);

int main(int argc, char** argv) {
    // check argc
    vector<cv::Mat> images;

    string folder("/Users/franc/Downloads/Food_leftover_dataset/tray1/*.jpg");
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
        HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, grayscale.rows/3, 180, 60, 230, 350);
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the outline
            //circle(image, center, radius, Scalar(0, 30, 205), 2, LINE_AA);
            circle(mask, center, radius, Scalar(255), -1);
        }
        // Code for the inverse mask
        Mat inverse_mask, mask2, mask3, bread_image2;
        bitwise_not(mask, inverse_mask);
        Mat result;
        image.copyTo(result, inverse_mask);
        breadFinder(result);
       
  
       // Mat gradient;
        //Sobel(bread_image, gradient, CV_8U, 2, 2);
       // imshow("Gradient", gradient);
       // 
       // watershed(gradient, bread_image);

       // Mat edges, img;
      //  bilateralFilter(bread_image, img, 5, 150, 150);
        //imshow("Bread Image", img);
        //Canny(img, edges, 10, 100, 3);
        //imshow("Canny image", edges);
       // kmeans(bread_image);
       
        /*
        cvtColor(result, result, COLOR_BGR2HLS);
        Mat chan[3];
        //watershed(result);
        split(result, chan);
        //imshow("Green", chan[1]);
        imshow("Red", chan[2]);
        //watershed(chan[2]);
        //imshow("Blue", chan[0]);
        threshold(chan[2], chan[2], 100, 235, THRESH_BINARY | THRESH_OTSU);
        imshow("Segmented Image", chan[2]);
        //imshow("gray", grayscale);
        */
        waitKey(0);
    }

    
    waitKey(0);
    return 0;
}

void breadFinder(Mat result) {
    Mat hsv_image, mask2, mask3, bread_image2;

    Mat bread_template = imread("/Users/franc/Downloads/pan_br.jpg");
    Mat crumbs_template = imread("/Users/franc/Downloads/pan_left.jpg");

    if (bread_template.data == NULL || crumbs_template.data == NULL)
        throw invalid_argument("Data does not exist");

    cvtColor(result, hsv_image, COLOR_BGR2HSV);
    std::vector<Mat> hsv_channels;
    split(hsv_image, hsv_channels);
    Mat hue_channel = hsv_channels[0];
    Mat sat_channel = hsv_channels[1];
    Mat value_channel = hsv_channels[2];

    inRange(sat_channel, 40, 170, mask3); // only for tray 5
    inRange(hue_channel, 5, 95, mask2); //threshold for bread hue
    Mat bread_image;
    bitwise_and(result, result, bread_image, mask2);
    bitwise_and(bread_image, bread_image, bread_image2, mask3);
    
    bool isBread = apply_ORB(bread_template, bread_image2);
    cout << "Bread matches: " << isBread << endl;
    if (isBread) {
        // BREAD

            imshow("Hue", bread_image2);


            Mat matchOut;
            matchTemplate(result, bread_template, matchOut, TM_SQDIFF);

            double* minVal = {}, * maxVal = {};
            Point minLoc, maxLoc;

            minMaxLoc(matchOut, minVal, maxVal, &minLoc, &maxLoc);
            //
            int offsetX = bread_template.cols / 2;  // Half the width of the template
            int offsetY = bread_template.rows / 2;  // Half the height of the template
            Point center(minLoc.x + offsetX, minLoc.y + offsetY);  // Calculate the center position

            // Define the top-left and bottom-right corners of the rectangle
            Point topLeft(center.x - offsetX, center.y - offsetY);
            Point bottomRight(center.x + offsetX, center.y + offsetY);

            rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            imshow("Rec", result);
            waitKey(0);
    }
    else {
        bool isCrumb = apply_ORB(crumbs_template, result);
        if (!isCrumb) return; // No bread on the tray
        else {
            Mat matchOut;
            matchTemplate(result, crumbs_template, matchOut, TM_SQDIFF);

            double* minVal = {}, * maxVal = {};
            Point minLoc, maxLoc;

            minMaxLoc(matchOut, minVal, maxVal, &minLoc, &maxLoc);
            // Create the rectangle
            int offsetX = crumbs_template.cols / 2;  // Half the width of the template
            int offsetY = crumbs_template.rows / 2;  // Half the height of the template
            Point center(minLoc.x + offsetX, minLoc.y + offsetY);  // Calculate the center position

            // Define the top-left and bottom-right corners of the rectangle
            Point topLeft(center.x - offsetX, center.y - offsetY);
            Point bottomRight(center.x + offsetX, center.y + offsetY);

            rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            imshow("Rec", result);
            waitKey(0);

        }
    }

   
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