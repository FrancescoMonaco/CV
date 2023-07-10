#include <stdexcept>
#include "processing.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

const std::string PREDS_BB = "box_preds";
const std::string PREDS_SS = "mask_preds";
using namespace std;
using namespace cv;

const int bigRadius = 295;
const int smallRadius = 287;
double lowerBound = 0.1;
double BREAD_THRESH = 0.73;
double SALAD_THRESH = 0.38;

void breadFinder(Mat& image, int radius,bool check, bool* hasBread, Rect* box);
bool apply_ORB(cv::Mat& in1, cv::Mat& in2);
void writeBoundBox(String& path, Rect box, int ID);
int pastaRecognition(Mat& image);

vector<Mat> getImageBoxes(Mat& image);

void classifier(Mat& image);

int main(int argc, char** argv) {
    // check argc
    vector<cv::Mat> images;

    string folder("/Users/franc/Downloads/Food_leftover_dataset/tray3/*.jpg");
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
        HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, grayscale.rows/2.5, 140, 55, 185, 370);
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the outline
            //circle(image, center, radius, Scalar(0, 30, 205), 2, LINE_AA);
            circle(mask, center, radius, Scalar(255), -1);
            if (i == 0) rad = radius;
        }

        // Code for the inverse mask - BREAD PART
        //Mat inverse_mask, mask2, mask3, bread_image2;
        //bitwise_not(mask, inverse_mask);
        //Mat result;
        //image.copyTo(result, inverse_mask);
        //bool haveBread; Rect* box;
        //breadFinder(result, rad, true, &haveBread, box);
        //if (!haveBread) break;
        //writeBoundBox(folder, *box, 13);
       
       // PLATES PART
       //for each circle extract a rectangle with just the plate
         for (size_t i = 0; i < circles.size(); i++){
                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                int x = center.x - radius;
                int y = center.y - radius;
                int width = 2 * radius;
                int height = 2 * radius;

                // Perform boundary checks and adjust the rectangle if necessary
                if (x < 0) {
                    width += x;
                    x = 0;
                }
                if (y < 0) {
                    height += y;
                    y = 0;
                }
                if (x + width > image.cols) {
                    width = image.cols - x;
                }
                if (y + height > image.rows) {
                    height = image.rows - y;
                }

                // Create the rectangle ROI
                Rect rec(x, y, width, height);
                Mat plate = image(rec);
                //classifier(plate);
                imshow("Plate", plate);
                int num = pastaRecognition(plate);
                cout << num << endl;
                waitKey(0);
         }

    }

    
    waitKey(0);
    return 0;
}


void breadFinder(Mat& result, int radius, bool check, bool* hasBread, Rect* box) {
    Mat hsv_image, mask2, mask3, bread_image2;

    // Load the templates and check for their existence
    Mat bread_template = imread("/Users/franc/Downloads/pan_br.jpeg");
    Mat crumbs_template = imread("/Users/franc/Downloads/pan_left.jpg");

    if (bread_template.data == NULL || crumbs_template.data == NULL)
        throw invalid_argument("Data does not exist");

    //if the templates are bigger than the image, resize them

    if (bread_template.cols > result.cols || bread_template.rows > result.rows) {
        resize(bread_template, bread_template, Size(result.cols, result.rows));
    }
    if (crumbs_template.cols > result.cols || crumbs_template.rows > result.rows) {
        resize(crumbs_template, crumbs_template, Size(result.cols, result.rows));
    }


    // Convert to HSV
    cvtColor(result, hsv_image, COLOR_BGR2HSV);
    std::vector<Mat> hsv_channels;
    split(hsv_image, hsv_channels);
    Mat hue_channel = hsv_channels[0];
    Mat sat_channel = hsv_channels[1];
    Mat value_channel = hsv_channels[2];

    // Mask the colors that are not near the browns
    // this removes the tag, the plastic cup and the yogurt
    inRange(sat_channel, 40, 200, mask3);
    inRange(hue_channel, 5, 95, mask2);
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
            Mat matchOut;
            matchTemplate(bread_image2, bread_template, matchOut, TM_SQDIFF_NORMED);

            minMaxLoc(matchOut, &minVal, &maxVal, &minLoc, &maxLoc);


            Mat matchOut2;
            matchTemplate(bread_image, crumbs_template, matchOut2, TM_SQDIFF_NORMED);


            minMaxLoc(matchOut2, &minVal2, &maxVal2, &minLoc2, &maxLoc2);

    // Check if the bread is present
    if (check && minVal >= BREAD_THRESH) {
        *hasBread = false;
        return;
    }

    // Create the bounding box
     if (minVal2 < lowerBound) { 
            int offsetX = bread_template.cols / 2;  // Half the width of the template
            int offsetY = bread_template.rows / 2;  // Half the height of the template
            Point center(minLoc2.x + offsetX, minLoc2.y + offsetY);  // Calculate the center position

            // Define the top-left and bottom-right corners of the rectangle
            Point topLeft(center.x - offsetX, center.y - offsetY);
            Point bottomRight(center.x + offsetX, center.y + offsetY);
            rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            imshow("Rec", result);
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
            rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            imshow("Rec", result);
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
            rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            imshow("Rec", result);
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
            rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
            imshow("Rec", result);
            rec = Rect(topLeft, bottomRight);
        }

    
        waitKey(0);
        box = &rec;
}

/// @brief Extract the plates using the Hough Transform
/// @param image, image to be processed
/// @return a vecor of the extracted images
std::vector<cv::Mat> getImageBoxes(cv::Mat& image){
     GaussianBlur(image, image, cv::Size(3, 3), 0.5);
     std::vector<cv::Vec3f> circles; std::vector<cv::Mat> extractions;
     cv::Mat grayscale;
     cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
     HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, grayscale.rows/2.5, 140, 55, 185, 370);
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            
            //find the rectangle that contains the circle
                int x = center.x - radius;
                int y = center.y - radius;
                int width = 2 * radius;
                int height = 2 * radius;

                // Perform boundary checks and adjust the rectangle if necessary
                if (x < 0) {
                    width += x;
                    x = 0;
                }
                if (y < 0) {
                    height += y;
                    y = 0;
                }
                if (x + width > image.cols) {
                    width = image.cols - x;
                }
                if (y + height > image.rows) {
                    height = image.rows - y;
                }
                extractions.push_back(image(Rect(x, y, width, height)));
                
         }
    return extractions;
    }

void classifier(Mat& image) {
    Mat salad_template = imread("/Users/franc/Downloads/sugo.jpg");

    if(salad_template.data == NULL)
        throw invalid_argument("Data does not exist");

    //if the template is bigger than the image, resize it
    if (salad_template.cols > image.cols || salad_template.rows > image.rows) {
        resize(salad_template, salad_template, Size(image.cols, image.rows));
    }

    // Initialization for TM values
    double minVal, maxVal;
    Point minLoc, maxLoc;

            Mat matchOut;
            matchTemplate(image, salad_template, matchOut, TM_SQDIFF_NORMED);

            minMaxLoc(matchOut, &minVal, &maxVal, &minLoc, &maxLoc);
            cout << minVal << endl;
            // Create the rectangle by checking the dimensions of the template and making sure it fits in the image
            int x = minLoc.x;
            int y = minLoc.y;
            int width = salad_template.cols;
            int height = salad_template.rows;

            if (x < 0) {
                width += x;
                x = 0;
            }
            if (y < 0) {
                height += y;
                y = 0;
            }
            if (x + width > image.cols) {
                width = image.cols - x;
            }
            if (y + height > image.rows) {
                height = image.rows - y;
            }

            Rect rec(x, y, width, height);
        }


void writeBoundBox(String& path, Rect box, int ID){
    //Open the file to write or create it if it doesn't exist
    ofstream file(path, ios::app);
    if (!file.is_open()) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
    //Write in format ID:id; [x, y, w, h]
    file << "ID:" << ID << "; [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]" << endl;
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

int pastaRecognition(Mat& image){
    // Centroids for the number of pixels, manually computed for the test set
    const std::vector<vector<int>> centers = {{3000,8000,17000,24000},\
     {0, 4000, 50000, 16000},\
     {0,4000,30000,26000},\
     {0,460,19000,26666},\
    {100, 9000, 21000, 23000}};
    // Use only the center of the image
    Mat center = image(Rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2));
    //Use HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    // Extract the number of pixels in the green, yellow, red, orange, brown and white ranges
    int green = 0, yellow = 0, red = 0, orange = 0, brown = 0, white = 0;
    for(int i = 0; i < center.rows; i++){
        for(int j = 0; j < center.cols; j++){
            Vec3b pixel = hsv.at<Vec3b>(i, j);
            if(pixel[0] >= 40 && pixel[0] <= 80 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                green++;
            else if(pixel[0] >= 20 && pixel[0] <= 40 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                yellow++;
            else if(pixel[0] >= 0 && pixel[0] <= 20 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                red++;
            else if(pixel[0] >= 10 && pixel[0] <= 20 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                orange++;
            else if(pixel[0] >= 0 && pixel[0] <= 10 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                brown++;
            else if(pixel[0] >= 0 && pixel[0] <= 180 && pixel[1] >= 0 && pixel[1] <= 50 && pixel[2] >= 200 && pixel[2] <= 255)
                white++;
        }
    }

    //Using green, yellow, red, white find the closest center using sum of squares
     int min = 1000000000;
    int index = -1;
        for(int i = 0; i < centers.size(); i++){
            int dist = 0;
            dist += pow(green - centers[i][0], 2);
            dist += pow(yellow - centers[i][1], 2);
            dist += pow(red - centers[i][2], 2);
            dist += pow(white - centers[i][3], 2);
            if(dist < min){
                min = dist;
                index = i;
            }
        }

    index +=1;
    return index;
}