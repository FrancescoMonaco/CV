#include <stdexcept>
#include "processing.h"
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

void watershed(Mat in);
void kmeans(Mat& img);
int main(int argc, char** argv) {
    // check argc
    vector<cv::Mat> images;

    string folder("/Users/franc/Downloads/Food_leftover_dataset/tray1/*.jpg");
    vector<cv::String> filenames;
    glob(folder, filenames, false);


    Mat bread_template = imread("/Users/franc/Downloads/pan_br.jpeg");
    if (bread_template.data == NULL)
        throw invalid_argument("Data does not exist");

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

        Mat hsv_image;
        cvtColor(result, hsv_image, COLOR_BGR2HSV);
        std::vector<Mat> hsv_channels;
        split(hsv_image, hsv_channels);

        Mat hue_channel = hsv_channels[0];
        Mat sat_channel = hsv_channels[1];
        Mat value_channel = hsv_channels[2];
        inRange(sat_channel, 40, 200, mask3); // only for tray 5
        inRange(hue_channel, 5, 95, mask2); //threshold for bread hue
        Mat bread_image, matchOut;
        bitwise_and(result, result, bread_image, mask2);
        bitwise_and(bread_image, bread_image, bread_image2, mask3);
        imshow("Hue", bread_image2);


        
        matchTemplate(bread_image2, bread_template, matchOut, TM_SQDIFF);

        //imshow("Output", matchOut);
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

        rectangle(result, topLeft, bottomRight, Scalar(0,0,255));
        imshow("Rec", result);

        // Create a mask image with the same size as the input image
        Mat zero_mask = Mat::zeros(image.size(), CV_8UC1);

        // Set the region outside the rectangle to black in the mask
        rectangle(zero_mask, Rect(topLeft, bottomRight), Scalar(255), FILLED);

        // Apply the mask to the original image
        Mat alone, alones;
        Rect ext(topLeft, bottomRight);
        alone = image(ext);
        cvtColor(alone, alone, COLOR_BGR2GRAY);
        medianBlur(alone, alone, 7);
        threshold(alone, alones, 50, 250, THRESH_BINARY_INV | THRESH_TRIANGLE);
        imshow("Seg", alones);
  
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

void watershed(Mat src) {
    int binThresh = 160;
    float distThresh = 0.88;
    // Mask the black
    Mat mask;
    inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(0, 0, 0), mask);

    // Blur the image
    Mat sharp;
    Mat imgResult;
    medianBlur(src, sharp, 9);
    medianBlur(sharp, imgResult, 5);
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);

    //imshow("Blurred", imgResult);

    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, binThresh, 255, THRESH_OTSU);
    imshow("Binary Image", bw);

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);

    //imshow("Distance Transform Image", dist);

    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, distThresh, 1.0, THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1, Point(-1, 1), 1);
    //imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }
    // Draw the background marker
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    //imshow("Markers", markers8u);
    // Perform the watershed algorithm
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }
    // Visualize the final image
    namedWindow("Final Result", WINDOW_GUI_NORMAL);
    imshow("Final Result", dst);
}

void kmeans(cv::Mat& img)
{
    // Blur the image
    Mat temp, dst, data;
    medianBlur(img, temp, 9);
    GaussianBlur(temp, img, Size(3, 3), 3);

    /*
    Vec3b colorRef(20, 30, 170); // yellows
    Vec3b buf(155, 155, 190);

    // Mask the colors
    temp, dst = Mat::zeros(img.rows, img.cols, CV_8U);
    inRange(img, colorRef, buf, temp);
    cvtColor(temp, temp, COLOR_GRAY2BGR);
    bitwise_and(img, temp, dst);

    //imshow("masked", temp);
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());
    */

    img.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());
    // do kmeans
    Mat labels, centers;
    kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3,
        KMEANS_PP_CENTERS, centers);

    // reshape both to a single row of Vec3f pixels:
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    // replace pixel values with their label:
    Vec3f* p = data.ptr<Vec3f>();
    for (size_t i = 0; i < data.rows; i++) {
        int center_id = labels.at<int>(i);
        p[i] = centers.at<Vec3f>(center_id);
    }

    // back to 2d, and uchar:
    img = data.reshape(3, img.rows);
    img.convertTo(img, CV_8U);
    namedWindow("Clustering", WINDOW_GUI_NORMAL);
    imshow("Clustering", img);
}