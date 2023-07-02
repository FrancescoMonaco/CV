#include "processing.h"
#include <iostream>
using namespace std;
using namespace cv;

void calibrate(std::vector<cv::Mat> images)
{
	// Detection

	vector<vector<Point2f>> projections;
	for (auto& image : images) {
		Mat detect;
		cvtColor(image, detect, COLOR_BGR2GRAY);
		vector<Point2f> corners;
		bool patternfound = findChessboardCorners(detect, Size(6, 5), corners);
		if (patternfound)
			cornerSubPix(detect, corners, Size(6, 5), Size(-1, -1),
				TermCriteria(TermCriteria::Type::MAX_ITER, 30, 0.1));

		// Push the projections into the vector
		projections.push_back(corners);
		drawChessboardCorners(image, Size(6, 5), Mat(corners), patternfound);

		namedWindow("Out", WINDOW_GUI_EXPANDED);
		imshow("Out", image);
	}

	// Calibration
	vector<vector<Vec3f>> points;
}

void breadFinder(std::vector<cv::Mat> images, std::vector<cv::Mat> bounding_boxes, std::vector<cv::Mat> segments)
{
    
    for (auto& image : images) {
        breadBox(image);
    }
}

cv::Mat breadBox(cv::Mat image)
{
    Mat bread_template = imread("/Users/franc/Downloads/bread_template.jpg");

    Mat mask2, mask3, bread_image, bread_image2, boxed;

    // Go into HSV colorplane
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    std::vector<Mat> hsv_channels;
    split(hsv_image, hsv_channels);
    Mat hue_channel = hsv_channels[0];
    Mat sat_channel = hsv_channels[1];
    Mat value_channel = hsv_channels[2];

    // Range over the browns
    inRange(sat_channel, 40, 170, mask3);
    inRange(hue_channel, 5, 75, mask2);

    // Mask the image leaving only the bread and some tray
    Mat matchOut;
    bitwise_and(image, image, bread_image, mask2);
    bitwise_and(bread_image, bread_image, bread_image2, mask3);
    imshow("Hue", bread_image2);


    matchTemplate(bread_image2, bread_template, matchOut, TM_SQDIFF);

    //imshow("Output", matchOut);
    double minVal = {}, maxVal = {};
    Point minLoc, maxLoc;

    minMaxLoc(matchOut, &minVal, &maxVal, &minLoc, &maxLoc);
    Point end = Point((minLoc).x + bread_template.cols, (minLoc).y + bread_template.rows);
    rectangle(boxed, minLoc, end, Scalar(0, 0, 255));
    imshow("Rec", boxed);
    waitKey(0);
    return boxed;
}
