#include "getImageBoxes.h"
#include "opencv2/imgproc.hpp"

std::vector<cv::Rect> getImageBoxes(cv::Mat& image) {
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.5);
    std::vector<cv::Vec3f> circles;
    std::vector<cv::Rect> extractions;
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    cv::HoughCircles(grayscale, circles, cv::HOUGH_GRADIENT, 1, grayscale.rows / 2.5, 140, 55, 185, 370);
    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
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
        extractions.push_back(cv::Rect(x, y, width, height));

    }
    return extractions;
}

std::vector<cv::Rect> getRectanglesBoxes(cv::Mat& image) {
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.5);
    std::vector<cv::Vec3f> circles; std::vector<cv::Rect> extractions;
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    cv::HoughCircles(grayscale, circles, cv::HOUGH_GRADIENT, 1, grayscale.rows / 2.5, 140, 55, 185, 370);
    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        //find the rectangle that contains the circle
        int x = center.x - radius;
        int y = center.y - radius;
        int width = 2 * radius;
        int height = 2 * radius;

        // Adjust the rectangle for a smaller image by 0.5 on each axis
        int offsetX = cvRound(0.5 * width);
        int offsetY = cvRound(0.5 * height);
        x += offsetX;
        y += offsetY;
        width -= 2 * offsetX;
        height -= 2 * offsetY;


        int decimationFactor = 2; // Adjust the decimation factor as needed
        int decimatedX = x / decimationFactor;
        int decimatedY = y / decimationFactor;
        int decimatedWidth = width / decimationFactor;
        int decimatedHeight = height / decimationFactor;

        // Perform boundary checks and adjust the rectangle if necessary
        if (decimatedX < 0) {
            decimatedWidth += decimatedX;
            decimatedX = 0;
        }
        if (decimatedY < 0) {
            decimatedHeight += decimatedY;
            decimatedY = 0;
        }
        if (decimatedX + decimatedWidth > image.cols) {
            decimatedWidth = image.cols - decimatedX;
        }
        if (decimatedY + decimatedHeight > image.rows) {
            decimatedHeight = image.rows - decimatedY;
        }


        extractions.push_back(cv::Rect(decimatedX, decimatedY, decimatedWidth, decimatedHeight));

    }
    return extractions;
}