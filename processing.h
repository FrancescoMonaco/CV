#ifndef PROCESSING
#define PROCESSING
#include "Constants.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


cv::Rect breadFinder(cv::Mat& image, int radius, bool check, bool* hasBread, const std::string RelPath);
cv::Rect sideDishClassifier(cv::Mat& in1,const std::string& relativePath, int& ID);
cv::Rect secondDishClassifier(cv::Mat& in1, const std::string& relativePath, int& ID);
int firstorSecond(cv::Mat& image);
void writeBoundBox(const std::string& path, cv::Rect box, int ID);
int pastaRecognition(cv::Mat& image);
cv::Rect matchSalad(cv::Mat& image, const std::string& relativePath);
cv::Rect findNewPosition(cv::Mat original, std::vector<cv::Mat> fit, int& MatchID);

#endif
