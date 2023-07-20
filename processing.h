#ifndef PROCESSING
#define PROCESSING
#include "Constants.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


/// @brief Finds the bread in the image and returns a rectangle that contains it
/// @param image , image where to find the bread 
/// @param radius , radius of a circle to dinamically choose the rectangle size
/// @param check , if true it will check if the bread is present in the image
/// @param hasBread , pointer to a boolean that will be true if the bread is present in the image
/// @param RelPath , relative path to the folder where the templates are stored
/// @return a rectangle that contains the bread
cv::Rect breadFinder(cv::Mat& image, int radius, bool check, bool* hasBread, const std::string RelPath);

/// @brief Finds the side dish in the image and returns a rectangle that contains it
/// @param in1 , image where to find the side dish
/// @param relativePath , relative path to the folder where the templates are stored
/// @param ID , pointer to an integer that will be the ID of the side dish
/// @return a rectangle that contains the side dish
cv::Rect sideDishClassifier(cv::Mat& in1, const std::string& relativePath, int& ID);


/// @brief Finds the second dish in the image and returns a rectangle that contains it
/// @param in1 , image where to find the second dish
/// @param relativePath , relative path to the folder where the templates are stored
/// @param ID , pointer to an integer that will be the ID of the second dish
/// @return a rectangle that contains the second dish
cv::Rect secondDishClassifier(cv::Mat& in1, const std::string& relativePath, int& ID);


/// @brief Given a box finds if it is a First Dish or a Second Dish
/// @param firstCircle , segmentation of the image that contains the first circle
/// @return 1 if first circle is a First Dish, 2 if first circle is a Second Dish
int firstorSecond(cv::Mat firstCircle);

/// @brief Writes the bounding box of the dish in a file
/// @param path , path of the file where to write the bounding box
/// @param box , bounding box of the dish
/// @param ID , ID of the dish
void writeBoundBox(const std::string& path, cv::Rect box, int ID);

/// @brief Classifies the pasta in the image and returns its ID
/// @param image , image where to classify the pasta
/// @return the ID of the pasta
int pastaRecognition(cv::Mat& image);

/// @brief Matches the salad in the image and returns a rectangle that contains it
/// @param image , image where to find the salad
/// @param relativePath , relative path to the folder where the templates are stored
/// @return a rectangle that contains the salad
cv::Rect matchSalad(cv::Mat& image, const std::string& relativePath);

/// @brief Using ORB matches the original with the leftovers
/// @param original , image where to find the leftovers
/// @param fit , vector of images that contains the classifed before
/// @param MatchID , pointer to an integer that will be the ID of the leftovers
/// @return a rectangle that contains the leftovers
cv::Rect findNewPosition(cv::Mat original, std::vector<cv::Mat> fit, int& MatchID);


/// @brief Writes the segmentation map of the dish in a file
/// @param local , local segmentation map
/// @param final , final segmentation map
/// @param where , rectangle where to put the label
/// @param ID , ID of the dish
void labelSegmentation(cv::Mat& local, cv::Mat& final, cv::Rect where,int ID);

#endif
