#ifndef VISUALIZE_IMAGE_H
#define VISUALIZE_IMAGE_H


#include<opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

std::vector<double> visualize_image(cv::Mat regions, std::string name_window);

#endif
