
#include "visualize_image.h"

//ottimizzare
std::vector<double> visualize_image(cv::Mat regions, std::string name_window) {

    if (regions.type() != CV_64F) {
        regions.convertTo(regions, CV_64F);
    }

    std::vector<double> region_label;

    for (int i = 0; i < regions.rows; i++) {
        for (int j = 0; j < regions.cols; j++) {
            double temp = regions.at<double>(i, j);
            auto trovato = std::find(region_label.begin(), region_label.end(), temp);

            // non trovato
            if (trovato == region_label.end()) {
                region_label.push_back(temp);
            }
        }
    }

    std::vector<cv::Vec3b> colors;
    for (double label : region_label) {
        if (label == std::numeric_limits<double>::max()) {

            colors.push_back(cv::Vec3b(0, 0, 0));
        }
        else {
            int b = cv::theRNG().uniform(64, 256);
            int g = cv::theRNG().uniform(64, 256);
            int r = cv::theRNG().uniform(64, 256);
            colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
    }

    cv::Mat result(regions.rows, regions.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < regions.rows; i++) {
        for (int j = 0; j < regions.cols; j++) {
            double index = regions.at<double>(i, j);

            auto it = std::find(region_label.begin(), region_label.end(), index);

            if (it != region_label.end()) {
                int label_index = std::distance(region_label.begin(), it);
                result.at<cv::Vec3b>(i, j) = colors[label_index];
            }
        }
    }

    cv::imshow(name_window, result);
    cv::waitKey(0);
    cv::destroyWindow(name_window);

    return region_label;
}
