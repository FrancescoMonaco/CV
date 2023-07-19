#include "region_growing.h"
#include "visualize_image.h"
#include <iostream>
#include <map>
#include<queue>
bool areMatsEqual(const cv::Mat& mat1, const cv::Mat& mat2) {
    if (mat1.size() != mat2.size() || mat1.type() != mat2.type()) {
        // Sizes or types are different, mats are not equal
        return false;
    }

    cv::Mat diff;
    cv::compare(mat1, mat2, diff,
        cv::CMP_NE); // Compare the two mats element-wise

    // Check if all elements are equal (zero non-zero elements in the difference
    // matrix)
    return cv::countNonZero(diff) == 0;
}

cv::Mat region_growing(std::vector<cv::Mat> J_images,
    std::vector<int> windowSizes, int& numRegions) {
    // for now consider only the first J_image
    cv::Mat J_image = J_images[0].clone();

    // Base case: start from the max scale (65x65 neighbourhood, first element of
    // windowSizes), consider the J_image as a single region and compute the
    // global threshold (mean + a*std_dev)
    int numberOfRegions = 1;

    //starting from 33 size window
    double minimum_size = 512;

    // this matrix will contain the label assigned to each pixel (for each
    // region). All pixels unassigned will have the max value for a double
    cv::Mat regions(J_image.rows, J_image.cols, CV_64FC1,
        cv::Scalar(std::numeric_limits<double>::max()));
    int label = 0;
    std::vector<double> lab;

    std::vector<double> finalLabels;

    for (int i = 0; i < windowSizes.size(); i++) {

        if (i != 0) {
            minimum_size = minimum_size / 4;
        }

        std::cout << "compute starting seed area" << std::endl;
        // modifies with the new regions Mat regions
        finalLabels = computeThresholdForEachRegion(J_images[i], regions, windowSizes[i], label, numberOfRegions, minimum_size, lab);
        std::cout << "end starting seed area" << std::endl;

        //visualize_image(regions, "seed area");

        std::cout << "start seed_area_growing\n";

        seed_area_growing(J_images[i], regions);
        std::cout << "end seed_area_growing\n";
    }

    complete_expansion(regions, J_images[J_images.size() - 1]);
    visualize_image(regions, "Regions at the end of region growing");


    std::sort(finalLabels.begin(), finalLabels.end());

    //assign the labels in order (excluding the max one, which needs to stay unchanged)
    for (int k = 0; k < finalLabels.size(); k++) {

        if (k == finalLabels.size() - 1) {
            for (int i = 0; i < regions.rows; i++) {
                for (int j = 0; j < regions.cols; j++) {
                    if (regions.at<double>(i, j) == std::numeric_limits<double>::max()) {
                        regions.at<double>(i, j) = k;
                    }
                }
            }
        }

        for (int i = 0; i < regions.rows; i++) {
            for (int j = 0; j < regions.cols; j++) {
                if (regions.at<double>(i, j) == finalLabels[k]) {
                    regions.at<double>(i, j) = k;
                }
            }
        }

        finalLabels[k] = k;
    }

    numRegions = numberOfRegions;

    return regions;
}

std::vector<double> computeThresholdForEachRegion(cv::Mat J_image, cv::Mat& regions,
    int windowSizes, int& label, int& numberOfRegions,
    int minimum_size, std::vector<double>& reg) {
    // base case when we have just one region
    if (numberOfRegions == 1) {
        // base case in which you determine the starting regions
        double mean = 0.0;
        double std_dev = 0.0;

        for (int i = 0; i < J_image.rows; i++) {
            for (int j = 0; j < J_image.cols; j++) {
                mean += J_image.at<double>(i, j);
            }
        }

        mean = mean / (J_image.cols * J_image.rows);

        for (int i = 0; i < J_image.rows; i++) {
            for (int j = 0; j < J_image.cols; j++) {
                std_dev += pow((J_image.at<double>(i, j) - mean), 2);
            }
        }

        std_dev = sqrt(std_dev / (J_image.cols * J_image.rows));

        // define the threshold using mean and std (the value a is an arbitrary
        // value that can be tuned) preset  a = [-0.6, -0.4, -0.2, 0, 0.2, 0.4]
        double a = -0.2;
        double threshold = mean + a * std_dev;
        std::cout << "threshold usato: " << threshold << std::endl;
        std::vector<cv::Point> starting_points;
        int counter = 0;

        // based on the threshold, find the starting points from which to start
        // region growing
        for (int i = 0; i < J_image.rows; i++) {
            for (int j = 0; j < J_image.cols; j++) {
                if (J_image.at<double>(i, j) < threshold) {
                    // add the starting point to the vector
                    starting_points.push_back(cv::Point(i, j));

                    regions.at<double>(i, j) = counter;
                    counter++;
                }
            }
        }

        cv::Mat mask(regions.size(), CV_8U, cv::Scalar(1));

        connect_starting_seeds(regions, starting_points, regions, std::numeric_limits<double>::max());

        //connect_starting_seeds(J_image, regions, starting_points, threshold, mask);
        // numberOfRegions = starting_points.size();
        // std::cout <<"number of region first iteration: "<<numberOfRegions <<
        // std::endl;
    }

    else {

        //used to understad what labels we can use
        int assigned_labels = label + 1;

        std::cout << "into the general case " << std::endl;
        // general case (for each region, the label is an int value from 0 to
        // starting_points.size() - 1)
        //std::cout << "number of regions: " << numberOfRegions << std::endl;
        cv::Mat updated_regions;//=regions.clone(); check if better


        for (int i = 0; i < reg.size(); i++) {

            //std::cout << "region number: " << i << std::endl;
            //clone the region calculated before

            cv::Mat current_regions = regions.clone();

            //set the threshold
            double region_mean = 0.0;
            double region_std_dev = 0.0;
            int regionSize = 0;

            //calculate the number of elements inside the region
            for (int k = 0; k < J_image.rows; k++) {
                for (int j = 0; j < J_image.cols; j++) {
                    if (regions.at<double>(k, j) == reg[i]) {
                        region_mean += J_image.at<double>(k, j);
                        regionSize++;
                    }
                }
            }

            if (regionSize != 0) {

                std::cout << "region number: " << i << "size image: " << regionSize << std::endl;
                region_mean = region_mean / regionSize;

                for (int k = 0; k < J_image.rows; k++) {
                    for (int j = 0; j < J_image.cols; j++) {
                        if (regions.at<double>(k, j) == reg[i])
                            region_std_dev += pow((J_image.at<double>(k, j) - region_mean), 2);
                    }
                }


                region_std_dev = sqrt(region_std_dev / regionSize);


                // define the threshold using mean and std (the value a is an arbitrary
                // value that can be tuned) preset  a = [-0.6, -0.4, -0.2, 0, 0.2, 0.4]
                double a = 0.4;
                double region_threshold = region_mean + a * region_std_dev;

                std::cout << "threshold: " << region_threshold << std::endl;
                //find the seed points of the region
                std::vector<cv::Point> region_starting_points;
                //int region_counter = 0;

                // based on the threshold, find the starting points from which to start
                // region growing
                int temp = assigned_labels;
                //std::cout << "max labels available till now: " << temp << std::endl;
                for (int k = 0; k < J_image.rows; k++) {
                    for (int j = 0; j < J_image.cols; j++) {
                        if (J_image.at<double>(k, j) < region_threshold && regions.at<double>(k, j) == reg[i]) {
                            // add the starting point to the vector
                            region_starting_points.push_back(cv::Point(k, j));

                            // note in the matrix that the starting point will belong to this
                            // region base case, so for now assign 1
                            current_regions.at<double>(k, j) = assigned_labels;
                            assigned_labels++;
                            //numberOfRegions * (k + 1) + region_counter;
                            //region_counter++;
                        }
                    }
                }

                //create the mask that has 1 for the pixels belonging to the region, 0 elsewhere
                /*cv::Mat mask = (current_regions == i);
                cv::Mat greyMask;
                mask.convertTo(greyMask, CV_8U);

                */
                connect_starting_seeds(current_regions, region_starting_points, regions, reg[i]);

                //visualize_image(current_regions, "Regione corrente");

                regions = current_regions;
            }
        }
    }

    // eliminate all regions that are not good
    std::map<double, std::vector<cv::Point>> pointsPerRegion;
    //indicates the maximum label used
    int max_label = 0;
    reg.clear();
    for (int i = 0; i < regions.rows; i++) {
        for (int j = 0; j < regions.cols; j++) {
            double num_reg = regions.at<double>(i, j);
            if (num_reg > max_label) {
                max_label = num_reg;
            }
            if (num_reg != std::numeric_limits<double>::max()) {
                pointsPerRegion[num_reg].push_back(cv::Point(i, j));

            }
        }
    }

    //for (const auto& pair : pointsPerRegion) {
      //  keys.push_back(pair.first);
    //}
    // temp

    for (const auto& myPair : pointsPerRegion) {
        // remove the region if its size isn't big enough
        if (myPair.second.size() < minimum_size) {

            // all the points which were inside the no good region need to become
            // unassigned again
            for (int i = 0; i < myPair.second.size(); i++) {
                regions.at<double>(myPair.second[i].x, myPair.second[i].y) =
                    std::numeric_limits<double>::max();
            }
        }
        else {
            reg.push_back(myPair.first);
        }
    }
    reg.push_back(std::numeric_limits<double>::max());

    std::cout << "end of elimination\n";
    std::cout << "numeber of regions left: " << reg.size() << std::endl;

    label = max_label;
    numberOfRegions = reg.size();

    visualize_image(regions, "starting seed areas");

    return reg;
}

void connect_starting_seeds(cv::Mat& currentRegions, std::vector<cv::Point> starting_points,
    cv::Mat prev_regions, double prev) {
    std::queue<cv::Point> points_queue;
    std::vector<cv::Point> neighbors = { cv::Point(-1, 0), cv::Point(1, 0), cv::Point(0, -1), cv::Point(0, 1) };
    cv::Mat visitedMatrix(currentRegions.rows, currentRegions.cols, CV_8UC1,
        cv::Scalar(0));
    std::cout << "starting seed number: " << starting_points.size() << std::endl;
    std::cout << "label :" << prev << std::endl;
    int iterations = 0;

    cv::Mat startingPointsMap(currentRegions.rows, currentRegions.cols, CV_8UC1, cv::Scalar(0));
    //fixing 
    for (int i = 0; i < starting_points.size(); i++) {
        startingPointsMap.at<uchar>(starting_points[i].x, starting_points[i].y) = 255;
    }

    while (!starting_points.empty()) {
        cv::Point current = starting_points.back();
        starting_points.pop_back();


        int row = current.x;
        int col = current.y;

        //visitedMatrix.at<uchar>(row, col) = 1;
        iterations++;
        double label = currentRegions.at<double>(row, col);
        //currentRegions.at<double>(row, col) = label;
        //if(prev !=std::numeric_limits<double>::max())

         //std::cout << label<<std::endl;
        points_queue.push(current);
        //visualize_image(prev_regions," regioni attuali");
        while (!points_queue.empty()) {
            cv::Point current_point = points_queue.front();

            points_queue.pop();

            //std::cout <<" label: " <<currentRegions.at<double>(current_point.x,current_point.y)<< std::endl;

            for (const cv::Point& neighbor : neighbors) {
                int new_row = current_point.x + neighbor.x;
                int new_col = current_point.y + neighbor.y;

                if (new_row >= 0 && new_row < currentRegions.rows && new_col >= 0 && new_col < currentRegions.cols) {

                    if (prev == std::numeric_limits<double>::max()) {
                        //auto it = std::find(starting_points.begin(), starting_points.end(), cv::Point(new_row, new_col));
                        if (visitedMatrix.at<uchar>(new_row, new_col) == 0 && startingPointsMap.at<uchar>(new_row, new_col) == 255) {
                            //std::cout << "entrato" << std::endl;
                            currentRegions.at<double>(new_row, new_col) = label;
                            points_queue.push(cv::Point(new_row, new_col));
                            /*if (prev != std::numeric_limits<double>::max()) {
                                std::cout << "points :" << new_row << " " << new_col << std::endl;
                            }*/
                            visitedMatrix.at<uchar>(new_row, new_col) = 1;

                        }
                    }
                    else {
                        //auto it = std::find(starting_points.begin(), starting_points.end(), cv::Point(new_row, new_col));
                        if (visitedMatrix.at<uchar>(new_row, new_col) == 0 && prev_regions.at<double>(new_row, new_col) == prev && startingPointsMap.at<uchar>(new_row, new_col) == 255) {
                            //std::cout << "entrato" << std::endl;
                            currentRegions.at<double>(new_row, new_col) = label;
                            points_queue.push(cv::Point(new_row, new_col));
                            /*if (prev != std::numeric_limits<double>::max()) {
                                std::cout << "points :" << new_row << " " << new_col << std::endl;
                            }*/
                            visitedMatrix.at<uchar>(new_row, new_col) = 1;

                        }


                    }

                }

            }
        }


    }
    //std:: cout<< "iterations" << iterations << std::endl;
}  //visualize_image(currentRegions, "end stariting seed points");


// grow all the points not already expanded

void seed_area_growing(cv::Mat Jimage, cv::Mat& regions) {
    cv::Mat original = regions.clone();
    std::vector<cv::Point> excluded_expansion;

    // calculate the average of unsegmented pixels
    double average = 0;
    int count = 0;

    for (int i = 0; i < regions.rows; i++) {
        for (int j = 0; j < regions.cols; j++) {
            if (regions.at<double>(i, j) == std::numeric_limits<double>::max()) {
                average += Jimage.at<double>(i, j);
                count++;
                excluded_expansion.push_back(cv::Point(i, j));
            }
        }
    }

    average = average / count;

    // check for all excluded pixels if their Jvalue is smaller than the local
    // average Jvalue
    cv::Mat binary_excluded(regions.size(), CV_8U, cv::Scalar(0));

    for (int i = 0; i < excluded_expansion.size(); i++) {
        if (Jimage.at<double>(excluded_expansion[i].x, excluded_expansion[i].y) < average)
            binary_excluded.at<uchar>(excluded_expansion[i].x, excluded_expansion[i].y) = 255;
        else
            binary_excluded.at<uchar>(excluded_expansion[i].x, excluded_expansion[i].y) = 127;
    }

    cv::Mat labels;

    // connect all the excluded points (labels contains all the labels assigned to
    // each pixel, label 0 is the background)
    int numLabels = cv::connectedComponents(binary_excluded, labels, 8, CV_32S);

    //std::cout << "number of labels: " << numLabels << std::endl;

    // compute the Laplacian to determine the borders
    cv::Mat labelsDisplay;
    labels.convertTo(labelsDisplay, CV_8U);
    // dealloc labels

    // label now on labels structure
    cv::Mat borders;

    cv::Laplacian(labelsDisplay, borders, CV_32S, 9, 1, 0, cv::BORDER_DEFAULT);
    // dealloc boarder
    cv::Mat bordersDisplay;
    borders.convertTo(bordersDisplay, CV_8U);

    /*cv::imshow("boarder image", bordersDisplay);
    cv::waitKey(0);
    */

    std::vector<bool> later(numLabels, true);
    std::vector<double> check(numLabels, -1.0);
    
    /*
    bool unchanged = false;
    int maxIterForUnchanged = 10000;
    int counter = 0;

    while (!excluded_expansion.empty()) {

        if (counter == maxIterForUnchanged)
            break;

        cv::Point current = excluded_expansion.back();
        excluded_expansion.pop_back();

        int row = current.x;
        int col = current.y;

        if (binary_excluded.at<uchar>(row, col) == 255) {
            std::map<double, int> neighbours;
            int maxNumber = 0;

            for (int i = -1; i < 2; i++) {
                for (int j = -1; j < 2; j++) {
                    // check if the current neighbour is in a valid position
                    if ((row + i >= 0 && row + i < regions.rows) && (col + j >= 0 && col + j < regions.cols)) {
                        if (binary_excluded.at<uchar>(row + i, col + j) == 0) {
                            maxNumber++;

                            neighbours[regions.at<double>(row + i, col + j)] = neighbours[regions.at<double>(row + i, col + j)] + 1;
                        }
                    }
                }
            }
  
            if (!neighbours.empty() && neighbours.begin()->second == maxNumber) {
                double label = neighbours.begin() -> first; // Get the label of the first neighbor (they all have the same label)
                regions.at<double>(row, col) = label;
            }
            else {
                excluded_expansion.insert(excluded_expansion.begin(), cv::Point(row, col));
                counter++;
            }
        }        
    }*/

    const int rows = labels.rows;
    const int cols = labels.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int numlab = labels.at<int>(i, j);

            // check if  label isn't the background
            if (numlab > 0) {
                // if the pixel was detected as part of a border
                if (bordersDisplay.at<uchar>(i, j) == 255) {
                    // then check its neighbours
                    for (int x = -1; x < 2; x++) {
                        for (int y = -1; y < 2; y++) {
                            if ((x + i >= 0 && x + i < bordersDisplay.rows) && (y + j >= 0 && y + j < bordersDisplay.cols)) {
                                // and exclude all the neighbors that aren't in a cross position

                                // if the neighbour doesn't belong to any region or it borders
                                // more than one, than skip it
                                if (regions.at<double>(i + x, j + y) == std::numeric_limits<double>::max()) {
                                    continue;
                                }
                                else {
                                    // if it doesn't belong to the border and
                                    if (later[numlab]) {
                                        // for the current pixel no label has been assigned yet
                                        if (check[numlab] == -1.0 &&
                                            regions.at<double>(i + x, j + y) !=
                                            std::numeric_limits<double>::max()) {
                                            check[numlab] = regions.at<double>(i + x, j + y);
                                        }
                                        // has the current pixel the same label as the neighbour? If
                                        // yes, continue
                                        else {
                                            if (check[numlab] == regions.at<double>(i + x, j + y))
                                                continue;
                                            else { // the label are different, so can't be merged
                                                later[numlab] = false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < regions.rows; i++) {
        for (int j = 0; j < regions.cols; j++) {

            int numlab = labels.at<int>(i, j);
            double new_lab = check[numlab];

            if (numlab > 0 && later[numlab] && new_lab != -1) {
                regions.at<double>(i, j) = new_lab;
            }
            else {
                if (numlab > 0 && !later[numlab] && new_lab != -1) {
                    regions.at<double>(i, j) = std::numeric_limits<double>::max();
                }
            }
        }
    }

    /*
    bool equal = areMatsEqual(original, regions);
    if (equal) {
        std::cout << "The mats are equal." << std::endl;
    }
    else {
        std::cout << "The mats are not equal." << std::endl;
    }*/

    visualize_image(regions, "result of region expansion");
}

void complete_expansion(cv::Mat& regions, const cv::Mat Jimage) {
    // save all the points which haven't been assigned yet
    std::vector<cv::Point> excluded_expansion;

    for (int i = 0; i < regions.rows; i++) {
        for (int j = 0; j < regions.cols; j++) {
            if (regions.at<double>(i, j) == std::numeric_limits<double>::max())
                excluded_expansion.push_back(cv::Point(i, j));
        }
    }

    // sort the excluded points based on their Jvalue
    std::sort(excluded_expansion.begin(), excluded_expansion.end(),
        [Jimage](const cv::Point& a, const cv::Point& b) {
            return Jimage.at<double>(a.x, a.y) < Jimage.at<double>(b.x, b.y);
        });


    while (!excluded_expansion.empty()) {

        cv::Point current = excluded_expansion.back();
        excluded_expansion.pop_back();

        int row = current.x;
        int col = current.y;

        std::map<double, int> neighbours;

        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                // check if the current neighbour is in a valid position
                if ((row + i >= 0 && row + i < regions.rows) &&
                    (col + j >= 0 && col + j < regions.cols)) {
                    // and exclude all the neighbors that aren't in a cross position
                    if (i - j == 0 || i + j == 0)
                        continue;
                    else {
                        // check that the neighbour hasn't been visited and assigned already
                        // to a region
                        if (regions.at<double>(row + i, col + j) != std::numeric_limits<double>::max()) {
                            // assign it to the same region
                            regions.at<double>(row, col) = regions.at<double>(row + i, col + j);
                            break;
                        }
                    }
                }
            }
        }

        if (regions.at<double>(row, col) == std::numeric_limits<double>::max())
            excluded_expansion.insert(excluded_expansion.begin(), cv::Point(row, col));
    }
    visualize_image(regions, "result of complete expansion");
}
