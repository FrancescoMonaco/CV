#include "processing.h"
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // check argc
        if (argc != 2) {
        std::cerr << "Usage: ./Project_Main <relative_path>\n";
        return 1;
    }

    int fileNum = 0;
    std::string relativePath = argv[1];
    

    // for each folder in the relative path
    for (const auto& entry : fs::directory_iterator(relativePath)) {
        // if entry is a folder
        if (entry.is_directory()) {
            // create the folder for the predictions
            fs::create_directory(entry.path().string() + PREDS_BB);

             //go in and work
            std::vector<cv::String> filenames;
            cv::glob(entry.path().string() + "/*.jpg", filenames, false);
            // save all images in a vector
            vector<cv::Mat> images;
            for (auto& str : filenames) {
                    Mat img = imread(str);
                    images.push_back(img);
                }
            // define some variables for the loop, shared info between images
            bool hasBread = false;
            int image_in_process = 0;
            vector<cv::Mat> recognizedFood;
            vector<int> recognizedFoodID;

            // for each image in the folder
            for (auto& image: images){
                // get name of the file
                string name = filenames[image_in_process].substr(filenames[image_in_process].find_last_of("/\\") + 1);
                GaussianBlur(image, image, Size(3, 3), 0.5);
                std::vector<Vec3f> circles;
                Mat grayscale;
                Mat mask = Mat::zeros(image.size(), CV_8UC1); // Mask initialization
                cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
                int rad = 0;
                HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, grayscale.rows/2.5, 140, 55, 185, 370);
                for (size_t i = 0; i < circles.size(); i++)
                {
                    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);
                    circle(mask, center, radius, Scalar(255), -1);
                    if (i == 0) rad = radius;
                }

                //BREAD PART
                Mat inverse_mask;
                bitwise_not(mask, inverse_mask);
                Mat rec_bread; Rect bread_box;
                image.copyTo(rec_bread, inverse_mask);

                if (image_in_process == 0) // for the first image check if there is bread, set hasBread for the future
                     bread_box = breadFinder(rec_bread, rad, true, &hasBread, relativePath);
                else if (hasBread)
                     bread_box = breadFinder(rec_bread, rad, false, &hasBread, relativePath);
                if(hasBread){
                    writeBoundBox(entry.path().string() + PREDS_BB + name + "_bouding_box.txt", bread_box, BREAD);
                }

                //FOR EACH PLATE
                //put the plates in a vector
                vector<Mat> plates;
                int smallestCircle = -1;
                for (size_t i = 0; i < circles.size(); i++){
                    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);
                    int x = center.x - radius;
                    int y = center.y - radius;
                    int width = 2 * radius;
                    int height = 2 * radius;
                    //find the smallest circle
                    if (smallestCircle == -1 || radius < smallestCircle) smallestCircle = radius;

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
                    plates.push_back(plate);
                }

                //The first time we need a recognition from 0
                if (image_in_process==0){
                    //for each plate
                    for (size_t i = 0; i < plates.size(); i++){
                        // if the corresponding circle radius is the smallest, check if it's salad
                        if (plates.size() > 2 && (cvRound(circles[i][2]) == smallestCircle)){
                            //check if it is salad
                            Rect salad_rect = matchSalad(plates[i], relativePath);
                            if (salad_rect != Rect(0, 0, 0, 0)){
                                //transform the rectangle to the original image
                                salad_rect.x += circles[i][0] - smallestCircle;
                                salad_rect.y += circles[i][1] - smallestCircle;
                                //write the bounding box
                                writeBoundBox(entry.path().string() + PREDS_BB + name + "_bouding_box.txt", salad_rect, SALAD);
                                recognizedFood.push_back(plates[i]);
                                recognizedFoodID.push_back(SALAD);
                                rectangle(image, salad_rect, Scalar(120, 200, 50), 2);
                            }
                            continue;
                        }
                        //check if it is the first or second dish
                        int dish = 1;//firstorSecond(plates[i]);
                        //if it is the first dish
                        if (dish == 1){
                            //classify the pasta
                            int pasta = pastaRecognition(plates[i]);
                            //pick the corresponding circle and find the biggest rectangle contained in it

                            Rect pasta_rect;
                            //create a rectangle with the same center of the circle and the same area
                            pasta_rect.x = 0.05 * plates[i].cols;
                            pasta_rect.y = 0.05 * plates[i].rows;
                            pasta_rect.width = plates[i].cols - 0.25 * plates[i].cols;
                            pasta_rect.height = plates[i].rows - 0.25 * plates[i].rows;
                            //transform the rectangle in the original image
                            pasta_rect.x += circles[i][0] - circles[i][2];
                            pasta_rect.y += circles[i][1] - circles[i][2];
                            //show it
                            rectangle(image, pasta_rect, Scalar(0, 255, 0), 2);
                            imshow("pasta", image);
                            waitKey(0);
                            //write the bounding box
                            writeBoundBox(entry.path().string() + PREDS_BB + name + "_bouding_box.txt", pasta_rect, pasta);
                            recognizedFood.push_back(plates[i]);
                            recognizedFoodID.push_back(pasta);
                        }
                        //if it is the second dish
                        else if (dish == 2){
                            //classify the second dish
                            int second_num, side_num;
                            Rect secondDish = secondDishClassifier(plates[i], relativePath, second_num);
                            // transform the rectangle in the original image
                            secondDish.x += circles[i][0] - circles[i][2];
                            secondDish.y += circles[i][1] - circles[i][2];
                            //write the bounding box
                            writeBoundBox(entry.path().string() + PREDS_BB + name + "_bouding_box.txt", secondDish, second_num);
                            recognizedFood.push_back(plates[i]);
                            recognizedFoodID.push_back(second_num);
                            cout << second_num << endl;

                            Rect sideDish = sideDishClassifier(plates[i], relativePath, side_num);
                            // transform the rectangle in the original image
                            sideDish.x += circles[i][0] - circles[i][2];
                            sideDish.y += circles[i][1] - circles[i][2];
                            //write the bounding box
                            writeBoundBox(entry.path().string() + PREDS_BB + name + "_bouding_box.txt", sideDish, side_num);
                            recognizedFood.push_back(plates[i]);
                            recognizedFoodID.push_back(side_num);

                            //put the rectangle and show image
                            rectangle(image, sideDish, Scalar(75, 125, 50), 2);
                            rectangle(image, secondDish, Scalar(20, 25, 70), 2);
                            imshow("Second-Side Dish", image);
                            waitKey(0);
                        }
                    }
                }
                //Subsequently we just match what we found with the new ones
                else {
                    //for(size_t i = 0; i < plates.size(); i++){
                    //   // check each plate with the ones we already recognized
                    //    int curr_index = 0;
                    //   Rect plate_rect = findNewPosition(plates[i], recognizedFood, curr_index);
                    //   // transform the rectangle in the original image
                    //    plate_rect.x += circles[i][0] - circles[i][2];
                    //    plate_rect.y += circles[i][1] - circles[i][2];
                    //    rectangle(image, plate_rect, Scalar(0, 254, 32), 2);
                    //    imshow("Sec", image);
                    //   writeBoundBox(entry.path().string() + PREDS_BB + name + "_bouding_box.txt", plate_rect, recognizedFoodID[curr_index]);
                    //   // Replace the recognized in the same place, to better classify the subsequent
                   // recognizedFood[curr_index] = plates[i];
                    //   waitKey(0);
                    //}
                }
                
                //increase image_in_process
                image_in_process++;
            }
        }
    }
    return 0;
}

