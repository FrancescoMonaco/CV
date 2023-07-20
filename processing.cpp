#include "processing.h"
#include "visualize_image.h"
using namespace std;
using namespace cv;



Rect breadFinder(Mat& result, int radius, bool check, bool* hasBread, const std::string RelPath) {
    Mat hsv_image, mask2, mask3, bread_image2;

    // Load the templates and check for their existence
    Mat bread_template = imread(RelPath + "/pan_br.jpeg");
    Mat crumbs_template = imread(RelPath + "/pan_left.jpg");

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
        return rec; // Return an empty rectangle
    }

    // Create the bounding box
    if (minVal2 < lowerBound) {
        int offsetX = bread_template.cols / 2;  // Half the width of the template
        int offsetY = bread_template.rows / 2;  // Half the height of the template
        Point center(minLoc2.x + offsetX, minLoc2.y + offsetY);  // Calculate the center position

        // Define the top-left and bottom-right corners of the rectangle
        Point topLeft(center.x - offsetX, center.y - offsetY);
        Point bottomRight(center.x + offsetX, center.y + offsetY);
        //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
        //imshow("Rec", result);
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
        //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
        //imshow("Rec", result);
        rec = Rect(topLeft, bottomRight);
    }
    else if (radius > bigRadius) {
        int offsetX = bread_template.cols / 2;  // Half the width of the template
        int offsetY = bread_template.rows / 2;  // Half the height of the template
        Point center(minLoc.x + offsetX, minLoc.y + offsetY);  // Calculate the center position

        // Define the top-left and bottom-right corners of the rectangle
        // as a bigger box due to perspective
        Point topLeft(center.x - 1.5 * offsetX, center.y - 1.5 * offsetY);
        Point bottomRight(center.x + 1.5 * offsetX, center.y + 1.5 * offsetY);
        //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
        //imshow("Rec", result);
        rec = Rect(topLeft, bottomRight);
    }
    else {
        int offsetX = bread_template.cols / 2;  // Half the width of the template
        int offsetY = bread_template.rows / 2;  // Half the height of the template
        Point center(minLoc.x + offsetX, minLoc.y + offsetY);  // Calculate the center position

        // Define the top-left and bottom-right corners of the rectangle
        // as a slightly bigger box due to perspective
        Point topLeft(center.x - 0.4 * offsetX, center.y - 1.5 * offsetY);
        Point bottomRight(center.x + 1.1 * offsetX, center.y + 1.2 * offsetY);
        //rectangle(result, topLeft, bottomRight, Scalar(0, 0, 255));
        //imshow("Rec", result);
        rec = Rect(topLeft, bottomRight);
    }
    *hasBread = true;
    return rec;
}

/// @brief Extract the plates using the Hough Transform
/// @param image, image to be processed
/// @return a vecor of the extracted images
std::vector<cv::Mat> getImageBoxes(cv::Mat& image) {
    GaussianBlur(image, image, cv::Size(3, 3), 0.5);
    std::vector<cv::Vec3f> circles; std::vector<cv::Mat> extractions;
    cv::Mat grayscale;
    cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, grayscale.rows / 2.5, 140, 55, 185, 370);
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

Rect matchSalad(Mat& image, const std::string& relativePath) {
    Mat salad_template = imread(relativePath + "/insalata.png");

    if (salad_template.data == NULL)
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
    // Check the threshold
    if (minVal > SALAD_THRESH)
        return Rect(0, 0, 0, 0);

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
    return rec;
}

int firstorSecond(cv::Mat firstCircle)
{
    //put fisrtCircle in grayscale
    cv::Mat firstCircleGray;
    cv::cvtColor(firstCircle, firstCircleGray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(firstCircleGray, circles, cv::HOUGH_GRADIENT, 1, firstCircleGray.rows / 2.5, 140, 55, 75, 185);
    //if there are no circles in the firstCircle, it means that it is the second dish
    if (circles.size() == 0)
        return 2;
    else
        return 1;
}


void writeBoundBox(const std::string& path, Rect box, int ID)
{
    //Open the file to write or create it if it doesn't exist
    ofstream file(path, ios::app);
    if (!file.is_open()) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
    //Write in format ID:id; [x, y, w, h]
    file << "ID:" << ID << "; [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]" << endl;
}

cv::Rect sideDishClassifier(cv::Mat& in1, const std::string& relativePath, int& ID)
{
    // Load the two templates
    Mat potato_template = imread(relativePath + "/patate.jpg");
    Mat beans_template = imread(relativePath + "/fagioli.jpg");
    Mat gray_potato, gray_bean, gray_in;
    cv::cvtColor(in1, gray_in, cv::COLOR_BGR2GRAY);
    cv::cvtColor(potato_template, gray_potato, cv::COLOR_BGR2GRAY);
    cv::cvtColor(beans_template, gray_bean, cv::COLOR_BGR2GRAY);
    //Compute the edges for the two templates and the image
    Mat edges_potato, edges_bean, edges_in;
    Canny(gray_potato, edges_potato, 50, 200, 3);
    Canny(gray_bean, edges_bean, 50, 200, 3);
    Canny(gray_in, edges_in, 50, 200, 3);
    //Use Moments to find the Hu moments of the two templates and the image
    Moments m1 = moments(edges_potato);
    Moments m2 = moments(edges_bean);
    Moments m3 = moments(edges_in);
    //Calculate the Hu moments of the two templates and the image
    Mat hu1, hu2, hu3;
    HuMoments(m1, hu1);
    HuMoments(m2, hu2);
    HuMoments(m3, hu3);
    //Calculate the distance between the Hu moments of the image and the two templates
    double dist1 = matchShapes(hu1, hu3, CONTOURS_MATCH_I1, 0);
    double dist2 = matchShapes(hu2, hu3, CONTOURS_MATCH_I1, 0);
    // Find the max and use that template to locate the side dish
    Mat template_to_use;
    if (dist1 > dist2) {
        template_to_use = edges_potato;
        ID = POTATOES;
    }
    else {
        template_to_use = edges_bean;
        ID = BEANS;
    }
    // Cut the edges image by 1/2 to avoid the border
    edges_in = edges_in(Rect(0, edges_in.rows / 2, edges_in.cols, edges_in.rows / 2));

    // Apply template matching with the two images, find a rect and return it
    Mat result;
    matchTemplate(edges_in, template_to_use, result, TM_SQDIFF);
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // Create a rect in the corner of the image
    int x = 0.05 * in1.cols;
    int y = 0.05 * in1.rows;
    int width = in1.cols - 0.25 * in1.cols;
    int height = in1.rows - 0.25 * in1.rows;
    // Move a bit to the minimum point
    x += 0.05 * minLoc.x;
    y += 0.05 * minLoc.y;

    if (x < 0) {
        width += x;
        x = 0;
    }
    if (y < 0) {
        height += y;
        y = 0;
    }
    if (x + width > in1.cols) {
        width = in1.cols - x;
    }
    if (y + height > in1.rows) {
        height = in1.rows - y;
    }

    Rect rec(x, y, width, height);
    return rec;
}

cv::Rect secondDishClassifier(cv::Mat& in1, const std::string& relativePath, int& ID)
{
    // Load the four templates
    Mat rabbit_template = imread(relativePath + "/rabbit.png");
    Mat pork_template = imread(relativePath + "/pork.png");
    Mat fish_template = imread(relativePath + "/fish.png");
    Mat sea_template = imread(relativePath + "/sea.png");

    // convert all to HSV
    Mat hsv_rabbit, hsv_pork, hsv_fish, hsv_sea, hsv_in;
    cv::cvtColor(rabbit_template, hsv_rabbit, cv::COLOR_BGR2HSV);
    cv::cvtColor(pork_template, hsv_pork, cv::COLOR_BGR2HSV);
    cv::cvtColor(fish_template, hsv_fish, cv::COLOR_BGR2HSV);
    cv::cvtColor(sea_template, hsv_sea, cv::COLOR_BGR2HSV);
    cv::cvtColor(in1, hsv_in, cv::COLOR_BGR2HSV);

    // Compute the histograms of the templates and the image
    cv::Mat hist_rabbit, hist_pork, hist_fish, hist_sea, hist_in;
    int histSize[] = { 8, 8, 8 };  // Number of bins for each channel
    float hRange[] = { 0, 180 };   // Hue range
    float sRange[] = { 0, 256 };   // Saturation range
    float vRange[] = { 0, 256 };   // Value range
    const float* ranges[] = { hRange, sRange, vRange };
    int channels[] = { 0, 1, 2 };

    calcHist(&hsv_rabbit, 1, channels, Mat(), hist_rabbit, 3, histSize, ranges, true, false);
    calcHist(&hsv_pork, 1, channels, Mat(), hist_pork, 3, histSize, ranges, true, false);
    calcHist(&hsv_fish, 1, channels, Mat(), hist_fish, 3, histSize, ranges, true, false);
    calcHist(&hsv_sea, 1, channels, Mat(), hist_sea, 3, histSize, ranges, true, false);
    calcHist(&hsv_in, 1, channels, Mat(), hist_in, 3, histSize, ranges, true, false);

    // Normalize the histograms
    normalize(hist_rabbit, hist_rabbit, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(hist_pork, hist_pork, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(hist_fish, hist_fish, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(hist_sea, hist_sea, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(hist_in, hist_in, 0, 1, NORM_MINMAX, -1, Mat());

    // Compute histogram intersection between the image and the templates
    double rabbit_score = compareHist(hist_rabbit, hist_in, HISTCMP_INTERSECT);
    double pork_score = compareHist(hist_pork, hist_in, HISTCMP_INTERSECT);
    double fish_score = compareHist(hist_fish, hist_in, HISTCMP_INTERSECT);
    double sea_score = compareHist(hist_sea, hist_in, HISTCMP_INTERSECT);

    // Find the max and use that template to locate the second dish
    Mat template_to_use;
    if (rabbit_score > pork_score && rabbit_score > fish_score && rabbit_score > sea_score) {
        template_to_use = rabbit_template;
        ID = RABBIT;
    }
    else if (pork_score > rabbit_score && pork_score > fish_score && pork_score > sea_score) {
        template_to_use = pork_template;
        ID = PORK;
    }
    else if (fish_score > rabbit_score && fish_score > pork_score && fish_score > sea_score) {
        template_to_use = fish_template;
        ID = FISH;
    }
    else {
        template_to_use = sea_template;
        ID = SEAFOOD;
    }
    // Resize the template if it is too big
    if (template_to_use.cols > in1.cols / 2) {
        double scale = (double)in1.cols / (2 * template_to_use.cols);
        resize(template_to_use, template_to_use, Size(), scale, scale);
    }
    // Use template matching to find the location of the second dish
    Mat result;
    matchTemplate(in1, template_to_use, result, TM_SQDIFF_NORMED);
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    int x = minLoc.x;
    int y = minLoc.y;
    int width = template_to_use.cols;
    int height = template_to_use.rows;
    //Make the rectangle 1.5 times bigger
    x -= width / 4;
    y -= height / 4;
    width += width / 2;
    height += height / 2;


    // Check if the template is too close to the border
    if (x < 0) {
        width += x;
        x = 0;
    }
    if (y < 0) {
        height += y;
        y = 0;
    }
    if (x + width > in1.cols) {
        width = in1.cols - x;
    }
    if (y + height > in1.rows) {
        height = in1.rows - y;
    }

    // Return the rectangle
    return cv::Rect(x, y, width, height);
}

int pastaRecognition(Mat& image) {
    /*
    * The thresholds for the recognition are used as centroids based on the
    * assumption that the plates appear similarly
    * (e.g. rice will always have peas, pasta will always have a ring without sauce)
    */
    const std::vector<vector<int>> centers = { {3000,8000,17000,24000},\
                                             {0, 4000, 50000, 16000},\
                                             {0,4000,30000,26000},\
                                             {0,460,19000,26666},\
                                             {100, 9000, 21000, 23000} };

    // Use only the center of the image, to have more uniform regions between different samples
    Mat center = image(Rect(image.cols / 4, image.rows / 4, image.cols / 2, image.rows / 2));

    // Use HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    // Extract the number of pixels in the green, yellow, red, orange, brown and white ranges
    int green = 0, yellow = 0, red = 0, orange = 0, brown = 0, white = 0;
    for (int i = 0; i < center.rows; i++) {
        for (int j = 0; j < center.cols; j++) {
            Vec3b pixel = hsv.at<Vec3b>(i, j);
            if (pixel[0] >= 40 && pixel[0] <= 80 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                green++;
            else if (pixel[0] >= 20 && pixel[0] <= 40 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                yellow++;
            else if (pixel[0] >= 0 && pixel[0] <= 20 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                red++;
            else if (pixel[0] >= 10 && pixel[0] <= 20 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                orange++;
            else if (pixel[0] >= 0 && pixel[0] <= 10 && pixel[1] >= 50 && pixel[1] <= 255 && pixel[2] >= 50 && pixel[2] <= 255)
                brown++;
            else if (pixel[0] >= 0 && pixel[0] <= 180 && pixel[1] >= 0 && pixel[1] <= 50 && pixel[2] >= 200 && pixel[2] <= 255)
                white++;
        }
    }

    //Using green, yellow, red, white find the closest center using sum of squares
    int min = 1000000000;
    int index = -1;
    for (int i = 0; i < centers.size(); i++) {
        int dist = 0;
        dist += pow(green - centers[i][0], 2);
        dist += pow(yellow - centers[i][1], 2);
        dist += pow(red - centers[i][2], 2);
        dist += pow(white - centers[i][3], 2);
        if (dist < min) {
            min = dist;
            index = i;
        }
    }
    // 0 is the background, so we move up the indices
    index += 1;
    return index;
}

cv::Rect findNewPosition(cv::Mat original, std::vector<cv::Mat> fit, int& MatchID)
{
    // Mask the white in all images
    for (int i = 0; i < fit.size(); i++) {
        cv::Mat mask;
        cv::inRange(fit[i], cv::Scalar(180, 180, 180), cv::Scalar(255, 255, 255), mask);
        fit[i].setTo(cv::Scalar(0, 0, 0), mask);
    }
    cv::Mat mask;
    cv::inRange(original, cv::Scalar(180, 180, 180), cv::Scalar(255, 255, 255), mask);
    original.setTo(cv::Scalar(0, 0, 0), mask);
    // Cut the fit images by a little bit to remove the background
    std::vector<cv::Mat> fit2;
    for (int i = 0; i < fit.size(); i++) {
        fit2.push_back(fit[i](cv::Rect(fit[i].cols / 6, fit[i].rows / 6, fit[i].cols / 1.2, fit[i].rows / 1.2)));
    }
    // For each image in fit find the keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    (*orb).detectAndCompute(original, cv::Mat(), keypoints1, descriptors1);
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    // find the best match for original in fit and return the rectangle
    int best = -1;
    int best_matches = 0;
    for (int i = 0; i < fit.size(); i++) {
        // use the center of the image to find the best match (to remove the background)
        cv::Mat center = fit[i](cv::Rect(fit[i].cols / 4, fit[i].rows / 4, fit[i].cols / 2, fit[i].rows / 2));
        (*orb).detectAndCompute(center, cv::Mat(), keypoints2, descriptors2);
        matcher.match(descriptors1, descriptors2, matches);
        if (matches.size() > best_matches) {
            best_matches = matches.size();
            best = i;
            MatchID = i;
        }
    }
    // If no match is found return an empty rectangle
    if (best != -1) {
        // Use the best match to do template matching and find the new bounding box
        cv::Mat result;
        cv::matchTemplate(original, fit2[best], result, cv::TM_SQDIFF_NORMED);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        cv::Rect rect = cv::Rect(minLoc.x, minLoc.y, fit2[best].cols, fit2[best].rows);

        // Check if the bounding box is inside the image
        if (rect.x < 0) {
            rect.width += rect.x;
            rect.x = 0;
        }
        if (rect.y < 0) {
            rect.height += rect.y;
            rect.y = 0;
        }
        if (rect.x + rect.width > original.cols) {
            rect.width = original.cols - rect.x;
        }
        if (rect.y + rect.height > original.rows) {
            rect.height = original.rows - rect.y;
        }

        return rect;

    }
    else {
        return cv::Rect();
    }
}

void labelSegmentation(cv::Mat& local, cv::Mat& final, cv::Rect where,int ID){
    //  take local and extract the Rect where
    cv::Mat local2 = local(where);

    // find the most common number in local2
    int histSize[] = {256}; // Number of bins 
    float range[] = {0, 256}; // Range of pixel values
    const float* histRange[] = {range};
    cv::Mat hist;
    cv::calcHist(&local2, 1, 0, cv::Mat(), hist, 1, histSize, histRange);

    // Find the bin with the most common label
    int most_common_label = 0;
    int max_count = 0;
    for (int i = 0; i < hist.rows; ++i) {
        int count = hist.at<float>(i);
        if (count > max_count) {
            max_count = count;
            most_common_label = i;
        }
    }

    // Substitute the most common label with ID, set the rest to 0 and copy to final into the correct position
    for (int i = 0; i < local2.rows; i++) {
        for (int j = 0; j < local2.cols; j++) {
            if (local2.at<uchar>(i, j) == most_common_label) {
                local2.at<uchar>(i, j) = ID;
            }
            else {
                local2.at<uchar>(i, j) = 0;
            }
        }
    }
    local2.copyTo(final(where));

}