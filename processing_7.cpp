#include "processing_7.h"
using namespace std;
using namespace cv;

static const double hFoV = 33;

vector<KeyPoint> compute_SIFT(cv::Mat& in1)
{
	// Variables initialization
	int count = 0;
	int tresh = 10;
	bool sim = false;
	double ratio = 0.3;
	Mat out, out_match;
	Ptr<SIFT> sift = SIFT::create(0, 3, 0.04, 10, 1.6);
	vector<KeyPoint> vec;

	// Find the keypoints in the two images
	(*sift).detect(in1, vec);
	return vec;
}

void stitch(std::vector<cv::Mat> in)
{
	// Transform each image into it's cylindrical version
	vector<Mat> cyl;
	for (auto& img : in) {
		cyl.push_back(cylindricalProj(img, hFoV));
	}

	// Find the keypoints
	vector<vector<KeyPoint>> keys;
	for (auto& img : cyl) {
		keys.push_back(compute_SIFT(img));
	}

	//Mat out;
	//drawKeypoints(cyl[0], keys[0], out);
	//cv::imshow("Keys", out);

	// Find the matches

	matcher(cyl,keys);

}

void matcher(std::vector<Mat> images, std::vector<vector<cv::KeyPoint>> vec)
{
	Ptr<SIFT> sift = SIFT::create(0, 3, 0.04, 10, 1.6);
	double ratio = 2;
	Size sz_img = Size(images[0].rows* 4, images[0].cols * images.size());
	Mat out = Mat::zeros(sz_img, images[0].type());
	int oldmx = 0, oldmy = 0;

	for (int i = 0; i < images.size() - 1; i++) {
			vector<cv::KeyPoint> vec1, vec2;
			Mat img1, img2;
			Mat descriptors1, descriptors2;
			img1 = images[i];
			img2 = images[i + 1];
			vec1 = vec[i];
			vec2 = vec[i + 1];
		sift->compute(img1, vec1, descriptors1);
		sift->compute(img2, vec2, descriptors2);

		// Create a brute force matcher and match the features
		Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
		vector<DMatch> matches;
		vector<DMatch> pruned_matches, pruned_matches2;

		matcher->match(descriptors1, descriptors2, matches);
		float min = {};

		// We need to use the MIN DISTANCE
		for (int j = 0; j < matches.size(); j++) {

			if (j == 0)
				min = matches[j].distance;
			else if (min > matches[j].distance)
				min = matches[j].distance;
		}
		// We prune the matches to remove the outliers
		for (int j = 0; j < matches.size(); j++) {
			if (matches[j].distance < ratio * min)
				pruned_matches.push_back(matches[j]);
		}


		//Mat img_matches;
		//cv::drawMatches(img1, vec1, img2, vec2, pruned_matches, img_matches);

		//-- Localize the object
		vector<Point2f> obj;
		vector<Point2f> scene;
		for (size_t j = 0; j < pruned_matches.size(); j++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(vec1[pruned_matches[j].queryIdx].pt);
			scene.push_back(vec2[pruned_matches[j].trainIdx].pt);
		}
		//Mat mask;
		Mat H = findHomography(obj, scene, RANSAC);
	
		int meanx, meany = {};
		for (size_t j = 0; j < obj.size() && j < scene.size(); j++) {
				meanx = abs(obj[j].y - scene[j].y);
				meany = abs(obj[j].x - scene[j].x);
		}
		meanx = meanx / obj.size();
		meany = meany / obj.size();

		if (i == 0)// First image
			img1.copyTo(out.operator()(Range(0, img1.rows), Range(0, img1.cols)));

		//warpPerspective(img1, out, H, sz_img);
		img2.copyTo(out.operator()(Range(oldmx+meanx, img2.rows), Range(oldmy+meany, oldmy + img2.cols + meany)));
		oldmx += meanx;
		oldmy += meany;
		

	}
	cv::namedWindow("Stitched", WINDOW_KEEPRATIO);
	cv::imshow("Stitched", out);
	imwrite("test.jpg", out);
	return;
}

