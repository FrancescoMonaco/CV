#include "processing.h"
using namespace std;
using namespace cv;

void apply_ORB(cv::Mat& templ, cv::Mat& image, bool drawBox)
{
	// Variables initialization
	int tresh = 10;
	bool sim = false;
	double ratio = 0.77;
	Mat out, out_match;
	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> vec, vec2;

	// Find the keypoints in the two images
	(*orb).detect(templ, vec);
	(*orb).detect(image, vec2);

	// Show them
	drawKeypoints(templ, vec, out);
	//namedWindow("ORB Keypoints", WINDOW_GUI_NORMAL);
	//imshow("ORB Keypoints", out);

	Mat descriptors1, descriptors2;
	orb->compute(templ, vec, descriptors1);
	orb->compute(image, vec2, descriptors2);

	// Create a brute force matcher and match the features
	Ptr<BFMatcher> matcher = BFMatcher::create();
	vector<vector<DMatch>> matches;
	vector<vector<DMatch>> pruned_matches;

	matcher->knnMatch(descriptors1, descriptors2, matches, 2);

	for (int i = 0; i < matches.size(); i++) {
		// check the similar matches
		if (matches[i][0].distance < ratio * matches[i][1].distance) {
			pruned_matches.push_back(matches[i]);
		}
		if (pruned_matches.size() == 10) break;
	}

	// show the matches
	drawMatches(templ, vec, image, vec2, pruned_matches, out_match);
	namedWindow("ORB Matches", WINDOW_GUI_NORMAL);
	imshow("ORB Matches", out_match);



	
}