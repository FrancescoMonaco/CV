#include "processing.h"
using namespace std;
using namespace cv;

void calibrate(std::vector<cv::Mat> images)
{
	// Detection

	vector<vector<Point2f>> projections;
	for (auto& image : images) {
		Mat detect;
		cvtColor(image, detect, COLOR_BGR2GRAY);
		vector<Point2f> corners;
		bool patternfound = findChessboardCorners(detect, Size(6, 5), corners);
		if (patternfound)
			cornerSubPix(detect, corners, Size(6, 5), Size(-1, -1),
				TermCriteria(TermCriteria::Type::MAX_ITER, 30, 0.1));

		// Push the projections into the vector
		projections.push_back(corners);
		drawChessboardCorners(image, Size(6, 5), Mat(corners), patternfound);

		namedWindow("Out", WINDOW_GUI_EXPANDED);
		imshow("Out", image);
	}

	// Calibration
	vector<vector<Vec3f>> points;
}