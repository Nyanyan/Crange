#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv) {
	Mat img = imread("./image.png", -1);
	if (img.empty()) return -1;
	namedWindow("ex", WINDOW_AUTOSIZE);
	imshow("ex", img);
	waitKey(0);
	destroyWindow("ex");
	return 0;
}