#include <opencv2/opencv.hpp>

void main()
{
    Mat img = imread("image.png");
    imshow("image",img);
    waitKey(0);
}