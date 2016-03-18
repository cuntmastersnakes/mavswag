// Approach2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

/// Global variables

Mat src_gray,image,blurred;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
//char* window_name = "Edge Map";

/**
* @function CannyThreshold
* @brief Trackbar callback - Canny thresholds input with a ratio 1:3
*/

void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	bilateralFilter(src_gray, detected_edges, 5, 100, 100);
	//blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	//image.copyTo(dst, detected_edges);
	imshow("Edge Detection Demo", detected_edges);
}


string hot_keys =
"\n\nHot keys: \n"
"\tESC - quit the program\n"
"\tc - stop the tracking\n"
"\tb - switch to/from backprojection view\n"
"\th - show/hide object histogram\n"
"\tp - pause video\n"
"To initialize tracking, select the object with mouse\n";

static void help()
{
	cout << "\nThis is a demo that shows mean-shift based tracking\n"
		"You select a color objects such as your face and it tracks it.\n"
		"This reads from video camera (0 by default, or the camera number the user enters\n"
		"Usage: \n"
		"   ./camshiftdemo [camera number]\n";
	cout << hot_keys;
}

const char* keys =
{
	"{help h | | show help message}{@camera_number| 0 | camera number}"
};

int main(int argc, const char** argv)
{
	VideoCapture cap;
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	int camNum = parser.get<int>(0);
	cap.open(camNum);

	if (!cap.isOpened())
	{
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}
	cout << hot_keys;
	namedWindow("Edge Detection Demo", 0);
	createTrackbar("Min threshold:", "Edge Detection Demo", &lowThreshold, max_lowThreshold, CannyThreshold);

	Mat frame = Mat::zeros(200, 320, CV_8UC3); 
	bool paused = false;

	for (;;)
	{
		if (!paused)
		{
			cap >> frame;
			if (frame.empty())
				break;
		}

		frame.copyTo(image);

		if (!paused)
		{
			dst.create(image.size(), image.type());
			resize(image, blurred, Size(image.cols / 2, image.rows / 2));
			GaussianBlur(blurred, image, Size(7, 7), 10);
			cvtColor(image, src_gray, CV_BGR2GRAY);
			//namedWindow(window_name, CV_WINDOW_AUTOSIZE);
			CannyThreshold(0, 0);
		}
		
		//imshow("Edge Detection Demo", image);

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'p':
			paused = !paused;
			break;
		default:
			;
		}
	}

	return 0;
}
