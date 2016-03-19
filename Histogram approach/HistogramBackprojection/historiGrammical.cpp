// HistogramBackprojection.cpp Defines the entry point to the application
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

Mat image, blurred, threshed, closed;

int trackInit = 0;		// Variable to initialize the histogram region
/// Definition of the reference area
Rect selection;
Point lefttop, rightbottom;				// corner points of the reference area
int refWidth = 60, refHeight = 20;		// Percentage of the screen as reference
int vmin = 10, vmax = 256, smin = 30;


int main(int argc, const char** argv)
{
	VideoCapture cap;
	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;
	
	cap.open(0);

	if (!cap.isOpened())
	{
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}

	/// Creation of the windows and the trackbars
	namedWindow("Histogram", 0);
	namedWindow("CamShift Demo", 0);

	createTrackbar("Width", "CamShift Demo", &refWidth, 100, 0);
	createTrackbar("Height", "CamShift Demo", &refHeight, 100, 0);
	createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);

	/// Reserve space for the different image representions and histograms
	Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
	int histchans[] = { 0,2 };

	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			break;
		
		frame.copyTo(image);

		resize(image, blurred, Size(image.cols / 2, image.rows / 2));
		GaussianBlur(blurred, image, Size(7, 7), 10);
		cvtColor(image, hsv, COLOR_BGR2HSV);

		/// Initialize the histogram calculations	
		int _vmin = vmin, _vmax = vmax;

		inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
			Scalar(180, 256, MAX(_vmin, _vmax)), mask);
		int ch[] = { 0,0 };
		///cout << hsv.type();

		hue.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &hue, 1, ch, 1);

		/// Define the selection region
		lefttop.x = image.cols * (0.5 - float(refWidth)/100/2);
		lefttop.y = image.rows * (1-float(refHeight)/100);
		rightbottom.x = image.cols * (0.5 + float(refWidth)/100 / 2);
		rightbottom.y = image.rows;

		selection = Rect(lefttop.x, lefttop.y, rightbottom.x-lefttop.x, rightbottom.y-lefttop.y);

		rectangle(image, lefttop, rightbottom, CV_RGB(255, 0, 0));
		
		///To be closer to the Nourbaksh implementation, the intensity
		///values should also be incorporated into the comparison
		Mat roi(hue, selection), maskroi(mask, selection);
		calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
		normalize(hist, hist, 0, 255, NORM_MINMAX);
			
		trackWindow = selection;

		histimg = Scalar::all(0);
		int binW = histimg.cols / hsize;
		Mat buf(1, hsize, CV_8UC3);
		for (int i = 0; i < hsize; i++)
			buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hsize), 255, 255);
		cvtColor(buf, buf, COLOR_HSV2BGR);
		
		for (int i = 0; i < hsize; i++)
		{
			int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);
			rectangle(histimg, Point(i*binW, histimg.rows),
				Point((i + 1)*binW, histimg.rows - val),
				Scalar(buf.at<Vec3b>(i)), -1, 8);
		}
		

		calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
		backproj &= mask;
		threshold(backproj, threshed, 127, 255, 0);
		Mat element = getStructuringElement(0, Size(9, 9));
		morphologyEx(threshed, closed, 1, element);
		morphologyEx(closed, backproj, 0, element);
		///Here is where the backprojection image is created
		RotatedRect trackBox = CamShift(backproj, trackWindow,
			TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
		if (trackWindow.area() <= 1)
		{
			int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
			trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
				trackWindow.x + r, trackWindow.y + r) &
				Rect(0, 0, cols, rows);
		}
		
		/// show windows with the results
		imshow("CamShift Demo", image);
		imshow("Histogram", histimg);
		imshow("backproj", backproj);

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		
	}

	return 0;
}