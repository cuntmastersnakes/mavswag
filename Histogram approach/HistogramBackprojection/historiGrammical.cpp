// historiGrammical.cpp Defines the entry point to the application
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

/// Initialization variables, global variable definition
Mat image, blurred, threshed, closed;

int trackInit = 0;		// Variable to initialize the histogram region
/// Definition of the reference area
Rect selection;
Point lefttop, rightbottom;				// corner points of the reference area
int refWidth = 50, refHeight = 10;		// Percentage of the screen as reference
int vmin = 10, vmax = 256, smin = 30;	// threshold values for hue, value, saturation

/// Section on settings for the distance search
/// Blocks are vertically run over a number of columns in the image
/// If the sum of 'ones' in an area reaches a threshold, the box is returned
Rect block;				// block that searches for the threshold area value
int numCols = 16;		// number of cols for distance search
int numRows = 10;		// number of rows for distance search
float distThresh = 0.5;	// percentage of area of cube in column that needs to be filled

int loopFrame = 0;		// keeps track of internal loopings
int distances[16]; // Initialize the array for the output

Rect distFind(Mat image,int frame)
{
	int stepSize = image.rows / numRows;
	int stepSizeHor = image.cols / numCols;

	for (int i = image.rows; i > 0; i-= stepSize)
	{
		Point samplePoint1, samplePoint2; // lefttop and rightbottom of distance box
		samplePoint1.x = frame*stepSizeHor; samplePoint1.y = i-stepSize;
		samplePoint2.x = (frame+1)*stepSizeHor; 
		samplePoint2.y = i;

		Mat imageROI;
		Rect searchSpace = Rect(samplePoint1.x, samplePoint1.y,
			samplePoint2.x - samplePoint1.x, samplePoint2.y - samplePoint1.y);
		imageROI = image(searchSpace);

		int noOfZeros = countNonZero(imageROI);
		
		if (noOfZeros < searchSpace.area()*distThresh)
		{
			return searchSpace;
			break;
		}
		else
		{
			if (i == 0)
			{
				return searchSpace;
			}
		}

	}
}


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
	namedWindow("Trackbar fiddlin'", 0);

	createTrackbar("Vmin", "Trackbar fiddlin'", &vmin, 256, 0);
	createTrackbar("Vmax", "Trackbar fiddlin'", &vmax, 256, 0);
	createTrackbar("Smin", "Trackbar fiddlin'", &smin, 256, 0);
	createTrackbar("Width", "Trackbar fiddlin'", &refWidth, 100, 0);
	createTrackbar("Height", "Trackbar fiddlin'", &refHeight, 100, 0);

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
		GaussianBlur(blurred, image, Size(5, 5), 5);
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
		
		/// Calculate the black and white result
		calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
		backproj &= mask;
		//threshold(backproj, threshed, 127, 255, 0);
		Mat element = getStructuringElement(0, Size(9, 9));
		morphologyEx(backproj, closed, 1, element);
		morphologyEx(closed, backproj, 0, element);

		/// Find distances in the frame
		block = distFind(backproj, loopFrame);
		distances[loopFrame] = block.y-block.height;
		for (int j = 0; j < backproj.cols / numCols; j++)
		{
			Point distLineL, distLineR;
			distLineL.x = j*(backproj.cols/numCols); distLineL.y = distances[j];
			distLineR.x = (j + 1)*(backproj.cols / numCols); distLineR.y = distLineL.y;
			line(image, distLineL, distLineR, CV_RGB(255, 0, 0));
		}
		
		
		/// show windows with the results
		imshow("CamShift Demo", image);
		imshow("Histogram", histimg);
		imshow("backproj", ~backproj);

		/// Add one to the frame we are currently calculating
		/// for updating the distance bars in the image
		loopFrame++;

		if (loopFrame == numCols)
			loopFrame = 0;

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		
	}

	return 0;
}