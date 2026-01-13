// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <iomanip>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void lbp(Mat img, Mat& features_lbp, int poz, int kernel, int hist_size) {
	int* hist = (int*)calloc(hist_size, sizeof(int));
	memset(hist, 0, hist_size);
	for (int i = kernel; i < img.rows - kernel;i++) {
		for (int j = kernel; j < img.cols - kernel;j++) {
			int code = 0;
			for (int a = i - kernel;a <= i + kernel;a++) {
				if (img.at<uchar>(i, j) < img.at<uchar>(a, j)) {
					int d = a - (i - kernel);
					code = code + (1 << d);
				}
			}
			int offset = 2 * kernel;
			for (int a = j - kernel + 1; a <= j + kernel;a++) {
				if (img.at<uchar>(i, j) < img.at<uchar>(i, a)) {
					int d = a - (j - kernel) + offset;
					code = code + (1 << d);
				}
			}
			offset += 2 * kernel;
			for (int a = i + kernel - 1; a >= i - kernel;a--) {
				if (img.at<uchar>(i, j) < img.at<uchar>(a, j)) {
					int d = (i + kernel) - a + offset;
					code = code + (1 << d);
				}
			}
			offset += 2 * kernel;
			for (int a = j + kernel - 1; a >= j - kernel + 1;a--) {
				if (img.at<uchar>(i, j) < img.at<uchar>(i, a)) {
					int d = j + kernel - a + offset;
					code = code + (1 << d);
				}
			}
			hist[code]++;
		}
	}
	for (int i = 0; i < hist_size;i++) {
		features_lbp.at<int>(poz, i) = hist[i];
	}
	free(hist);
}

void showHistForMat(Mat img, const std::string& name) {
	int hist[256] = { 0 };
	for (int i = 0;i < img.rows;i++) {
		for (int j = 0; j < img.cols;j++) {
			hist[img.at<uchar>(i, j)]++;
		}
	}
	showHistogram(name, hist, 256, 200);
}

#include <string>
#include <fstream>
#include <sstream>

std::vector<std::string> split(const std::string& s) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string item;

	while (std::getline(ss, item, ',')) {
		result.push_back(item);
	}

	return result;
}

void transformToMat(Mat& img, std::vector<int>& label, std::string s, int poz, int imgSize) {
	bool isLabel = true;
	int k = 0;
	for (auto it : split(s)) {
		if (isLabel) {
			isLabel = false;
			label[poz] = std::stoi(it);
		}
		else
		{
			img.at<uchar>(k / imgSize, k % imgSize) = std::stoi(it);
			k++;
		}

	}
}

void histEqualization(Mat& img) {
	int hist[256] = { 0 };
	const int WHITE = 255;
	float cpdf[256] = { 0 };
	for (int i = 0; i < img.rows;i++) {
		for (int j = 0; j < img.cols;j++) {
			hist[img.at<uchar>(i, j)]++;
		}
	}
	cpdf[0] = hist[0];
	for (int i = 1;i < 256;i++) {
		cpdf[i] = hist[i] + cpdf[i - 1];
	}
	for (int i = 0; i < 256;i++) {
		cpdf[i] = cpdf[i] / (img.rows * img.cols);
	}
	for (int i = 0; i < img.rows;i++) {
		for (int j = 0; j < img.cols;j++) {
			img.at<uchar>(i, j) = (uchar)std::round(WHITE * cpdf[img.at<uchar>(i, j)]);
		}
	}
}

void HOG(Mat src, Mat features, int poz, int featureStart, int hogSize) {
	const int SobelX[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
	const int SobelY[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	const float epsilon = 0.001;
	float hog[10] = { 0 };
	const float scaleFactor = 400.0f;
	int k = 1;
	for (int i = k; i < src.rows - k;i++) {
		for (int j = k; j < src.cols - k; j++) {
			float gx = 0, gy = 0;

			for (int a = 0; a < 3;a++) {
				for (int b = 0; b < 3; b++) {
					gx += (SobelX[a][b] * (int)src.at<uchar>(i - k + a, j - k + b));
					gy += (SobelY[a][b] * (int)src.at<uchar>(i - k + a, j - k + b));
				}
			}
			float mag = sqrt(gx * gx + gy * gy);
			float angle = atan2(gy, gx) * 180 / PI;
			if (angle < 0) {
				angle += 180;
			}
			for (int a = 0;a < 10;a++) {
				if (abs(a * 20 - angle) < epsilon) {
					hog[a] += mag;
					break;
				}
				if (abs(a * 20 - angle) < 20) {
					hog[a] += (abs(a * 20 - angle) / 20 * mag);
				}
			}
		}
	}
	float l2 = 0;
	for (int i = 0; i < 10;i++) {
		l2 += hog[i] * hog[i];
	}
	l2 = sqrt(l2);
	for (int i = 0; i < 10; i++) {
		hog[i] = hog[i] / (l2 + epsilon);
	}
	for (int i = 0; i < hogSize; i++) {
		features.at<int>(poz, featureStart + i) = (int)std::round(hog[i] * scaleFactor);
	}
}

//Pretty print the confusion matrix
void printConfusionMatrix(const Mat& confMat, const std::vector<std::string>& classes, std::ostream& out) {
	int width = 12;
	out << "\n" << std::string(80, '=') << "\n";
	out << "CONFUSION MATRIX (Rows = True, Cols = Predicted)\n";
	out << std::string(80, '-') << "\n";

	out << std::setw(width) << " ";
	for (const auto& cls : classes) {
		std::string name = cls.length() > width - 2 ? cls.substr(0, width - 2) : cls;
		out << std::setw(width) << name;
	}
	out << "\n";

	for (int i = 0; i < confMat.rows; i++) {
		std::string name = classes[i].length() > width - 2 ? classes[i].substr(0, width - 2) : classes[i];
		out << std::setw(width) << name;
		for (int j = 0; j < confMat.cols; j++) {
			out << std::setw(width) << confMat.at<int>(i, j);
		}
		out << "\n";
	}
	out << std::string(80, '=') << "\n";
}

//Pretty print Precision, Recall, F1, and Accuracy
void calculateAndPrintMetrics(const Mat& confMat, const std::vector<std::string>& classes, std::ostream& out) {
	int n = confMat.rows;
	double totalSamples = 0;
	double totalCorrect = 0;

	std::vector<double> precision(n), recall(n), f1(n);

	//out << "\nMODEL PERFORMANCE METRICS\n";
	//out << std::string(80, '-') << "\n";
	//out << std::setw(15) << "Class"
	//	<< std::setw(15) << "Precision"
	//	<< std::setw(15) << "Recall"
	//	<< std::setw(15) << "F1 Score" << "\n";
	//out << std::string(80, '-') << "\n";

	for (int i = 0; i < n; i++) {
		int tp = confMat.at<int>(i, i);
		int fp = 0;
		int fn = 0;

		//false negatives
		for (int col = 0; col < n; col++) {
			if (col != i) fn += confMat.at<int>(i, col);
		}

		//false positives
		for (int row = 0; row < n; row++) {
			if (row != i) fp += confMat.at<int>(row, i);
		}

		totalCorrect += tp;
		totalSamples += (tp + fn);

		precision[i] = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
		recall[i] = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
		f1[i] = (precision[i] + recall[i]) > 0 ? 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i]) : 0.0;

		//out << std::setw(15) << classes[i]
		//	<< std::setw(15) << std::fixed << std::setprecision(4) << precision[i]
		//	<< std::setw(15) << std::fixed << std::setprecision(4) << recall[i]
		//	<< std::setw(15) << std::fixed << std::setprecision(4) << f1[i] << "\n";
	}

	double macroP = 0, macroR = 0, macroF1 = 0;
	for (int i = 0; i < n; i++) {
		macroP += precision[i];
		macroR += recall[i];
		macroF1 += f1[i];
	}
	macroP /= n;
	macroR /= n;
	macroF1 /= n;
	double accuracy = totalSamples > 0 ? totalCorrect / totalSamples : 0.0;

	out << std::string(80, '-') << "\n";
	out << "METRICS:\n";
	out << "  Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%\n";
	out << "  Precision:  " << macroP << "\n";
	out << "  Recall:     " << macroR << "\n";
	out << "  F1 Score:   " << macroF1 << "\n";
	out << std::string(80, '=') << "\n\n";
}

void fashionImageClassification(int mode) {
	const char train_path[] = "fashion-mnist_train.csv";
	const char test_path[] = "fashion-mnist_test.csv";
	const char train_features_file[] = "train_features.yml";
	const char test_features_file[] = "test_features.yml";

	const int nrTrain = 60000;
	const int nrTest = 10000;
	const int nrClasses = 10;
	const int imgSize = 28;
	const int hogSize = 10;
	const int featureSize = 255;
	const int kernelSize = 1;
	const int neighborsNum = 31;
	const int demoSize = 100;

	std::vector<std::string> classes = {
		"Tshirt/top", "Trouser", "Pullover", "Dress", "Coat",
		"Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
	};

	std::vector<int> trainLabel(nrTrain);
	std::vector<int> testLabel(nrTest);

	Mat features;
	Mat testFeatures;

	if (mode == 1) {
		//Mode 1: Compute features and labels
		std::vector<Mat> trainImages(nrTrain);
		std::vector<Mat> testImages(nrTest);

		std::ifstream trainIn(train_path);
		std::string line;
		trainIn >> line;
		int k = 0;
		while (trainIn >> line) {
			Mat a(imgSize, imgSize, CV_8UC1);
			transformToMat(a, trainLabel, line, k, imgSize);
			trainImages[k++] = a;
			if (k % 5000 == 0) std::cout << k << " training images processed\n";
			if (k >= nrTrain) break;
		}
		for (int i = 0; i < nrTrain; i++) {
			histEqualization(trainImages[i]);
		}
		showHistForMat(trainImages[0], "trainImage");

		std::ifstream testIn(test_path);
		testIn >> line;
		k = 0;
		while (testIn >> line) {
			Mat a(imgSize, imgSize, CV_8UC1);
			transformToMat(a, testLabel, line, k, imgSize);
			testImages[k++] = a;
			if (k % 5000 == 0) {
				std::cout << k << " test images processed\n";
			}
			if (k >= nrTest) {
				break;
			}
		}
		for (int i = 0; i < nrTest; i++) {
			histEqualization(testImages[i]);
		}

		std::cout << "Calculating features...\n";
		features = Mat(nrTrain, featureSize + hogSize, CV_32SC1);
		for (int i = 0; i < nrTrain; i++) {
			lbp(trainImages[i], features, i, kernelSize, featureSize);
			HOG(trainImages[i], features, i, featureSize, hogSize);
		}

		testFeatures = Mat(nrTest, featureSize + hogSize, CV_32SC1);
		for (int i = 0; i < nrTest; i++) {
			lbp(testImages[i], testFeatures, i, kernelSize, featureSize);
			HOG(testImages[i], testFeatures, i, featureSize, hogSize);
		}

		cv::FileStorage fs_train(train_features_file, cv::FileStorage::WRITE);
		fs_train << "features" << features;
		fs_train << "labels" << trainLabel;
		fs_train.release();
		std::cout << "Training data (features + labels) saved to: " << train_features_file << "\n";

		cv::FileStorage fs_test(test_features_file, cv::FileStorage::WRITE);
		fs_test << "testFeatures" << testFeatures;
		fs_test << "labels" << testLabel;
		fs_test.release();
		std::cout << "Test data (features + labels) saved to: " << test_features_file << "\n";
	}
	else if (mode == 2) {
		//Mode 2: Load cached data and classify
		std::cout << "Loading cached data...\n";
		cv::FileStorage fs_train(train_features_file, cv::FileStorage::READ);
		if (!fs_train.isOpened()) {
			std::cerr << "ERROR: Run Mode 1 first!\n"; return;
		}
		fs_train["features"] >> features;
		fs_train["labels"] >> trainLabel;
		fs_train.release();

		cv::FileStorage fs_test(test_features_file, cv::FileStorage::READ);
		if (!fs_test.isOpened()) {
			std::cerr << "ERROR: Run Mode 1 first!\n"; return;
		}
		fs_test["testFeatures"] >> testFeatures;
		fs_test["labels"] >> testLabel;
		fs_test.release();

		if (features.empty() || testFeatures.empty()) {
			std::cerr << "ERROR: Data loaded is empty.\n"; return;
		}
		std::cout << "Data loaded successfully. Starting classification...\n";

		std::ofstream fileOut("comparison_results.txt");

		//Model 1: KNN
		std::cout << "\nRunning KNN (k=" << neighborsNum << ")...\n";
		Mat confMatKNN(nrClasses, nrClasses, CV_32SC1);
		confMatKNN.setTo(0);

		for (int i = 0; i < demoSize; i++) {
			std::vector<std::pair<float, int>> dist(nrTrain);
			if (i % 10 == 0) {
				std::cout << "KNN Progress: " << i << "/" << demoSize << std::endl;
			}
			for (int j = 0; j < nrTrain; j++) {
				float currentDist = 0;
				for (int a = 0; a < featureSize + hogSize; a++) {
					float diff = (float)(features.at<int>(j, a) - testFeatures.at<int>(i, a));
					currentDist += diff * diff;
				}
				dist[j] = { sqrt(currentDist), trainLabel[j] };
			}
			std::nth_element(dist.begin(), dist.begin() + neighborsNum, dist.end(),
				[](const std::pair<float, int>& a, const std::pair<float, int>& b) {
					return a.first < b.first;
				}
			);
			std::vector<int> c(nrClasses, 0);
			for (int j = 0; j < neighborsNum; j++) {
				c[dist[j].second]++;
			}
			int cl = -1, mx = -1;
			for (int j = 0; j < nrClasses; j++) {
				if (mx < c[j]) {
					mx = c[j];
					cl = j;
				}
			}
			confMatKNN.at<int>(testLabel[i], cl)++;
		}
		std::cout << "\KNN Finished.\n";

		fileOut << "\n[MODEL: KNN (k=" << neighborsNum << ")]";
		printConfusionMatrix(confMatKNN, classes, fileOut);
		calculateAndPrintMetrics(confMatKNN, classes, fileOut);

		//the other two models need floating point data and labels in a specific format
		Mat trainDataFloat, testDataFloat;
		features.convertTo(trainDataFloat, CV_32F);
		testFeatures.convertTo(testDataFloat, CV_32F);
		//SVM is really sensitive to data ranges, needs normalization
		normalize(trainDataFloat, trainDataFloat, 0, 1, NORM_MINMAX);
		normalize(testDataFloat, testDataFloat, 0, 1, NORM_MINMAX);
		Mat trainLabelsMat(nrTrain, 1, CV_32S, trainLabel.data());

		//Model 2: SVM
		std::cout << "Running SVM...\n";

		using namespace cv::ml;
		Ptr<SVM> svm = SVM::create();
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::LINEAR);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
		svm->train(trainDataFloat, ROW_SAMPLE, trainLabelsMat);
		std::cout << "SVM Trained. Predicting on Test Set...\n";
		Mat confMatSVM(nrClasses, nrClasses, CV_32SC1);
		confMatSVM.setTo(0);
		for (int i = 0; i < demoSize; i++) {
			if (i % 10 == 0) {
				std::cout << "SVM Progress: " << i << "/" << demoSize << std::endl;
			}
			float response = svm->predict(testDataFloat.row(i));
			confMatSVM.at<int>(testLabel[i], (int)response)++;
		}

		fileOut << "\n[MODEL: SVM (RBF)]";
		printConfusionMatrix(confMatSVM, classes, fileOut);
		calculateAndPrintMetrics(confMatSVM, classes, fileOut);


		//Model 3: Random Forest
		std::cout << "Training Random Forest (300 Trees)...\n";

		Ptr<RTrees> rf = RTrees::create();
		rf->setMaxDepth(30);
		rf->setMinSampleCount(2);
		rf->setActiveVarCount(16);
		rf->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 300, 0.1));
		rf->train(trainDataFloat, ROW_SAMPLE, trainLabelsMat);
		std::cout << "Random Forest Trained. Predicting on Test Set...\n";
		Mat confMatRF(nrClasses, nrClasses, CV_32SC1);
		confMatRF.setTo(0);
		for (int i = 0; i < demoSize; i++) {
			if (i % 10 == 0) {
				std::cout << "RF Progress: " << i << "/" << demoSize << std::endl;
			}
			float response = rf->predict(testDataFloat.row(i));
			confMatRF.at<int>(testLabel[i], (int)response)++;
		}

		fileOut << "\n[MODEL: Random Forest (300 Trees)]";
		printConfusionMatrix(confMatRF, classes, fileOut);
		calculateAndPrintMetrics(confMatRF, classes, fileOut);


		fileOut.close();
		std::cout << "Comparison finished. All results saved to comparison_results.txt\n";
		waitKey(0);
	}
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - FashionImageClassification\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testNegativeImage();
			break;
		case 4:
			testNegativeImageFast();
			break;
		case 5:
			testColor2Gray();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 7:
			testBGR2HSV();
			break;
		case 8:
			testResize();
			break;
		case 9:
			testCanny();
			break;
		case 10:
			testVideoSequence();
			break;
		case 11:
			testSnap();
			break;
		case 12:
			testMouseClick();
			break;
		case 13: {
			int k;
			std::cout << "Introduce the mode(1 - Train, 2 - Classify)\n";
			std::cin >> k;
			fashionImageClassification(k);
			break;
		}

		}
	} while (op != 0);
	return 0;
}