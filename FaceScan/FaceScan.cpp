#include "pch.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

void writeMatToFile(cv::Mat& m, const char* filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			fout << m.at<float>(i, j) << "\t";
		}
		fout << endl;
	}

	fout.close();
}


int main(int argc, char** argv) 
{
	Net faceDetector = readNet("../opencv_face_detector.prototxt",
		"../opencv_face_detector.caffemodel");
	Net faceRecogn = readNet("../openface_nn4.small2.v1.t7");

	VideoCapture cap(0);
	Mat frame, blob, blobRecogn, targetEmbedding;
	while (cap.read(frame))
	{
		

		int key = waitKey(1);
		// Get detections
		blobFromImage(frame, blob, 1.0, Size(160, 120), Scalar(104, 177, 123));

		faceDetector.setInput(blob);
		Mat out = faceDetector.forward();

		float* detections = (float*)out.data;
		// An every detection is a vector [batchId(0),classId(0),confidence,left,top,right,bottom]
		for (int i = 0; i < out.total() / 7; ++i)
		{
			float confidence = detections[i * 7 + 2];
			if (confidence < 0.7)
				continue;

			int xmin = std::max(0.0f, std::min(detections[i * 7 + 3], 1.0f)) * frame.cols;
			int ymin = std::max(0.0f, std::min(detections[i * 7 + 4], 1.0f)) * frame.rows;
			int xmax = std::max(0.0f, std::min(detections[i * 7 + 5], 1.0f)) * frame.cols;
			int ymax = std::max(0.0f, std::min(detections[i * 7 + 6], 1.0f)) * frame.rows;

			// Crop detected face and predict an embedding vector.
			Mat roi = frame.rowRange(ymin, ymax).colRange(xmin, xmax);

			blobFromImage(roi, blobRecogn, 1.0 / 255, Size(96, 96), Scalar(), true);
			faceRecogn.setInput(blobRecogn);
			Mat embedding = faceRecogn.forward();

			// Register a new embedding vector or compare with existing one.
			cout << "Press space" << endl;
			if (key == 32)
			{
				if (targetEmbedding.empty())
					targetEmbedding = embedding.clone();
				string FT;
				cout << "Write your name and press enter"<< endl;
				getline(cin,FT);
				stringstream ss;
				ss << FT << ".txt";
				cv::Mat m = cv::Mat::eye(5, 5, CV_32FC1);
				writeMatToFile(m, ss);
				
			}
			else if (key == 27)
				return 0;

			Scalar color(0, 0, 255);  // Red
			if (!targetEmbedding.empty() && embedding.dot(targetEmbedding) > 0.8)
			{
				color = Scalar(0, 255, 0);  // Green
			}
			else if (!targetEmbedding.empty() &&  ST.str() > 0.8)
			{
				color = Scalar(0, 255, 0);  // Green
			}
			rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), color, 3);
		}
		imshow("out", frame);
	}

	return 0;
}
