#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;



// ROI COORDINATES STRUCTURE
struct ROIC
{
	int xmin;
	int xmax;
	int ymin;
	int ymax;
};

vector<ROIC*> getDetections(Mat& frame, Net& faceDetect); // FUNCTION TO GET DETECTIONS

Mat getEmbedding(ROIC* roi, Mat & frame, Net& faceRecogn); // GET EMBEDDING VECTOR FROM FACE

void embeddingHandler(Mat& embedding);

void writeEmbedding(Mat& embedding, vector<Mat>& targets, vector<string>& names, vector<vector<float>>& embeddings);

void readDatabase(vector<vector<float>>& embeddings, vector<string>& names, vector<Mat>& targets);
/*
void writeEmbedding(cv::Mat& embedding, const char* personName);



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
*/

int maxVector(vector<vector<float>>& embeddings, Mat & embedding, vector<Mat>& targets,int& number);
 