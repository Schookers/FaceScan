#include "functions.h"

vector<ROIC*> getDetections(Mat& frame, Net& faceDetector)
{
	Mat blob;
	vector<ROIC*> facesBoxes; // ARRAY OF FACE DETECTIONS
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

		int xmin = max(0.0f, min(detections[i * 7 + 3], 1.0f)) * frame.cols;
		int ymin = max(0.0f, min(detections[i * 7 + 4], 1.0f)) * frame.rows;
		int xmax = max(0.0f, min(detections[i * 7 + 5], 1.0f)) * frame.cols;
		int ymax = max(0.0f, min(detections[i * 7 + 6], 1.0f)) * frame.rows;

		ROIC* roi = new ROIC{ xmin, xmax, ymin, ymax };

		facesBoxes.push_back(roi);
	}
	return facesBoxes;
}

Mat getEmbedding(ROIC * roi, Mat & frame, Net& faceRecogn)
{
	Mat blobRecogn;
	Mat curRoi = frame.rowRange(roi->ymin, roi->ymax).colRange(roi->xmin, roi->xmax); // CURRENT ROI
	blobFromImage(curRoi, blobRecogn, 1.0 / 255, Size(96, 96), Scalar(), true);
	faceRecogn.setInput(blobRecogn);
	Mat embedding = faceRecogn.forward();
	return embedding;
}
void embeddingHandler(Mat & embedding)
{

}

void writeEmbedding(Mat & embedding, vector<Mat>& targets, vector<string>& names, vector<vector<float>>& embeddings)
{
	string un;//username
	ofstream fout;
	
//	targets.push_back(embedding);

	fout.open("database.txt", ofstream::out | ofstream::app);

	cout << "Write your name and press enter" << endl;

	getline(cin, un);
	fout << un << endl;
	names.push_back(un);

	if (!fout)
	{
		cout << "File Not Opened" << endl;
		return;
	}

	vector<float> fEmbed; //EMBEDDING TO FLOAT

	for (int i = 0; i < embedding.rows; i++)
	{

		for (int j = 0; j < embedding.cols; j++)
		{
			fEmbed.push_back(embedding.at<float>(i, j));
			fout << embedding.at<float>(i, j) << "\t";
		}
		fout << endl;
	}
	embeddings.push_back(fEmbed);
	targets.push_back(Mat(1, embeddings[embeddings.size() - 1].size(), CV_32FC1, (float*)embeddings[embeddings.size()-1].data()));
	fout.close();

}

void readDatabase(vector<vector<float>>& embeddings, vector<string>& names, vector<Mat>& targets)
{
	fstream dbfile("database.txt");
	string line;

	int i = 0;
	while (getline(dbfile, line))
	{
		if (i%2 == 1)
		{
			float value;
			stringstream ss(line);

			embeddings.push_back(vector<float>());

			while (ss >> value)
			{
				embeddings[i/2].push_back(value);
			}

			targets.push_back(Mat( 1, embeddings[i/2].size(), CV_32FC1, (float*)embeddings[i/2].data()));

		}
		else
		{
			names.push_back(line);
		}
		++i;
	}

}
/*
void writeEmbedding(cv::Mat & embedding, const char * personName)
{

}
*/
