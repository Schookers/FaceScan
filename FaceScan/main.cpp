﻿#include "functions.h"

int main(int argc, char** argv)
{
	Net faceDetector = readNet("../opencv_face_detector.prototxt",
		"../opencv_face_detector.caffemodel");
	Net faceRecogn = readNet("../openface_nn4.small2.v1.t7");

	Mat frame, targetEmbedding;// , blob, blobRecogn, targetEmbedding;

	vector<Mat> emDB; // DATABASE OF VECTORS

	VideoCapture cap(0);

	while (cap.read(frame))
	{
		vector<ROIC*> decs = getDetections(frame, faceDetector); // ARRAY OF FACES ROIS

		int key = waitKey(1);

		for (int i = 0; i < decs.size(); i++)
		{
			Mat embedding = getEmbedding(decs[i], frame, faceRecogn); //GET EMBEDDING VECTOR



			if (key == 32)
			{
				if (targetEmbedding.empty())
				{
					targetEmbedding = embedding.clone();
					writeEmbedding(targetEmbedding);

				}
			}
			else if (key == 27)
				return 0;
			Scalar color(0, 0, 255);  // Red
			if (!targetEmbedding.empty() && embedding.dot(targetEmbedding) > 0.8)
			{
				color = Scalar(0, 255, 0);  // Green
			}
			rectangle(frame, Point(decs[i]->xmin, decs[i]->ymin), Point(decs[i]->xmax, decs[i]->ymax), color, 3);
		}
		imshow("out", frame);
	}

	return 0;
}
