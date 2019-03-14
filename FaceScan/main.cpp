#include "functions.h"

int main(int argc, char** argv)
{
	Net faceDetector = readNet("../opencv_face_detector.prototxt",
		"../opencv_face_detector.caffemodel");

	Net faceRecogn = readNet("../openface_nn4.small2.v1.t7");

	Mat frame, targetEmbedding;

	vector<Mat> targets; // DATABASE OF MAT-FORMATTED EMBEDDING VECTORS

	vector<vector<float>> embeddings; // EMBEDDINGS FROM DATABASE

	vector<string> names; // NAMES FROM DATABASE

	VideoCapture cap(0);

	readDatabase(embeddings, names, targets); // GET PREVIOUSLY WRITED EMBEDDINGS

	while (cap.read(frame))
	{
		vector<ROIC*> decs = getDetections(frame, faceDetector); // ARRAY OF FACES ROIS

		int key = waitKey(1);

		for (int i = 0; i < decs.size(); i++)
		{
			Mat embedding = getEmbedding(decs[i], frame, faceRecogn); //GET EMBEDDING VECTOR

			if (key == 32)
			{
				writeEmbedding(embedding, targets, names, embeddings);
			}
			else if (key == 27)
				return 0;

			Scalar color(0, 0, 255);  // Red
			int maxAt = findBest(targets, embedding);
			if (maxAt > -1)
			{

				color = Scalar(0, 255, 0);  // Green
				putText(frame, names[maxAt], Point(decs[i]->xmin, decs[i]->ymin - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 255, 0), 1);
			}
			
			rectangle(frame, Point(decs[i]->xmin, decs[i]->ymin), Point(decs[i]->xmax, decs[i]->ymax), color, 3);
		}
		imshow("out", frame);
	}

	return 0;
}
