#include <opencv2/opencv.hpp>
#include <vector>
#include "PhotoMontage.h"

int main()
{
	std::vector<cv::Mat> images;
	///请在此处读入N 张照片,例子如下:
	images.push_back(cv::imread("./data/0.JPG"));
	images.push_back(cv::imread("./data/1.JPG"));
	//Images.push_back(cv::imread("2.JPG"));
	//Images.push_back(cv::imread("3.JPG"));
	//Images.push_back(cv::imread("4.JPG"));
	
	///请在此处读入N 张Label,注意这些label里面,白色的代表用户的笔触,黑色的代表背景,例子如下:
	std::vector<cv::Mat> labels;
	labels.push_back(cv::imread("./data/0.BMP"));
	labels.push_back(cv::imread("./data/1.BMP"));
	//Labels.push_back(cv::imread("2.BMP"));
	//Labels.push_back(cv::imread("3.BMP"));
	//Labels.push_back(cv::imread("4.BMP"));



	// the label of all images
	cv::Mat label(images[0].rows, images[0].cols, CV_8SC1);
	label.setTo(PhotoMontage::undefined);
	///set the labels according to the Labels read
	int height = images[0].rows;
	int width = images[0].cols;
	for (int label_idx = 0 ; label_idx < images.size(); label_idx++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0 ; x < width; x++)
			{
				if (labels[label_idx].at<cv::Vec3b>(y,x)(0) > 0)
				{
					label.at<uchar>(y,x) = label_idx;
				}
			}
		}
	}

	///Run photomontage
	PhotoMontage PM;
	PM.Run(images,label);

	return 0;
}

