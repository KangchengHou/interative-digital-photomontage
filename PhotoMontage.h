#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class PhotoMontage
{
private:
	// solve the result
	void BuildSolveMRF(const std::vector<cv::Mat> & Images, const cv::Mat & Label);
	void VisResultLabelMap(const cv::Mat & ResultLabel, int n_label);
	void VisCompositeImage(const cv::Mat & ResultLabel, const std::vector<cv::Mat> & Images);
	void BuildSolveGradientFusion(const std::vector<cv::Mat> & Images, const cv::Mat & ResultLabel);
	void SolveChannel( int channel_idx, int constraint, const cv::Mat &color_gradient_x, const cv::Mat &color_gradient_y , cv::Mat & output  );
	void GradientAt(const cv::Mat & Image, int x, int y, cv::Vec3f & grad_x, cv::Vec3f & grad_y);
	
public:
	// program entry
	void Run(const std::vector<cv::Mat> & Images, const cv::Mat & Label);
private:
	cv::flann::Index * AddInertiaConstraint( const cv::Mat & Label );
public:
	enum
	{
		undefined = -1
	};
};