#include <opencv2/highgui/highgui.hpp>
#ifndef _GUIDEFILTER_H
#define _GUIDEFILTER_H

cv::Mat boxfilter(cv::Mat &imSrc, int r);
cv::Mat guidedfilter(cv::Mat &I, cv::Mat &p, int r, double eps);



#endif //_GUIDEFILTER_H

