#include <iostream>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "guidefilter.h"
#include <queue>

#define SHOW_DEBUG_INFORMATION 1

#define MIN_FILTER_WINDOW_SIZE 3                                            //最小值滤波窗口大小
#define OMEGA_VALUE 0.95                                                    //图像去雾中去雾程度

//#define IMAGE_A 225

using namespace cv;
using namespace std;

int ImageCols;                                                              //图像宽度
int ImageRows;                                                              //图像高度
int ImageSize;                                                              //图像大小
int ImageA = 0;                                                             //全球大气光成分
const double omega = OMEGA_VALUE;                                           //图像去雾中去雾程度（保留1 - omega的雾）
uchar *ImageR;                                                              //图像的R分量
uchar *ImageG;                                                              //图像的G分量
uchar *ImageB;                                                              //图像的B分量
uchar *minRGB;                                                              //图像RGB各分量的最小值
uchar *ImageDarkChannel;                                                    //暗通道图像，即图像RGB各分量的最小值
double *ImageTx;                                                            //图像的透射率


void minMatrix(uchar ImageR[], uchar ImageG[], uchar ImageB[],uchar minRGB[]);  //计算RGB分量中最小值函数
void MinFilter(uchar minRGB[], uchar ImageDarkChannel[]);                       //最小值滤波函数
int getA(uchar *DarkChannel);
int getA_quick(uchar *DarkChannel);


int main()
{
    //读取的文件目录及名字
    const char *filename_pic[] = 
    {
        "../input_picture/0-tree.png",
        "../input_picture/1-building.png",          //图片有天空，非天空部分效果挺好
        "../input_picture/2-little_house.png",      //效果不好
        "../input_picture/3-many_house.jpg",        //效果很好
        "../input_picture/4-mountain.png",          //图片有天空，非天空部分效果挺好
        "../input_picture/5-red_tree.jpg",          //原始图像似乎没雾，处理前后无明显变化
        "../input_picture/6-river.png",             //效果不好
        "../input_picture/7-tiananmen.png",         //图片有大片天空，整体效果不太好
        "../input_picture/8-train.png",             //效果很好
        "../input_picture/9-tree.png",              //效果还可以
        "../input_picture/10-tree_house.png",       //效果很好
        "../input_picture/11-tree_mountain.jpg",    //效果很好，并且图片很大，适合做加速用例
        "../input_picture/12-tree_haze.png",        //效果一般
        "../input_picture/13-cars.jpg",             //路灯边缘效应严重
        "../input_picture/14-chongda-1.jpg",
        "../input_picture/15-chongda-2.jpg",
        "../input_picture/16-chongda-3.jpg",
    };
    const char *filename = filename_pic[ 2 ];
    
    //const char *filename = "../input_picture/重大亭子.jpg";
    //const char *filename = "../input_picture/鲜花.jpg";
    //const char *filename = "../input_picture/大门.jpg";
    //const char *filename = "../input_picture/绿山.bmp";
    //const char *filename = "../input_picture/有雾球场.jpg";
    //const char *filename = "../input_picture/树林.png";




    //读取图像，存储在srcImg中
    Mat srcImg = imread(filename, CV_LOAD_IMAGE_COLOR);
    if(srcImg.empty())
    {
        cout << "打开图片失败！" << endl;
        return -1;
    }
    
    imshow("srcImg", srcImg);                                               //显示读取的原始图像

    //获取并显示原始图像的灰度图
    //Mat srcImg_gray;
    //cvtColor(srcImg, srcImg_gray, CV_RGB2GRAY);
    //imshow("srcImg_gray", srcImg_gray);

    ImageRows = srcImg.rows;                                                //获取图像的高度
    ImageCols = srcImg.cols;                                                //获取图像的宽度
    ImageSize = ImageRows * ImageCols;                                      //获取图像像素的大小

    cout << "ImageCols：" << ImageCols << endl;                              //输出原始图像宽度信息
    cout << "ImageRows：" << ImageRows << endl;                              //输出原始图像高度信息
    cout << "ImageSize：" << ImageSize << endl << endl;                      //输出原始图像长宽大小信息
    
    //开辟内存空间
    ImageR = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageG = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageB = (uchar *)malloc(sizeof(uchar) * ImageSize);
    minRGB = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageDarkChannel = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageTx = (double *)malloc(sizeof(double) * ImageSize);

    //初始化开辟的内存空间（清零）
    memset(ImageR, 0, sizeof(uchar) * ImageSize);
    memset(ImageG, 0, sizeof(uchar) * ImageSize);
    memset(ImageB, 0, sizeof(uchar) * ImageSize);
    memset(minRGB, 0, sizeof(uchar) * ImageSize);
    memset(ImageDarkChannel, 0, sizeof(uchar) * ImageSize);
    memset(ImageTx, 0, sizeof(double) * ImageSize);   
    

    //获取原始图像的RGB分量
    for (int i = 0; i < ImageSize; i++)
    {
        *(ImageR + i) = srcImg.data[3 * i + 2];
        *(ImageG + i) = srcImg.data[3 * i + 1];
        *(ImageB + i) = srcImg.data[3 * i + 0];
    }


    double timeSpent = (double)getTickCount();                              //opencv 计算时间 开始计时
    //----------------------------------------------------------------------//开始时间


    minMatrix(ImageR, ImageG, ImageB, minRGB);                              //求原始图像RGB分量中的最小值
 /*  
    #if SHOW_DEBUG_INFORMATION
    //显示minRGB值对应的灰度图像
    Mat minRGB_gray(ImageRows, ImageCols, CV_8UC1);
    for (int i = 0; i < ImageSize; i++)
    {
        minRGB_gray.data[i] = *(minRGB + i);
    }
    imshow("minRGB_gray", minRGB_gray);    
    #endif
*/
    MinFilter(minRGB, ImageDarkChannel);                                    //对minRGB进行最小值滤波，得到原始图像的暗通道图像

    #if SHOW_DEBUG_INFORMATION
    //显示暗通道图像
    Mat ImageDarkChannel_gray(ImageRows, ImageCols, CV_8UC1);
    for (int i = 0; i < ImageSize; i++)
    {
        ImageDarkChannel_gray.data[i] = *(ImageDarkChannel + i);
    }
    imshow("ImageDarkChannel_gray", ImageDarkChannel_gray);
    #endif



    //通过暗通道图像ImageDarkChannel，计算A的值imageA    
    

    double time_s = (double)getTickCount(); 

    for (int i = 0; i < ImageSize; i++)
    {
        if (ImageA < *(ImageDarkChannel + i))
        {
            ImageA = *(ImageDarkChannel + i);
        }
    }
    
    time_s = ((double)getTickCount() - time_s) / getTickFrequency();
    cout << "maxA Time spent: " << time_s * 1000 << endl << ImageA << endl;         //opencv 计算程序处理时间


    time_s = (double)getTickCount(); 

    ImageA = getA(ImageDarkChannel);

    time_s = ((double)getTickCount() - time_s) / getTickFrequency();
    cout << "getA Time spent: " << time_s * 1000 << endl << ImageA << endl;         //opencv 计算程序处理时间


    time_s = (double)getTickCount(); 

    ImageA = getA_quick(ImageDarkChannel);

    time_s = ((double)getTickCount() - time_s) / getTickFrequency();
    cout << "getA_quick Time spent: " << time_s * 1000 << endl << ImageA << endl;         //opencv 计算程序处理时间

    

    #ifdef IMAGE_A
    ImageA = IMAGE_A;
    #endif
    cout << "ImageA:" << ImageA << endl;


    //通过暗通道图像ImageDarkChannel和A的值imageA计算透射率Tx
    for (int i = 0; i < ImageSize; i++)
    {
        *(ImageTx + i) = 1 - omega * ((double)*(ImageDarkChannel + i) / ImageA);
    }
    
    #if SHOW_DEBUG_INFORMATION
    //显示透射率Tx图像
    Mat ImageTx_gray(ImageRows, ImageCols, CV_8UC1);
    for (int i = 0; i < ImageSize; i++)
    {
        ImageTx_gray.data[i] = *(ImageTx + i) * ImageA;
    }
    imshow("ImageTx_gray", ImageTx_gray);
    #endif

    //#if SHOW_DEBUG_INFORMATION
    //根据公式，对原始图像进行去雾处理，把处理后的去雾图像的RGB分量合并成文件，准备输出
    Mat outImg_notGuideFilter(ImageRows, ImageCols, CV_8UC3); 
    for (int i = 0; i < ImageSize; i++)
    {
        int tempR = ImageA + (*(ImageR + i) - ImageA) / (*(ImageTx + i));
        if (tempR > 255)
        {
            tempR = 255;
        }
        else if (tempR < 0)
        {
            tempR = 0;
        }
        int tempG = ImageA + (*(ImageG + i) - ImageA) / (*(ImageTx + i));
        if (tempG > 255)
        {
            tempG = 255;
        }
        else if (tempG < 0)
        {
            tempG = 0;
        }
        int tempB = ImageA + (*(ImageB + i) - ImageA) / (*(ImageTx + i));
        if (tempB > 255)
        {
            tempB = 255;
        }
        else if (tempB < 0)
        {
            tempB = 0;
        }

        outImg_notGuideFilter.data[3 * i + 0] = tempB;
        outImg_notGuideFilter.data[3 * i + 1] = tempG;
        outImg_notGuideFilter.data[3 * i + 2] = tempR;
    }
    imshow("outImg_notGuideFilter", outImg_notGuideFilter);                   //输出最终去雾图像
    //#endif



    int r = MIN_FILTER_WINDOW_SIZE * 4;
    double eps = 0.0000001;

    //获取并显示原始图像的灰度图
    Mat srcImg_gray(ImageRows, ImageCols, CV_8UC1);    
    cvtColor(srcImg, srcImg_gray, CV_RGB2GRAY);

    Mat ImageI(ImageRows, ImageCols, CV_64FC1);
    for (int i = 0; i < ImageSize; i++)
    {
        ImageI.at<double>(i) = srcImg_gray.data[i] / 255.0;
    }   


    Mat ImageTx_beforeGuide(ImageRows, ImageCols, CV_64FC1);
    for (int i = 0; i < ImageSize; i++)
    {
        ImageTx_beforeGuide.at<double>(i) = *(ImageTx + i);
    }
    //imshow("ImageTx_beforeGuide", ImageTx_beforeGuide);    
        
    Mat ImageTx_afterGuide = guidedfilter(ImageI, ImageTx_beforeGuide, r, eps);

    //imshow("ImageTx_afterGuide", ImageTx_afterGuide);

    for (int i = 0; i < ImageSize; i++)
    {
        *(ImageTx + i) = ImageTx_afterGuide.at<double>(i);
    }

    //根据公式，对原始图像进行去雾处理，处理后的RGB存储在ImageR[],ImageG[],ImageB[]
    for (int i = 0; i < ImageSize; i++)
    {
        int tempR = ImageA + (*(ImageR + i) - ImageA) / (*(ImageTx + i));
        if (tempR > 255)
        {
            tempR = 255;
        }
        else if (tempR < 0)
        {
            tempR = 0;
        }
        int tempG = ImageA + (*(ImageG + i) - ImageA) / (*(ImageTx + i));
        if (tempG > 255)
        {
            tempG = 255;
        }
        else if (tempG < 0)
        {
            tempG = 0;
        }
        int tempB = ImageA + (*(ImageB + i) - ImageA) / (*(ImageTx + i));
        if (tempB > 255)
        {
            tempB = 255;
        }
        else if (tempB < 0)
        {
            tempB = 0;
        }

        *(ImageR + i) = tempR;
        *(ImageG + i) = tempG;
        *(ImageB + i) = tempB;
    }


    //把处理后的去雾图像的RGB分量合并成文件，准备输出
    Mat outImg_usedGuideFilter(ImageRows, ImageCols, CV_8UC3);       
    for (int i = 0; i < ImageSize; i++)
    {
        outImg_usedGuideFilter.data[3 * i + 0] = *(ImageB + i);
        outImg_usedGuideFilter.data[3 * i + 1] = *(ImageG + i);
        outImg_usedGuideFilter.data[3 * i + 2] = *(ImageR + i);
    }


    //----------------------------------------------------------------------//结束时间    
    timeSpent = ((double)getTickCount() - timeSpent) / getTickFrequency();
    cout << "Time spent in ms(opencv): " << timeSpent * 1000 << endl;         //opencv 计算程序处理时间



    imshow("outImg_usedGuideFilter", outImg_usedGuideFilter);                   //输出最终去雾图像




    //释放所有自己开辟的内存空间
    free(ImageR);
    free(ImageG);
    free(ImageB);
    free(minRGB);
    free(ImageDarkChannel);
    free(ImageTx);

    //把所有指针幅值NULL，防止出现野指针
    ImageR = NULL;
    ImageG = NULL;
    ImageB = NULL;
    minRGB = NULL;
    ImageDarkChannel = NULL;
    ImageTx = NULL;    
    
    waitKey(0);
    return 0;
}

/*
 * 函数功能：求解输入图像RGB各分量中的最小值，存储与之等大小的空间
 * 输入：ImageR[m*n], ImageG[m*n], ImageB[m*n]
 * 输出：minRGB[m*n]
 */
void minMatrix(uchar ImageR[], uchar ImageG[], uchar ImageB[],uchar minRGB[])
{
    for (int i = 0; i < ImageSize; i++)
    {
        if (ImageR[i] < ImageG[i])
        {
            if (ImageR[i] < ImageB[i])
            {
                minRGB[i] = ImageR[i];
            }
            else
            {
                minRGB[i] = ImageB[i];
            }
        }
        else
        {
            if (ImageG[i] < ImageB[i])
            {
                minRGB[i] = ImageG[i];
            }
            else
            {
                minRGB[i] = ImageB[i];
            }
        }
    }
}


/*
 * 函数功能：最小值滤波函数，以固定窗口大小对图像进行最小值滤波
 * 输入：minRGB[m*n]
 * 输出：ImageDarkChannel[m*n]
 * 注释：边缘值不满足窗口大小的，以现有窗口大小寻找最小值
 */
void MinFilter(uchar minRGB[], uchar ImageDarkChannel[])
{
    int start_amendment = 0;
    if (MIN_FILTER_WINDOW_SIZE % 2 == 0)
    {
        start_amendment = 1;
    }
    else
    {
        start_amendment = 0;
    }
    for(int i = 0; i < ImageRows; i++)
    {
        for(int j = 0; j < ImageCols; j++)
        {
            int minValue = minRGB[(ImageCols * i) + j];
            int tempValue;
                
            //求窗口内的最小值
            for (int m = i - MIN_FILTER_WINDOW_SIZE / 2 + start_amendment; m < i + MIN_FILTER_WINDOW_SIZE / 2 + 1; m++)
            {
                for (int n = j - MIN_FILTER_WINDOW_SIZE / 2 + start_amendment; n < j + MIN_FILTER_WINDOW_SIZE / 2 + 1; n++)
                {
                    if (m < 0 || m > ImageRows - 1 || n < 0 || n > ImageCols - 1)
                    {
                        continue;
                    }
                    else
                    {
                        tempValue = minRGB[(ImageCols * m) + n];
                    }                        
                    if (tempValue < minValue)
                    {
                        minValue = tempValue;
                    }
                }
            }
            ImageDarkChannel[(ImageCols * i) + j] = minValue;            
        }
    }
}


int getA(uchar *DarkChannel)
{
    std::priority_queue<uchar, vector<uchar>, greater<uchar>> pq;
    int num = ImageSize * 0.001;
    for (int i = 0; i < ImageSize; ++i) 
    {
        pq.push(DarkChannel[i]);
        int tem = pq.top();
        if (pq.size() > num) 
        {
            pq.pop();
        }
    }
    int tempA = 0;
    while (!pq.empty()) 
    {
        uchar tmp = pq.top();
        tempA += tmp;
        pq.pop();
    }

    tempA = tempA / num;

    return tempA;
}

int getA_quick(uchar *DarkChannel)
{
    int darkchannel_num[256] = {0};
    for (int i = 0; i < ImageSize; ++i) 
    {
        darkchannel_num[DarkChannel[i]]++;
    }
    int num = ImageSize * 0.001;
    int tempA = 0;
    int num_i = 0;
    for (int i = 255; i > 0; --i)
    {
        if (darkchannel_num[i] > 0)
        {
            num_i += darkchannel_num[i];
            if (num_i < num)
            {
                tempA += i * darkchannel_num[i];
            }
            else
            {
                tempA += i * (darkchannel_num[i] - (num_i - num));
                break;
            }
        }
    }

    tempA = tempA / num;

    return tempA;
}