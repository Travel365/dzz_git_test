#include <iostream>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "guidefilter.h"
#include <queue>

#define SHOW_DEBUG_INFORMATION 1

#define MIN_FILTER_WINDOW_SIZE 3                                            //��Сֵ�˲����ڴ�С
#define OMEGA_VALUE 0.95                                                    //ͼ��ȥ����ȥ��̶�

//#define IMAGE_A 225

using namespace cv;
using namespace std;

int ImageCols;                                                              //ͼ����
int ImageRows;                                                              //ͼ��߶�
int ImageSize;                                                              //ͼ���С
int ImageA = 0;                                                             //ȫ�������ɷ�
const double omega = OMEGA_VALUE;                                           //ͼ��ȥ����ȥ��̶ȣ�����1 - omega����
uchar *ImageR;                                                              //ͼ���R����
uchar *ImageG;                                                              //ͼ���G����
uchar *ImageB;                                                              //ͼ���B����
uchar *minRGB;                                                              //ͼ��RGB����������Сֵ
uchar *ImageDarkChannel;                                                    //��ͨ��ͼ�񣬼�ͼ��RGB����������Сֵ
double *ImageTx;                                                            //ͼ���͸����


void minMatrix(uchar ImageR[], uchar ImageG[], uchar ImageB[],uchar minRGB[]);  //����RGB��������Сֵ����
void MinFilter(uchar minRGB[], uchar ImageDarkChannel[]);                       //��Сֵ�˲�����
int getA(uchar *DarkChannel);
int getA_quick(uchar *DarkChannel);


int main()
{
    //��ȡ���ļ�Ŀ¼������
    const char *filename_pic[] = 
    {
        "../input_picture/0-tree.png",
        "../input_picture/1-building.png",          //ͼƬ����գ�����ղ���Ч��ͦ��
        "../input_picture/2-little_house.png",      //Ч������
        "../input_picture/3-many_house.jpg",        //Ч���ܺ�
        "../input_picture/4-mountain.png",          //ͼƬ����գ�����ղ���Ч��ͦ��
        "../input_picture/5-red_tree.jpg",          //ԭʼͼ���ƺ�û������ǰ�������Ա仯
        "../input_picture/6-river.png",             //Ч������
        "../input_picture/7-tiananmen.png",         //ͼƬ�д�Ƭ��գ�����Ч����̫��
        "../input_picture/8-train.png",             //Ч���ܺ�
        "../input_picture/9-tree.png",              //Ч��������
        "../input_picture/10-tree_house.png",       //Ч���ܺ�
        "../input_picture/11-tree_mountain.jpg",    //Ч���ܺã�����ͼƬ�ܴ��ʺ�����������
        "../input_picture/12-tree_haze.png",        //Ч��һ��
        "../input_picture/13-cars.jpg",             //·�Ʊ�ԵЧӦ����
        "../input_picture/14-chongda-1.jpg",
        "../input_picture/15-chongda-2.jpg",
        "../input_picture/16-chongda-3.jpg",
    };
    const char *filename = filename_pic[ 2 ];
    
    //const char *filename = "../input_picture/�ش�ͤ��.jpg";
    //const char *filename = "../input_picture/�ʻ�.jpg";
    //const char *filename = "../input_picture/����.jpg";
    //const char *filename = "../input_picture/��ɽ.bmp";
    //const char *filename = "../input_picture/������.jpg";
    //const char *filename = "../input_picture/����.png";




    //��ȡͼ�񣬴洢��srcImg��
    Mat srcImg = imread(filename, CV_LOAD_IMAGE_COLOR);
    if(srcImg.empty())
    {
        cout << "��ͼƬʧ�ܣ�" << endl;
        return -1;
    }
    
    imshow("srcImg", srcImg);                                               //��ʾ��ȡ��ԭʼͼ��

    //��ȡ����ʾԭʼͼ��ĻҶ�ͼ
    //Mat srcImg_gray;
    //cvtColor(srcImg, srcImg_gray, CV_RGB2GRAY);
    //imshow("srcImg_gray", srcImg_gray);

    ImageRows = srcImg.rows;                                                //��ȡͼ��ĸ߶�
    ImageCols = srcImg.cols;                                                //��ȡͼ��Ŀ��
    ImageSize = ImageRows * ImageCols;                                      //��ȡͼ�����صĴ�С

    cout << "ImageCols��" << ImageCols << endl;                              //���ԭʼͼ������Ϣ
    cout << "ImageRows��" << ImageRows << endl;                              //���ԭʼͼ��߶���Ϣ
    cout << "ImageSize��" << ImageSize << endl << endl;                      //���ԭʼͼ�񳤿��С��Ϣ
    
    //�����ڴ�ռ�
    ImageR = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageG = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageB = (uchar *)malloc(sizeof(uchar) * ImageSize);
    minRGB = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageDarkChannel = (uchar *)malloc(sizeof(uchar) * ImageSize);
    ImageTx = (double *)malloc(sizeof(double) * ImageSize);

    //��ʼ�����ٵ��ڴ�ռ䣨���㣩
    memset(ImageR, 0, sizeof(uchar) * ImageSize);
    memset(ImageG, 0, sizeof(uchar) * ImageSize);
    memset(ImageB, 0, sizeof(uchar) * ImageSize);
    memset(minRGB, 0, sizeof(uchar) * ImageSize);
    memset(ImageDarkChannel, 0, sizeof(uchar) * ImageSize);
    memset(ImageTx, 0, sizeof(double) * ImageSize);   
    

    //��ȡԭʼͼ���RGB����
    for (int i = 0; i < ImageSize; i++)
    {
        *(ImageR + i) = srcImg.data[3 * i + 2];
        *(ImageG + i) = srcImg.data[3 * i + 1];
        *(ImageB + i) = srcImg.data[3 * i + 0];
    }


    double timeSpent = (double)getTickCount();                              //opencv ����ʱ�� ��ʼ��ʱ
    //----------------------------------------------------------------------//��ʼʱ��


    minMatrix(ImageR, ImageG, ImageB, minRGB);                              //��ԭʼͼ��RGB�����е���Сֵ
 /*  
    #if SHOW_DEBUG_INFORMATION
    //��ʾminRGBֵ��Ӧ�ĻҶ�ͼ��
    Mat minRGB_gray(ImageRows, ImageCols, CV_8UC1);
    for (int i = 0; i < ImageSize; i++)
    {
        minRGB_gray.data[i] = *(minRGB + i);
    }
    imshow("minRGB_gray", minRGB_gray);    
    #endif
*/
    MinFilter(minRGB, ImageDarkChannel);                                    //��minRGB������Сֵ�˲����õ�ԭʼͼ��İ�ͨ��ͼ��

    #if SHOW_DEBUG_INFORMATION
    //��ʾ��ͨ��ͼ��
    Mat ImageDarkChannel_gray(ImageRows, ImageCols, CV_8UC1);
    for (int i = 0; i < ImageSize; i++)
    {
        ImageDarkChannel_gray.data[i] = *(ImageDarkChannel + i);
    }
    imshow("ImageDarkChannel_gray", ImageDarkChannel_gray);
    #endif



    //ͨ����ͨ��ͼ��ImageDarkChannel������A��ֵimageA    
    

    double time_s = (double)getTickCount(); 

    for (int i = 0; i < ImageSize; i++)
    {
        if (ImageA < *(ImageDarkChannel + i))
        {
            ImageA = *(ImageDarkChannel + i);
        }
    }
    
    time_s = ((double)getTickCount() - time_s) / getTickFrequency();
    cout << "maxA Time spent: " << time_s * 1000 << endl << ImageA << endl;         //opencv ���������ʱ��


    time_s = (double)getTickCount(); 

    ImageA = getA(ImageDarkChannel);

    time_s = ((double)getTickCount() - time_s) / getTickFrequency();
    cout << "getA Time spent: " << time_s * 1000 << endl << ImageA << endl;         //opencv ���������ʱ��


    time_s = (double)getTickCount(); 

    ImageA = getA_quick(ImageDarkChannel);

    time_s = ((double)getTickCount() - time_s) / getTickFrequency();
    cout << "getA_quick Time spent: " << time_s * 1000 << endl << ImageA << endl;         //opencv ���������ʱ��

    

    #ifdef IMAGE_A
    ImageA = IMAGE_A;
    #endif
    cout << "ImageA:" << ImageA << endl;


    //ͨ����ͨ��ͼ��ImageDarkChannel��A��ֵimageA����͸����Tx
    for (int i = 0; i < ImageSize; i++)
    {
        *(ImageTx + i) = 1 - omega * ((double)*(ImageDarkChannel + i) / ImageA);
    }
    
    #if SHOW_DEBUG_INFORMATION
    //��ʾ͸����Txͼ��
    Mat ImageTx_gray(ImageRows, ImageCols, CV_8UC1);
    for (int i = 0; i < ImageSize; i++)
    {
        ImageTx_gray.data[i] = *(ImageTx + i) * ImageA;
    }
    imshow("ImageTx_gray", ImageTx_gray);
    #endif

    //#if SHOW_DEBUG_INFORMATION
    //���ݹ�ʽ����ԭʼͼ�����ȥ�����Ѵ�����ȥ��ͼ���RGB�����ϲ����ļ���׼�����
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
    imshow("outImg_notGuideFilter", outImg_notGuideFilter);                   //�������ȥ��ͼ��
    //#endif



    int r = MIN_FILTER_WINDOW_SIZE * 4;
    double eps = 0.0000001;

    //��ȡ����ʾԭʼͼ��ĻҶ�ͼ
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

    //���ݹ�ʽ����ԭʼͼ�����ȥ����������RGB�洢��ImageR[],ImageG[],ImageB[]
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


    //�Ѵ�����ȥ��ͼ���RGB�����ϲ����ļ���׼�����
    Mat outImg_usedGuideFilter(ImageRows, ImageCols, CV_8UC3);       
    for (int i = 0; i < ImageSize; i++)
    {
        outImg_usedGuideFilter.data[3 * i + 0] = *(ImageB + i);
        outImg_usedGuideFilter.data[3 * i + 1] = *(ImageG + i);
        outImg_usedGuideFilter.data[3 * i + 2] = *(ImageR + i);
    }


    //----------------------------------------------------------------------//����ʱ��    
    timeSpent = ((double)getTickCount() - timeSpent) / getTickFrequency();
    cout << "Time spent in ms(opencv): " << timeSpent * 1000 << endl;         //opencv ���������ʱ��



    imshow("outImg_usedGuideFilter", outImg_usedGuideFilter);                   //�������ȥ��ͼ��




    //�ͷ������Լ����ٵ��ڴ�ռ�
    free(ImageR);
    free(ImageG);
    free(ImageB);
    free(minRGB);
    free(ImageDarkChannel);
    free(ImageTx);

    //������ָ���ֵNULL����ֹ����Ұָ��
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
 * �������ܣ��������ͼ��RGB�������е���Сֵ���洢��֮�ȴ�С�Ŀռ�
 * ���룺ImageR[m*n], ImageG[m*n], ImageB[m*n]
 * �����minRGB[m*n]
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
 * �������ܣ���Сֵ�˲��������Թ̶����ڴ�С��ͼ�������Сֵ�˲�
 * ���룺minRGB[m*n]
 * �����ImageDarkChannel[m*n]
 * ע�ͣ���Եֵ�����㴰�ڴ�С�ģ������д��ڴ�СѰ����Сֵ
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
                
            //�󴰿��ڵ���Сֵ
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