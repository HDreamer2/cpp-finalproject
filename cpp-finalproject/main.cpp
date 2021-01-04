#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>
#include "face_binary_cls.cpp"
using namespace cv;
class matrix {
public:
	const int row;
	const int col;
	float* data;
	
	matrix(int row, int col, float* data):row(row),col(col),data(data){}
	matrix(int row, int col):row(row),col(col){}
};
//the size of data is 3*row * col and 
//the sequence of every matrix is row 1->row n


float simd_dot(const float* x, const float* y, const long& len) 
{
	float inner_prod = 0.0f;
	__m128 X, Y; //声明两个存放在SSE的128位专用寄存器的变量
	__m128 acc = _mm_setzero_ps(); // 声明一个存放在SSE的128位专用寄存器的变量，用来存放X+Y的结果，初始值为0
	float temp[4];//存放中间变量的参数

	long i;
	for (i = 0; i + 4 < len; i += 4) {//128位专用寄存器，一次性可以处理4组32位变量的运算
		X = _mm_loadu_ps(x + i); // 将x加载到X（由于128位可以存放四个32位数据，所以默认一次加载连续的4个参数）
		Y = _mm_loadu_ps(y + i);//同上
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));//x*y，每轮的x1*y1求和，x2*y2求和，x3*y3求和，x4*y4求和,最终产生的四个和，放在acc的128位寄存器中。
	}
	_mm_storeu_ps(&temp[0], acc); // 将acc中的4个32位的数据加载进内存
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];//点乘求和

	// 刚才只是处理了前4的倍数个元素的点乘，如果len不是4的倍数，那么还有个小尾巴要处理一下
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];//继续累加小尾巴的乘积
	}
	return inner_prod;//大功告成
}

float* BGRtoRGB(Mat image, float* in)
{
	for (int i = 2; i >= 0; i--)
	{
		for (int j = 0; j < 128; j++)
		{
			for (int k = 0; k < 128; k++)
			{
				in[128 * 128 * (2 - i) + j * 128 + k] = (float)image.at<Vec3b>(j, k)[i] / 255.f;
				//R-G-B
			}
		}
	}
	return in;
}

int main()
{
	Mat image = imread("1.jpg");
	float* data = new float[3*128*128];
	data = BGRtoRGB(image, data);
	//conv0
	float* conv0_data = new float[16*64*128];
	for (int i = 0; i < 16; i++)//16 out_channels
	{
		for (int j = 0; j < 3; j++)//3 in_channels
		{
			for (int k = 0; k < 128; k++)//start from kth row
			{
				for (int l = 0; l+2 < 128; l+=2)//start from lth col
				{
					conv0_data[i * 64 * 128 + ] = data[j * 128 * 128 + ] * conv0_weight[i * 3 * 3 * 3 + j * 3 * 3 + 0];
				}
			}
		}
	}


	/*
	imshow("我的图片",image);

	waitKey(0);
	*/
	return 0;
}