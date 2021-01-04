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
	__m128 X, Y; //�������������SSE��128λר�üĴ����ı���
	__m128 acc = _mm_setzero_ps(); // ����һ�������SSE��128λר�üĴ����ı������������X+Y�Ľ������ʼֵΪ0
	float temp[4];//����м�����Ĳ���

	long i;
	for (i = 0; i + 4 < len; i += 4) {//128λר�üĴ�����һ���Կ��Դ���4��32λ����������
		X = _mm_loadu_ps(x + i); // ��x���ص�X������128λ���Դ���ĸ�32λ���ݣ�����Ĭ��һ�μ���������4��������
		Y = _mm_loadu_ps(y + i);//ͬ��
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));//x*y��ÿ�ֵ�x1*y1��ͣ�x2*y2��ͣ�x3*y3��ͣ�x4*y4���,���ղ������ĸ��ͣ�����acc��128λ�Ĵ����С�
	}
	_mm_storeu_ps(&temp[0], acc); // ��acc�е�4��32λ�����ݼ��ؽ��ڴ�
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];//������

	// �ղ�ֻ�Ǵ�����ǰ4�ı�����Ԫ�صĵ�ˣ����len����4�ı�������ô���и�Сβ��Ҫ����һ��
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];//�����ۼ�Сβ�͵ĳ˻�
	}
	return inner_prod;//�󹦸��
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
	imshow("�ҵ�ͼƬ",image);

	waitKey(0);
	*/
	return 0;
}