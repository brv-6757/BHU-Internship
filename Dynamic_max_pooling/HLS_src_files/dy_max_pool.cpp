#include <ap_int.h>
#include <iostream>
#include<stdio.h>

typedef float Dtype_f;

#define MAX_HEIGHT 30
#define MAX_WIDTH 30

void dy_max_pool(int img_height,int img_width,int kernel_size,Dtype_f input[],Dtype_f output[])
{
#pragma HLS INTERFACE m_axi port=input depth=900
#pragma HLS INTERFACE m_axi port=output depth=900
#pragma HLS INTERFACE s_axilite port=img_height bundle=CTRL
#pragma HLS INTERFACE s_axilite port=img_width bundle=CTRL
#pragma HLS INTERFACE s_axilite port=kernel_size bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

int out_height = img_height/kernel_size;
int out_width = img_width/kernel_size;


for (int h = 0; h < out_height; ++h) {
        for (int w = 0; w < out_width; ++w) {
//#pragma HLS PIPELINE
            float max_val = 0;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    float cur_val = input[(h * kernel_size + kh) * img_width + (w * kernel_size + kw)];
                    printf("%f ",cur_val);
                    if (cur_val > max_val) {
                        max_val = cur_val;
                    }
                }
            }
            output[h * out_width + w] = max_val;
        }
    }
}


