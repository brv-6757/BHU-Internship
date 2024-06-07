#include <stdio.h>
#include <string.h>

//typedef ap_int<8> int_8;
void example(volatile int* a,volatile int* b,volatile int* c) {

#pragma HLS INTERFACE m_axi port = a depth = 10000
#pragma HLS INTERFACE m_axi port = b depth = 9
#pragma HLS INTERFACE m_axi port = c depth = 10000
#pragma HLS INTERFACE mode=s_axilite bundle=CTRL_BUS port=return


    int i,j,m,n;
    int pad = 0;
    int sum[100][100];

    int buff[10000];
    int ker[9];
    int mat[100][100];
    int kernel[3][3];

    memcpy(buff, (const int*)a, 10000 * sizeof(int));
    memcpy(ker, (const int*)b, 9 * sizeof(int));



    for (i = 0; i < 100; i++) {
    		for(j = 0 ; j < 100 ; j++) {
    			mat[i][j]=buff[100 * i + j ] ;
    		}
        }
    for (i = 0; i < 3; i++) {
    		for(j = 0 ; j < 3 ; j++) {
    			kernel[i][j] = ker[3 * i + j] ;
    		}
        }

//    for(i = pad; i < 99 - pad; i++){
//    	for(j = pad; j < 99 - pad; j++ ){
//    		mult_val = 0;
//    		for(m = 0; m < 3; m++){
//    			for(n = 0; n < 3; n++){
//    				mult_val = mult_val + ker[m][n] * mat[i + m - pad][j + n - pad];
//#pragma HLS PIPELINE
//    			}
//
//    		}
//    		sum[i - pad][j - pad] = mult_val;
//    	}
//    }
    for(int i = 0; i < 100; i++)
    {
    	for(int j = 0; j < 100; j++)
    	{
    		int convolved = 0;
    		if(i == 0 or i == 99 or j == 0 or j == 99) convolved = 0;
    		else
    		{
    			convolved += ((mat[i - 1][j - 1] * kernel[0][0]) + (mat[i - 1][j] * kernel[0][1])+
						(mat[i - 1][j + 1] * kernel[0][2]) + ((mat[i][j - 1] * kernel[1][0]))+
						(mat[i][j] * kernel[1][1]) + (mat[i][j + 1] * kernel[1][2]) +
						(mat[i + 1][j - 1] * kernel[2][0]) + (mat[i + 1][j] * kernel[2][1])+
						(mat[i + 1][j + 1] * kernel[2][2]));
    		}
    		sum[i][j] = convolved;
    	}
    }
    for(i = 0; i < 100 ; i++){
    	for(j = 0; j < 100 ; j++){
    		buff[100 * i + j] = sum[i][j];
    	}
    }

    memcpy((int*)c, buff, 10000 * sizeof(int));
}
