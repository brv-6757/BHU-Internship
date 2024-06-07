#include "stdio.h"
//#include "pool_core.h"

//#define MODE 	2	//mode: 0:MEAN, 1:MIN, 2:MAX
#define IN_WIDTH 6
#define IN_HEIGHT 6
#define IN_CH 1

#define KERNEL 3
//#define KERNEL_HEIGHT 3

#define OUT_WIDTH (IN_WIDTH/KERNEL)
#define OUT_HEIGHT (IN_HEIGHT/KERNEL)
typedef float Dtype_f;


void dy_max_pool(
    int img_height,
    int img_width,
    int kernel_size,
	Dtype_f	input[],
	Dtype_f output[]
);

int main(void)
{
	Dtype_f feature_in[IN_HEIGHT*IN_WIDTH];
	Dtype_f feature_out[OUT_HEIGHT*OUT_WIDTH];

	for(int i=0;i<IN_HEIGHT;i++)
		{for(int j=0;j<IN_WIDTH;j++)
			{ feature_in[i*IN_WIDTH+j]=i*IN_WIDTH+j; printf("%d ",i*IN_WIDTH+j);}
		printf("\n");
		}


	dy_max_pool(IN_HEIGHT,IN_WIDTH,KERNEL,feature_in,feature_out);//mode: 0:MEAN, 1:MIN, 2:MAX

	printf("\n");
	for(int i=0;i<OUT_HEIGHT;i++)
		{for(int j=0;j<OUT_WIDTH;j++)
			{
				printf("%f ",feature_out[i*IN_WIDTH+j]);
			}
		printf("\n");
		}
}
