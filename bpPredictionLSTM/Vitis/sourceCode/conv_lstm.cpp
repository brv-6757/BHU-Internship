#include<stdio.h>
#include<iostream>
#include<iomanip>
#include<math.h>
using namespace std;
#define OP 8128 // buffer to take ip and send output
#define IP_SZ 1024 //Initial input size 1D
#define POOL_SZ 2 //Max pooling size
#define CONV_SZ 3 //Convolution filter size
#define L1_FILTERS 16 //no filters at layer1
#define L2_FILTERS 32 //no of filters at layer2
#define KERNEL1 CONV_SZ*L1_FILTERS // kernel1 filter_size*no of filters
#define KERNEL2 CONV_SZ*L1_FILTERS*L2_FILTERS //kernel2 filter_size*depth*no of filters
#define L1_OP 511 //1st layer output
#define L2_OP 254 //2nd layer output
#define DEPTH L1_FILTERS //input depth for second layer
#define OP1 L1_OP*L1_FILTERS //1D output size of layer1
#define MAX_OP L2_OP*L2_FILTERS //1D output size of layer1
#define MAX(a,b) (a>b)? a:b //MAX macro function


void conv1d_sep_ip(float* input,float* k1,float* k2,float* bias1,float* bias2,float *output)
{

//	printf("%f\n",L2_OP);
//	cout<<L2_OP<<endl;


//	int op_size = (ip_size-conv_size+1)/max_kern;
		float output1[OP1];
	//	float output[MAX_OP];
		float ans = -40;
		float conv = 0;
		float ind_conv = 0;
		l1_nfilters:for(int n=0;n<L1_FILTERS;n++)
		{
	#pragma HLS UNROLL off = true
		l1_op:for(int i=0;i<L1_OP;i++) // loop for iterating through output
		{
	#pragma HLS UNROLL off=true
			ans = -40.0;
			l1_pool:for(int j=0;j<POOL_SZ;j++) //loop for max_pooling
			{
	#pragma HLS UNROLL
				conv = 0; //variable to store conv result
					l1_conv:for(int l=0;l<CONV_SZ;l++) //loop to perform conv with filter
					{
	#pragma HLS UNROLL

						conv = conv + input[i*POOL_SZ+j+l]*k1[n*CONV_SZ+l];
						//printf("%d,%d,%d %d,%d : %f,%f  ",n,i,j,i*POOL_SZ+j+l,n*CONV_SZ+l,buffer[i*POOL_SZ+j+l],k1[n*CONV_SZ+l]);
					}

					//printf("  %f\n",conv);
					conv = conv+bias1[n];
					if(conv<0) conv = conv*0.3;
				ans = MAX(conv,ans);
			}
			output1[n*L1_OP+i] = ans;//printf("%f\n",ans);
		}
		}

		l2_nfilters:for(int n=0;n<L2_FILTERS;n++){
		l2_op:for(int i=0;i<L2_OP;i++) // loop for iterating through output
			{
	#pragma HLS UNROLL off = true
	#pragma HLS PIPELINE off = true
				ans = -40.0;
				l2_pool:for(int j=0;j<POOL_SZ;j++) //loop for max_pooling
				{
	#pragma HLS PIPELINE off = true
					conv = 0; //variable to store conv result
					l2_depth:for(int d=0;d<DEPTH;d++)
					{
	#pragma HLS UNROLL
						ind_conv=0; //variable to accumulate the conv output of each layer
						l2_conv:for(int l=0;l<CONV_SZ;l++) //loop to perform conv with filter
						{
	#pragma HLS UNROLL
//							if(n==2 && i==0) printf("%d,%d,%d,%d,%d   %f  %f , %f\n",n,i,j,d,l,ind_conv,output1[i*POOL_SZ+j+L1_OP*d+l],k2[n*KERNEL1+CONV_SZ*d+l]);
							ind_conv = ind_conv + output1[i*POOL_SZ+j+L1_OP*d+l]*k2[n*KERNEL1+CONV_SZ*d+l];

						}
						conv = conv+ind_conv;

					}
					conv = conv+bias2[n];
					if(conv<0) conv = conv*0.30000;
					ans = MAX(conv,ans);
				}
//				if(n==2 && i==0) printf("ans = %f\n",ans);

				output[n*L2_OP+i] = ans;
//				if(n==2 && i==0) printf("%d,%d,%d,%d,%d  array %f\n",(n*L2_OP+i),n,i,L2_OP,n*L2_OP,output[508]);

			}
		}

}






















#define width 32
//#define IP_SZ 8128
void LSTM_cell(float* input_x,   float* weights_f,   float* weights_h_f,   float* bias_f,   float *weights_icg,   float* weights_h_icg,   float* bias_icg,    float* weights_gg,   float* weights_h_gg,   float* bias_gg,   float* weights_out,   float* weights_h_out,   float* bias_out, float *hidden_curr){

	int idx=0;
	int idx1=0;
	//depth of long_mem, short_mem, forget_gate,hidden_state , bias_depth should be equal
	float hidden_prev[width];
	float long_prev[width];
	set_to_0 : for(int i=0;i<width;i++)
	{
#pragma HLS UNROLL off=true
		hidden_prev[i] = 0;
		long_prev[i] = 0;
	}



//	float 	   mat_weights_f[width][width];
//	float 	   mat_weights_icg[width][width];
//	float 	   mat_weights_gg[width][width];
//	float 	   mat_weights_out[width][width];
//
//	float 	   mat_weights_h_f[width][width];
//	float 	   mat_weights_h_icg[width][width];
//	float 	   mat_weights_h_gg[width][width];
//	float 	   mat_weights_h_out[width][width];


	float      forget[width];
	float 	      icg[width];	//filter gate for gg
	float 	       gg[width];
//	float  	 temp_add;
	float 	long_temp;
	float  		  out[width];
//	float 	tanh_long[width];

//Converting weights to 2D matrix (as buffer ) for matrix multiplication

//	convert2d: for (int i = 0; i < width; i++) {
//	 	    convert2din:	for(int j = 0 ; j < width ; j++) {
//	 	    			mat_weights_f[i][j]	   =	 weights_f[width * i + j ] ;
//	 	    			mat_weights_f[i][j]	   =	 weights_f[width * i + j ] ;
//	 	    			mat_weights_icg[i][j]  =   weights_icg[width * i + j ] ;
//	 	    			mat_weights_gg[i][j]   =    weights_gg[width * i + j ] ;
//	 	    			mat_weights_out[i][j]  =   weights_out[width * i + j ] ;
//	 	    			mat_weights_h_f[i][j]  =   weights_h_f[width * i + j ] ;
//	 	    			mat_weights_h_icg[i][j]= weights_h_icg[width * i + j ] ;
//	 	    			mat_weights_h_gg[i][j] =  weights_h_gg[width * i + j ] ;
//	 	    			mat_weights_h_out[i][j]= weights_h_out[width * i + j ] ;
//	 	    		}
//	 	  }


time_steps : for(int n=0;n <254;n++)
{
 main:	for(int i = 0; i < width; i++){
	 //i < long_curr_depth

	 	 	forget[i] =    bias_f[i];
	 		icg[i]	  =  bias_icg[i];
	 		gg[i]	  =   bias_gg[i];
	 		out[i]	  =  bias_out[i];
		// j < hiddn_prev_depth + input_x_depth(concatenate hidden_prev and input_x)
		// calculating dense layer --> matrix multiplication

//		input_hidden : for(int j = 0; j < width; j++){
//	#pragma HLS UNROLL off=true
//			idx = j+n*width;
//					forget[i] += input_x[idx] * mat_weights_f[idx1] + hidden_prev[j] * mat_weights_h_f[idx1];
//					icg[i]	  += input_x[idx] * mat_weights_icg[idx1] + hidden_prev[j] * mat_weights_h_icg[idx1];
//					gg[i]	  += input_x[idx] * mat_weights_gg[idx1] + hidden_prev[j] * mat_weights_h_gg[idx1];
//					out[i]	  += input_x[idx] * mat_weights_out[idx1] + hidden_prev[j] * mat_weights_h_out[idx1];
//				}
	 		input_hidden : for(int j = 0; j < width; j++){
	 			#pragma HLS UNROLL off=true
	 					idx = j+n*width;
	 					idx1 = j*width+i;
	 							forget[i] += input_x[idx] * weights_f[idx1] + hidden_prev[j] * weights_h_f[idx1];
	 							icg[i]	  += input_x[idx] * weights_icg[idx1] + hidden_prev[j] * weights_h_icg[idx1];
	 							gg[i]	  += input_x[idx] * weights_gg[idx1] + hidden_prev[j] * weights_h_gg[idx1];
	 							out[i]	  += input_x[idx] * weights_out[idx1] + hidden_prev[j] * weights_h_out[idx1];
	 						}
//		hidden : for(int j = 0; j < width; j++){
//	#pragma HLS UNROLL off=true
//			forget[i] += hidden_prev[j] * mat_weights_h_f[idx1];
//			icg[i]	  += hidden_prev[j] * mat_weights_h_icg[idx1];
//			gg[i]	  += hidden_prev[j] * mat_weights_h_gg[idx1];
//			out[i]	  += hidden_prev[j] * mat_weights_h_out[idx1];
//		}
		//cout<<setprecision(20)<<forget[i]<<" ";



		//cout<< forget[i]<<" ";

		// adding bias


		//Applying Activation function


		forget[i] = 1/(1+ (exp(-forget[i])));							//sigmoid
		icg[i]    = 1/(1+ (exp(-icg[i])));	 							//sigmoid
//		gg[i]     = (1 - exp(-(2 * gg[i])))/(1 + exp(-(2 * gg[i])));	//tanh
		gg[i]     = tanh(gg[i]);
		out[i]	  = 1/(1+ (exp(-out[i])));								//sigmoid

		long_temp = forget[i] * long_prev[i];	// forgetting/deleting from LSTM-long term mem
				 //cout<< long_temp[i] << "  = " << forget[i] << " * " << long_prev[i]<<"\n";
				// Adding new word to LSTM-long term mem

		//		 temp_add = icg[i]	    * 		 gg[i];
				long_prev[i] = long_temp +  icg[i] * gg[i];
				// cout<< long_curr[i] << "  = " << long_temp[i] << " + " << temp_add[i]<<"\n";


		//		tanh_long[i] = tanh(long_prev[i]);
		//		cout<<tanh_long[i]<<" "<< long_curr[i]<< " ";
		//		cout<<exp(-(2 * long_curr[i]))<<" \n"<<endl;
				hidden_prev[i] = (out[i] * tanh(long_prev[i]));
		}



	//	calculating updated value of long term memory

//	 hidden_long_curr : for(int i = 0; i < width; i++){
//
//
//	}


//	hiddencurr : for(int i = 0; i< width; i++){
//		hidden_prev[i] = (out[i] * tanh_long[i]);
//		// cout<< hidden_curr[i] << " = " << out[i] <<" * "<< tanh_long[i]<<"\n"<<endl;
//	}

		//hidden_prev = hidden_curr;
		//long_prev = long_curr;
}

copy_to_curr : for(int i=0;i<width;i++)
{
#pragma HLS UNROLL off=true
	hidden_curr[i] = hidden_prev[i];
}

}



void conv_lstm(float* input_conv, float* k1, float* k2, float * bias1, float* bias2, float* weights_f, float* weights_h_f,   float* bias_f,   float* weights_f_rev,   float* weights_h_f_rev,   float* bias_f_rev,   float* weights_icg,   float* weights_h_icg,   float* bias_icg,   float* weights_icg_rev,   float* weights_h_icg_rev,   float* bias_icg_rev,    float* weights_gg,   float* weights_h_gg,   float* bias_gg,    float* weights_gg_rev,   float* weights_h_gg_rev,   float* bias_gg_rev,   float* weights_out,   float* weights_h_out,   float* bias_out,    float* weights_out_rev,   float* weights_h_out_rev,   float* bias_out_rev, float* hidden_curr,   float* hidden_curr_rev){

#pragma HLS INTERFACE m_axi port = input_conv depth = IP_SZ

#pragma HLS INTERFACE m_axi port=k1 depth=KERNEL1
	#pragma HLS INTERFACE m_axi port=k2 depth=KERNEL2
	#pragma HLS INTERFACE m_axi port=bias1 depth=L1_FILTERS
	#pragma HLS INTERFACE m_axi port=bias2 depth=L2_FILTERS

#pragma HLS INTERFACE m_axi port = weights_f depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_f depth = width*width
#pragma HLS INTERFACE m_axi port = bias_f depth = width

#pragma HLS INTERFACE m_axi port = weights_icg depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_icg depth = width*width
#pragma HLS INTERFACE m_axi port = bias_icg depth = width

#pragma HLS INTERFACE m_axi port = weights_gg depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_gg depth = width*width
#pragma HLS INTERFACE m_axi port = bias_gg depth = width

#pragma HLS INTERFACE m_axi port = weights_out depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_out depth = width*width
#pragma HLS INTERFACE m_axi port = bias_out depth = width

#pragma HLS INTERFACE m_axi port = hidden_curr depth = width


#pragma HLS INTERFACE m_axi port = weights_f_rev depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_f_rev depth = width*width
#pragma HLS INTERFACE m_axi port = bias_f_rev depth = width

#pragma HLS INTERFACE m_axi port = weights_icg_rev depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_icg_rev depth = width*width
#pragma HLS INTERFACE m_axi port = bias_icg_rev depth = width

#pragma HLS INTERFACE m_axi port = weights_gg_rev depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_gg_rev depth = width*width
#pragma HLS INTERFACE m_axi port = bias_gg_rev depth = width

#pragma HLS INTERFACE m_axi port = weights_out_rev depth = width*width
#pragma HLS INTERFACE m_axi port = weights_h_out_rev depth = width*width
#pragma HLS INTERFACE m_axi port = bias_out_rev depth = width

#pragma HLS INTERFACE m_axi port = hidden_curr_rev depth = width

#pragma HLS INTERFACE mode=s_axilite bundle=CTRL_BUS port=return

	float input_x[8128];
	float rev_ip[8128];
	float output[8128];
	int idx =0;

	conv1d_sep_ip(input_conv, k1, k2, bias1, bias2, output);

	out_to_in : for(int i = 0;i < 254; i++)
	{
		for(int j = 0;j < 32; j++)
		{
			input_x[idx++] = output[j * 254 + i];
		}
	}
//	for(int i = 0;i<8128;i++){
//		cout<<input_x[i]<< "  "<< output[i]<<endl;
//	}

	input_rev : for(int i=0;i<254;i++)
	{
		for(int j=0;j<32;j++) rev_ip[i*32+j] = input_x[(253-i)*32+j];
	}


	LSTM_cell(input_x,  weights_f,  weights_h_f, bias_f, weights_icg, weights_h_icg, bias_icg,weights_gg, weights_h_gg, bias_gg, weights_out, weights_h_out, bias_out, hidden_curr);
//	for(int i=0;i<32;i++) cout<<setprecision(20)<<hidden_curr[i]<<endl;
//	cout<<endl;
	LSTM_cell( rev_ip,  weights_f_rev,  weights_h_f_rev, bias_f_rev, weights_icg_rev, weights_h_icg_rev, bias_icg_rev, weights_gg_rev, weights_h_gg_rev, bias_gg_rev,  weights_out_rev, weights_h_out_rev, bias_out_rev, hidden_curr_rev);
//	for(int i=0;i<32;i++) cout<<setprecision(20)<<hidden_curr_rev[i]<<endl;
//	cout<<endl;
}





