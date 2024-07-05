#include <iostream>
#include <math.h>
#include<iomanip>
using namespace std;

#define width 32
#define IP_SZ 8128
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


time_steps : for(int n=0;n<254;n++)
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



void lstm_network(float* input_x,   float* weights_f,   float* weights_h_f,   float* bias_f,   float* weights_f_rev,   float* weights_h_f_rev,   float* bias_f_rev,   float* weights_icg,   float* weights_h_icg,   float* bias_icg,   float* weights_icg_rev,   float* weights_h_icg_rev,   float* bias_icg_rev,    float* weights_gg,   float* weights_h_gg,   float* bias_gg,    float* weights_gg_rev,   float* weights_h_gg_rev,   float* bias_gg_rev,   float* weights_out,   float* weights_h_out,   float* bias_out,    float* weights_out_rev,   float* weights_h_out_rev,   float* bias_out_rev, float* hidden_curr,   float* hidden_curr_rev){

#pragma HLS INTERFACE m_axi port = input_x depth = width*254

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

	float rev_ip[8128];
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
