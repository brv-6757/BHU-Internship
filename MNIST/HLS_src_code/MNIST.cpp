#include<iostream>
#include<math.h>
#include<algorithm>
using namespace std;



void convolution28_26(float image[28][28], float filter[3][3], float output_image[26][26])
{
	for(int i = 1; i < 27; i++)
	{
		for(int j = 1; j < 27; j++)
		{
			float convolved = 0;
			convolved += (image[i - 1][j - 1] * filter[0][0] +
                        image[i - 1][j] * filter[0][1] +
                        image[i - 1][j + 1] * filter[0][2] +
                        image[i][j - 1] * filter[1][0] +
                        image[i][j] * filter[1][1] +
                        image[i][j + 1] * filter[1][2] +
                        image[i + 1][j - 1] * filter[2][0] +
                        image[i + 1][j] * filter[2][1] +
                        image[i + 1][j + 1] * filter[2][2]);
			if(convolved < 0) convolved = 0;
			if(convolved > 255) convolved = 255;
			output_image[i - 1][j - 1] = convolved +  0.0034502416383475065;
		}
	}
}

void maxpooling26_13(float input_image[26][26], float output_image[13][13])
{
	int op_row_idx = 0;
	int op_col_idx = 0;
	for(int i = 0; i < 26; i += 2)
	{
		op_col_idx = 0;
		for(int j = 0; j < 26; j += 2)
		{
			output_image[op_row_idx][op_col_idx] =
					max(max(input_image[i][j],input_image[i + 1][j]), max(input_image[i][j + 1], input_image[i + 1][j + 1]));
			op_col_idx++;
		}
		op_row_idx++;
	}
}

void convolution13_11(float image[13][13], float filter[3][3], float output_image[11][11])
{
	for(int i = 1; i < 12; i++)
	{
		for(int j = 1; j < 12; j++)
		{
			float convolved = 0;
			convolved += (image[i - 1][j - 1] * filter[0][0] +
                        image[i - 1][j] * filter[0][1] +
                        image[i - 1][j + 1] * filter[0][2] +
                        image[i][j - 1] * filter[1][0] +
                        image[i][j] * filter[1][1] +
                        image[i][j + 1] * filter[1][2] +
                        image[i + 1][j - 1] * filter[2][0] +
                        image[i + 1][j] * filter[2][1] +
                        image[i + 1][j + 1] * filter[2][2]);
			if(convolved < 0) convolved = 0;
			if(convolved > 255) convolved = 255;
			output_image[i - 1][j - 1] = convolved;
		}
	}
}

void maxpooling11_5(float input_image[11][11], float output_image[5][5])
{
	int op_row_idx = 0;
	int op_col_idx = 0;
	for(int i = 0; i < 10; i += 2)
	{
		op_col_idx = 0;
		for(int j = 0; j < 10; j += 2)
		{
			output_image[op_row_idx][op_col_idx] =
					max(max(input_image[i][j],input_image[i + 1][j]), max(input_image[i][j + 1], input_image[i + 1][j + 1]));
			op_col_idx++;
		}
		op_row_idx++;
	}
}

void mnist_2(int *input,int *output)
{
	#pragma HLS INTERFACE mode = m_axi port = input depth = 784
	#pragma HLS INTERFACE mode = m_axi port = output depth = 1
	#pragma HLS INTERFACE mode = s_axilite port = return

	float image[28][28];
	for(int i = 0; i < 28; i++)
	{
		for(int j = 0; j < 28; j++)
		{
			image[i][j] = (input[i * 28 + j])/255.0;
		}
	}

	float filter_1[3][3] = {{ 1.1635185,   0.1642524,  -0.21736963},
 							{ 1.4468029,   1.2773782,   1.143021 },
 							{0.4278373,   0.6104295,   0.9171918 }};
	float first_layer_image[26][26];
	convolution28_26(image, filter_1, first_layer_image);

	float second_layer_image[13][13];
	maxpooling26_13(first_layer_image, second_layer_image);

	float filter_2[3][3] = {{-0.07698992,  0.5034029,   0.3484637 },
 							{ 0.57425666,  0.4974812,   0.24629028},
 							{-0.08710181, -0.17777562,  0.7530503 }};
	float filter_3[3][3] = {{-0.7302613,  -0.29056314, -0.8340109 },
 							{ 0.22477376, -0.46012846,  0.22821836},
 							{ 0.7908042,   0.29400975,  0.4697234 }};
	float third_layer_image_1[11][11];
	float third_layer_image_2[11][11];

	convolution13_11(second_layer_image,filter_2, third_layer_image_1);

    for(int i = 0; i < 11; i++)
    {
    	for(int j = 0; j < 11; j++)
    	{
    		third_layer_image_1[i][j] += (-0.1107659861445427);
    		if(third_layer_image_1[i][j] < 0) third_layer_image_1[i][j] = 0;
		}
	}
	convolution13_11(second_layer_image,filter_3, third_layer_image_2);

	for(int i = 0; i < 11; i++)
    {
    	for(int j = 0; j < 11; j++)
    	{
    		third_layer_image_2[i][j] += (-0.002924372674897313);
    		if(third_layer_image_2[i][j] < 0) third_layer_image_2[i][j] = 0;
		}
	}

	float fourth_layer_image_1[5][5];
	float fourth_layer_image_2[5][5];

	maxpooling11_5(third_layer_image_1, fourth_layer_image_1);
	maxpooling11_5(third_layer_image_2, fourth_layer_image_2);


	float flatten_layer[50];
	int it = 0;
	for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 5; j++)
		{
			flatten_layer[it] = fourth_layer_image_1[i][j];
			it++;
			flatten_layer[it] = fourth_layer_image_2[i][j];
			it++;
		}
	}

float neuron0_in[50] = {0.14331928, 0.5266376, 0.13922833, 0.4847004, 0.3586352, 0.4313449, 0.09371827, 0.49055603, 0.051070247, 0.008914519, -0.13200869, 0.37794974, -0.075905725, 0.5204634, -0.2603485, 0.08087567, 0.21212512, 0.21603288, 0.21337031, 0.46008772, -0.58314687, 0.06797363, -0.098192, 0.08230187, 0.03182268, 0.058935996, -0.26162502, 0.26651743, -0.13822395, 0.18227822, -0.038760494, 0.7725733, 0.104363285, -0.112716995, 0.043701466, 0.03689389, -0.3787963, -0.04381409, -0.08337801, 0.50899166, 0.32215738, -0.51631063, 0.07187317, 0.60490215, -0.41673928, 1.1233478, 0.31116706, 0.4681724, 0.001202451, -0.22942512};
float neuron1_in[50] = {-0.05688115, 0.68517923, 0.091472566, 0.16607212, -0.07801943, 0.37880966, -0.14367716, 0.25461558, 0.053958032, 0.47226226, -0.27238196, -0.24818504, -0.059115607, -0.051164515, -0.22940984, -0.16299032, 0.05807604, -0.05151214, 0.003156703, 0.13435182, 0.093434535, -0.6275576, 0.54327977, -0.5953245, 0.3290967, 0.1725946, 0.29361525, 0.001320407, -0.14459902, 0.1438719, 0.37944674, 0.34364155, 0.108427055, 0.49960592, 0.05156267, 0.14056157, 0.16861969, 0.14345248, -0.3593537, -0.20242538, -0.325396, -0.058789365, -0.55266696, 0.7447664, -0.49801525, 0.4817179, 0.10790747, 0.42316633, -0.3683127, -0.643156};
float neuron2_in[50] = {-0.17736642, 0.104413256, 0.18268122, 0.74650794, 0.37064126, 0.1880746, 0.14257437, -0.024345843, -0.038127415, 0.11984753, 0.31407955, 0.66379446, -0.1403978, -0.0911178, 0.03268238, 0.33040667, 0.08172666, 0.058543514, -0.03986859, -0.22751376, -0.2576078, 0.28162637, -0.34060457, -0.1900192, 0.47003055, 0.44234544, 0.153612, 0.17683992, 0.20113857, -0.3127701, -0.06503758, 1.1750768, -0.43485525, 0.22099969, 0.3526877, -0.570533, 0.03686365, -0.006854794, -0.5273006, 0.5101487, 0.008412099, 0.32880944, -0.24871549, -0.1405568, 0.2205843, 0.23815882, -0.52288693, 0.5067521, 0.1141332, 0.16377595};
float neuron3_in[50] = {0.07807213, -0.050314695, -0.112148605, 0.32036817, 0.213227, -0.3085307, -0.20699556, 0.034906317, -0.16441308, 0.28318745, -0.10746589, 0.05012843, -0.1047429, 0.22098416, -0.31821185, 0.30150062, 0.18841574, -0.18230228, 0.040767718, -0.04446861, -0.14608927, -0.25595015, -0.12864336, -0.27448845, -0.28607488, -0.07842427, 0.20449293, 0.009657741, -0.16631179, -0.030871142, 0.27729985, -0.023371194, 0.07083235, 0.28224146, 0.107148506, -0.31074122, 0.078628436, -0.27608907, -0.30100188, -0.26898617, 0.060366284, 0.00864017, -0.19091742, -0.099398404, -0.31369942, 0.061196327, -0.13915679, -0.182149, 0.013026703, 0.24341136 };
float neuron4_in[50] = {-0.11926256, 0.23960827, -0.13300177, 0.2430994, -0.19890845, -0.13176663, -0.1292667, 0.26773652, -0.110946395, 0.024506899, -0.13057372, 0.07779554, -0.23950419, 0.2115388, -0.07741631, -0.19781457, 0.13690342, 0.17074753, -0.2166603, 0.2096451, -0.28659308, 0.16972785, -0.08901659, -0.24479152, -0.37075824, 0.10719671, -0.26832166, 0.04806182, 0.11874475, 0.17421947, -0.019929899, -0.22153014, 0.20040825, -0.116288766, -0.004554459, 0.11630486, -0.23183371, -0.04109045, -0.005973787, 0.10364057, -0.3253953, -0.1863017, -0.1625131, 0.044699863, -0.13256201, -0.256175, -0.08657863, 0.28906143, 0.028403323, 0.02915571};
float neuron5_in[50] = {0.16043411, -0.25205323, -0.1950993, -0.12146988, -0.14573647, -0.09424447, 0.10411161, -0.01509318, 0.1986763, 0.19414061, -0.04463734, -0.4314622, 0.27781343, -0.38651106, -0.11946956, -0.42905802, -0.2081102, 0.06720913, 0.08124305, 0.3941533, 0.16546719, -0.8755066, 0.37932447, -0.4245405, 0.37971357, 0.14093246, -0.14627789, 0.46607283, -0.14153555, 0.5281776, 0.15844022, 0.56813794, 0.09195824, 0.27155113, 0.3332983, -0.39110383, -0.23417582, 0.0831782, -0.5822268, 0.23727526, 0.079025276, -0.13208891, 0.06645857, 0.03271004, -0.029021954, 0.7433112, 0.10267386, 0.56568074, 0.24283206, -0.06813275};
float neuron6_in[50] = {0.23011573, 0.2748482, -0.11080569, -0.17463796, 0.20274156, -0.19857357, 0.19387674, -0.12616083, 0.26670226, -0.04969871, -0.007785688, 0.27096263, 0.17392267, -0.10516982, -0.3315407, 0.07354474, -0.16218403, -0.10952523, -0.28912136, 0.15445204, 0.15886094, -0.07978876, -0.21384388, -0.012426112, -0.13799323, -0.032188382, -0.28777304, -0.30571052, -0.21191515, -0.032123566, -0.05657447, -0.0554699, -0.05624379, 0.2528597, -0.11158932, -0.24572584, 0.009378204, 0.07031464, -0.09538886, -0.019935187, 0.00036844632, 0.29415447, -0.0854766, 0.20248483, 0.18815896, 0.10101745, -0.27474555, 0.13257304, -0.030140618, -0.17146395};
float neuron7_in[50] = {-0.15842283, -0.14268592, 0.051027346, 0.16241434, -0.22281927, -0.22198239, 0.08778612, -0.2586485, -0.20156777, 0.08198565, -0.14736642, -0.003676275, -0.1631814, -0.03719401, 0.157712, 0.23053741, 0.12453392, -0.28532165, -0.26720333, -0.09413434, 0.20437403, -0.27097267, 0.15180999, -0.09915824, 0.01535248, 0.10882372, -0.17822962, 0.110666275, -0.28345174, -0.2985653, 0.17901935, 0.21018004, 0.042964876, 0.10737318, -0.30075088, 0.06187929, -0.18430732, -0.16135825, 0.28416955, -0.19175014, 0.120979026, -0.10194735, 0.005747288, 0.07286471, -0.29465684, -0.10472416, -0.23591575, -0.2311781, -0.2644514, 0.25028527};

	float neuron_op[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	for(int i = 0; i < 50; i++)
	{
		neuron_op[0] += (flatten_layer[i] * neuron0_in[i]);
		neuron_op[1] += (flatten_layer[i] * neuron1_in[i]);
		neuron_op[2] += (flatten_layer[i] * neuron2_in[i]);
		neuron_op[3] += (flatten_layer[i] * neuron3_in[i]);
		neuron_op[4] += (flatten_layer[i] * neuron4_in[i]);
		neuron_op[5] += (flatten_layer[i] * neuron5_in[i]);
		neuron_op[6] += (flatten_layer[i] * neuron6_in[i]);
		neuron_op[7] += (flatten_layer[i] * neuron7_in[i]);
	}

	float biases_hidden[8] = {-0.9117920398712158, 0.28616219758987427, -0.2608160972595215, -0.013337160460650921, -0.04107463359832764, 0.47512897849082947, -0.02162751369178295, -0.002048331778496504};
	for(int i = 0; i < 8; i++)
	{
		neuron_op[i] += biases_hidden[i];
		if(neuron_op[i] < 0) neuron_op[i] = 0;
	}

float n0_weights[8] = {0.4130056, -0.34275252, -0.76931417, 0.4307383, -0.439536, -0.04301196, -0.27752024, 0.28659597};
float n1_weights[8] = {-0.76404583, -0.71734756, 0.7140804, -0.4580981, 0.46828252, -0.08306392, 0.51266056, 0.012444746};
float n2_weights[8] = {0.22494334, -0.936014, 0.2318033, 0.2042866, -0.5014504, -0.29828203, -0.3687686, 0.30333784};
float n3_weights[8] = {0.17704248, -0.01591697, 0.12568651, 0.45588356, -0.17092243, -0.39505404, 0.29319072, 0.18386103};
float n4_weights[8] = {-0.95692337, 0.55792654, -0.50614417, 0.14481695, 0.23780902, 0.28523952, -0.47033265, 0.46070245};
float n5_weights[8] = {-0.082555406, 0.13251354, -0.48571923, 0.33638403, -0.5251565, 0.35943094, -0.12513414, -0.57471603};
float n6_weights[8] = {-0.3895531, -0.56730974, -0.7906231, 0.2957038, 0.31567565, 0.95464003, -0.04737595, -0.0057591507};
float n7_weights[8] = {-0.061407354, 0.29449022, 0.2544075, 0.18216869, 0.016241897, -1.2875234, 0.2583854, 0.11593165};
float n8_weights[8] = {-0.21868607, -0.23992896, 0.19710661, 0.43130237, -0.07376745, 0.18971865, 0.46283087, 0.48478046};
float n9_weights[8] = {-0.29944, 0.67111325, -0.13763598, -0.26430634, -0.40522504, -0.45182148, 0.24348679, -0.5624977};


	float output_layer[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	for(int i = 0; i < 8; i++)
	{
		output_layer[0] += (neuron_op[i] * n0_weights[i]);
		output_layer[1] += (neuron_op[i] * n1_weights[i]);
		output_layer[2] += (neuron_op[i] * n2_weights[i]);
		output_layer[3] += (neuron_op[i] * n3_weights[i]);
		output_layer[4] += (neuron_op[i] * n4_weights[i]);
		output_layer[5] += (neuron_op[i] * n5_weights[i]);
		output_layer[6] += (neuron_op[i] * n6_weights[i]);
		output_layer[7] += (neuron_op[i] * n7_weights[i]);
		output_layer[8] += (neuron_op[i] * n8_weights[i]);
		output_layer[9] += (neuron_op[i] * n9_weights[i]);
	}
	float biases_output[10] = {0.5541055202484131, 0.14498095214366913, 1.4624994993209839, -1.8387385606765747, 1.3326361179351807, -0.9932224154472351, 0.2979242503643036, 1.1443182229995728, -0.8530760407447815, -0.3239060342311859};

	for(int i = 0; i < 10; i++)
	{
		output_layer[i] += (biases_output[i]);
	}

	float maxi = output_layer[0];
	cout<<maxi<<endl;
	int idx = 0;
	for(int i = 1; i < 10; i++)
	{
		if(output_layer[i] > maxi)
		{
			maxi = output_layer[i];
			idx = i;
		}
		cout<<output_layer[i]<<endl;
	}

	*output = idx;
	cout<<*output;
}

