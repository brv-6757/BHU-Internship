#include<iostream>
using namespace std;
void dense(float *input1, float *input2, float *output);
int main()
{
	float input1[64] = {
			0.9965884685516357,0.9953402280807495,-0.9976345896720886,0.9980317950248718,-0.9946983456611633,0.9959136843681335,0.995067834854126,0.9957820177078247,0.9968215227127075,0.9956980347633362,0.9954906702041626,-0.9958710074424744,0.995869517326355,0.9981555938720703,0.9955520629882812,0.9974410533905029,0.9919921159744263,0.9985225200653076,0.9966452717781067,-0.9969984889030457,0.9949290752410889,-0.995833694934845,0.9956985712051392,-0.995528519153595,0.9936937689781189,-0.9981105923652649,-0.9973344802856445,-0.9943596124649048,0.9959948658943176,0.994669497013092,0.9964267015457153,0.9945480227470398,-0.9941883087158203,-0.9932186007499695,0.9976599812507629,0.9933790564537048,-0.9983279705047607,-0.9948232769966125,0.9975493550300598,0.9971052408218384,0.9978008270263672,0.9932178854942322,-0.994038999080658,-0.990826427936554,0.9980947375297546,-0.9944567680358887,0.9952878355979919,-0.9866926670074463,0.991095244884491,0.9923940896987915,0.9968992471694946,0.9985838532447815,-0.994303286075592,-0.9979183673858643,0.9938892722129822,-0.9950196743011475,-0.9923300743103027,0.9906014800071716,-0.9955750703811646,-0.9975438117980957,-0.9918872117996216,0.9968681931495667,0.9974219799041748,-0.9917773008346558,
			};
	float input2[64] = {
			-0.7243195176124573,0.8818707466125488,-0.8952701091766357,0.15500882267951965,0.28271085023880005,0.9694435596466064,0.9325430989265442,0.00717275682836771,0.8010897636413574,-0.949138343334198,0.9556844234466553,0.04228132218122482,0.9833584427833557,0.669366717338562,-0.9906228184700012,0.10405009984970093,0.4605194628238678,-0.2572612464427948,0.8018508553504944,-0.03281233832240105,-0.9808116555213928,-0.9972630739212036,0.5662274360656738,0.8387657403945923,0.9256998896598816,-0.9949410557746887,-0.5295265913009644,-4.231560888001695e-05,0.9682772159576416,-0.9593425393104553,0.9778885841369629,-0.2651499807834625,0.9918728470802307,0.9975682497024536,-0.992225706577301,-0.9962953329086304,0.9927908182144165,0.9945339560508728,-0.11012744158506393,0.9863577485084534,0.9959632754325867,-0.2747885584831238,-0.9931154251098633,0.9952908158302307,-0.9904733300209045,-0.998345673084259,0.9973673820495605,0.9978863596916199,-0.9897728562355042,0.9936492443084717,0.9966548681259155,-0.9965894818305969,0.9946702122688293,-0.9956621527671814,-0.9901623725891113,-0.9981792569160461,0.9940491914749146,-0.9947652220726013,0.9951426386833191,0.9971879720687866,-0.9935596585273743,-0.9953718185424805,-0.9928273558616638,0.9979473352432251,
			};
	float output[2];
	dense(input1, input2, output);
	cout<<output[0]<<" "<<output[1];
	return 0;
}


