#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define input  7
#define hidden1  20
#define hidden2  10
#define output 1
#define numPatterns 1999
#define numEpochs 2000
#define stoping_error 0.01
#define lR 0.01
#define n 10

int validationnum = (int)(numPatterns/5);
int testnum = (int)(numPatterns/7);

double trainInputs[numPatterns][input] = {0};
double trainOutput[numPatterns][output] = {0};
double validationinput[numPatterns][input]= {0};
double validationoutput[numPatterns][input]= {0};
double testinput[numPatterns][input] = {0};
double testoutput[numPatterns][output] = {0};


void initData()
{
    FILE *fp,*fout,*fvalidate,*ftest,*asdf;
    double inp[7],op;
    int count=0,i,j;
    double inputset1[numPatterns],inputset2[numPatterns],inputset3[numPatterns],outputset[numPatterns]={0};
    double *minmax1,*minmax2,*minmax3,*minmaxout;

    fp = fopen("training_input", "r");
    fout = fopen("training_output", "r");

    printf("Init inputs");

    //fp = fopen ("f1.txt","w");
    fvalidate = fopen ("vali.txt","w");
    ftest = fopen ("test.txt","w");
    asdf = fopen ("train.txt","w");

    while(!feof(fp))
    {

        fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",&inp[0],&inp[1],&inp[2],&inp[3],&inp[4],&inp[5],&inp[6]);
        fscanf(fout,"%lf\n",&op);
        //printf("%d %lf %lf %lf %lf\n",count, inp[0],inp[1],inp[2],op);
        //printf("%lf",count/5);
        if (count%5 == 0){
        printf("hel %d\n",count);
        validationinput[count][0] = inp[0];
        validationinput[count][1] = inp[1];
        validationinput[count][2] = inp[2];
        validationinput[count][3] = inp[3];
        validationinput[count][4] = inp[4];
        validationinput[count][5] = inp[5];
        validationinput[count][6] = inp[6];
        validationoutput[count][0] = op;
        fprintf(fvalidate,"%lf %lf %lf %lf %lf %lf %lf\n",inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6]);
        }

        else if(count%7 == 0){
        testinput[count][0] = inp[0];
        testinput[count][1] = inp[1];
        testinput[count][2] = inp[2];
        testinput[count][3] = inp[3];
        testinput[count][4] = inp[4];
        testinput[count][5] = inp[5];
        testinput[count][6] = inp[6];
        testoutput[count][0] = op;
        fprintf(ftest,"%lf %lf %lf %lf %lf %lf %lf\n",inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6]);


        }

        else{
        trainInputs[count][0] = inp[0];
        trainInputs[count][1] = inp[1];
        trainInputs[count][2] = inp[2];
        trainInputs[count][3] = inp[3];
        trainInputs[count][4] = inp[4];
        trainInputs[count][5] = inp[5];
        trainInputs[count][6] = inp[6];
        trainOutput[count][0] = op;
        fprintf(asdf,"%lf %lf %lf %lf %lf %lf %lf\n",inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6]);

        }
        //printf("%d %lf %lf %lf %lf %lf %lf %lf %lf\n",count, inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6],op);
        //fprintf(pFile,"%d %lf %lf %lf %lf\n",count, inputset1[count],inputset2[count],inputset3[count],outputset[count]);
        count++;
        if(count>numPatterns)
        {
            printf("%d\n",count);
            break;
        }

    }

    fclose(fp);
    fclose(fout);
    fclose(fvalidate);
    fclose(ftest);
    fclose(asdf);

    //minmax1 = minmax(inputset1);
//    minmax2 = minmax(inputset2);
//    minmax3 = minmax(inputset3);
//    minmaxout = minmax(outputset);

    //printf(" %lf %lf\n",minmax1[0],minmax1[1]);
//    printf(" %lf %lf\n",minmax2[0],minmax2[1]);
//    printf(" %lf %lf\n",minmax3[0],minmax3[1]);
//
//    for(i=0; i<numPatterns; i++)
//    {
//
//        trainInputs[i][0] = (trainInputs[i][0] - minmax1[1])/(minmax1[1]-minmax1[0]);
//        trainInputs[i][1] = (trainInputs[i][1] - minmax2[1])/(minmax2[1]-minmax2[0]);
//        trainInputs[i][2] = (trainInputs[i][2] - minmax3[1])/(minmax3[1]-minmax3[0]);
//        trainOutput[i][0] = (trainOutput[i][0] - minmaxout[1])/(minmaxout[1]-minmaxout[0]);
//
//        //printf("%d %lf %lf %lf %lf\n",i, trainInputs[i][0],trainInputs[i][1],trainInputs[i][2],trainOutput[i][0]);
//
//    }




}

int main(){


initData();
return 0;
}
