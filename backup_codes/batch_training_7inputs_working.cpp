#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define input  7
#define hidden1  20
#define hidden2  10
#define output 1
#define numPatterns 1999
#define numEpochs 5000
#define stoping_error 0.01
#define lR 0.013
#define n 10

double buffer[n]= {0};

int x = 0, y = 0, i = 0, j = 0,patNum = 0;
double weights1[input][hidden1]= {{1,1},{1,1}};
double weights2[hidden1][hidden2]= {{1,1},{1,1}};
double weights3[hidden2][output]= {{1},{1}};

double bias1[hidden1] = {0};
double bias2[hidden2] = {0};
double bias3[output] = {0};

double hiddenVal1[hidden1] = {0};
double hiddenVal2[hidden2] = {0};
double outPred[output] = {0};

double hidden2delta[hidden2] = {0.0};
double hidden1delta[hidden1] = {0.0};

double weightedsum = 0.0;
double output_delta[output] = {0.0};
double inputdelta[input] = {0.0};
double errThisPat[output] = {0.0};
double RMSerror = 0.0;
double netoutputdelta[numPatterns][output]= {0.0};
double nethidden2delta[numPatterns][hidden2]= {0.0};
double nethidden1delta[numPatterns][hidden1]= {0.0};

double dc_dw_hidden1[numPatterns][hidden1] = {0.0};
double dc_dw_hidden2[numPatterns][hidden2] = {0.0};
double dc_dw_output[numPatterns][output] = {0.0};


double trainInputs[numPatterns][input] = {0};
double trainOutput[numPatterns][output] = {0};
double validationinput[numPatterns][input]= {0};
double validationoutput[numPatterns][input]= {0};
double testinput[numPatterns][input] = {0};
double testoutput[numPatterns][output] = {0};


FILE * pFile;

//// functions ////
void initWeights();

void calcNet();

void calcOverallError();

void displayResults();




//***********************************
// calculates the sigmoid of x
double sigmoid(double x)
{
    double exp_value;
    double return_value;

    /*** Exponential calculation ***/
    exp_value = exp((double) -x);

    /*** Final sigmoid value ***/
    return_value = 1 / (1 + exp_value);

    return return_value;
}

///Forward Pass.
void calcNet(void)
{
    ///calculate the outputs of the hidden layer 1 neurons
    int i = 0, j = 0;
    for (i = 0; i < hidden1; ++i)
    {
        hiddenVal1[i] = 0.0 + bias1[i];

        for (j = 0; j < input; ++j)
        {
            hiddenVal1[i] = hiddenVal1[i] + (trainInputs[patNum][j] * weights1[j][i]);

        }
        hiddenVal1[i] = sigmoid(hiddenVal1[i]);
        //printf("Hidden 1 activation\t%lf\n",hiddenVal1[i]);
    }
    //output of hidden layer 2 neurons
    for (i = 0; i < hidden2; ++i)
    {
        hiddenVal2[i] = 0.0 + bias2[i];

        for (j = 0; j < hidden1; ++j)
        {

            hiddenVal2[i] = hiddenVal2[i] + (hiddenVal1[j] * weights2[j][i]);
        }
        hiddenVal2[i] = sigmoid(hiddenVal2[i]);
        //printf("Hidden 2 activation\t%lf\n",hiddenVal2[i]);
        //printf("%lf\t", hiddenVal2[i]);

    }
    //printf("\n");

    //calculate the output of the network

    for (i = 0; i < output; ++i)
    {
        outPred[i] = 0.0 + bias3[0];

        for (j = 0; j < hidden2; ++j)
        {
            outPred[i] = outPred[i] + (hiddenVal2[j] * weights3[j][i]);
        }
        //printf("\n output activation %lf  ", outPred[i]);
        outPred[i] = sigmoid(outPred[i]); /// Thats the individual output layer neuron prediction
        //printf("\n output activation %lf\n", outPred[i]);


    }

    for (i = 0; i < output; ++i)
    {
        errThisPat[i] =  trainOutput[patNum][i] - outPred[i];
        //printf("e %lf\t",errThisPat[i]);
    }


}

void calcNetValidate(void)
{
    ///calculate the outputs of the hidden layer 1 neurons
    int i = 0, j = 0;
    for (i = 0; i < hidden1; ++i)
    {
        hiddenVal1[i] = 0.0 + bias1[i];

        for (j = 0; j < input; ++j)
        {
            hiddenVal1[i] = hiddenVal1[i] + (validationinput[patNum][j] * weights1[j][i]);

        }
        hiddenVal1[i] = sigmoid(hiddenVal1[i]);
        //printf("Hidden 1 activation\t%lf\n",hiddenVal1[i]);
    }
    //output of hidden layer 2 neurons
    for (i = 0; i < hidden2; ++i)
    {
        hiddenVal2[i] = 0.0 + bias2[i];

        for (j = 0; j < hidden1; ++j)
        {

            hiddenVal2[i] = hiddenVal2[i] + (hiddenVal1[j] * weights2[j][i]);
        }
        hiddenVal2[i] = sigmoid(hiddenVal2[i]);
        //printf("Hidden 2 activation\t%lf\n",hiddenVal2[i]);
        //printf("%lf\t", hiddenVal2[i]);

    }
    //printf("\n");

    //calculate the output of the network

    for (i = 0; i < output; ++i)
    {
        outPred[i] = 0.0 + bias3[0];

        for (j = 0; j < hidden2; ++j)
        {
            outPred[i] = outPred[i] + (hiddenVal2[j] * weights3[j][i]);
        }
        //printf("\n output activation %lf  ", outPred[i]);
        outPred[i] = sigmoid(outPred[i]); /// Thats the individual output layer neuron prediction
        //printf("\n output activation %lf\n", outPred[i]);


    }

    for (i = 0; i < output; ++i)
    {
        errThisPat[i] =  validationoutput[patNum][i] - outPred[i];
        //printf("e %lf\t",errThisPat[i]);
    }


}


void backprop()
{

    /// Backpropagation
    //printf("%lf %lf = error %lf\n",outPred[patNum],trainOutput[patNum],-(outPred[patNum] - trainOutput[patNum]));

    //printf("\nInside backprop at %d pattern \n\n",patNum);
    for (i = 0; i < output; ++i)
    {
        output_delta[i] = -(trainOutput[patNum][i]-outPred[i]) * outPred[i] * (1 - outPred[i]);//output neuron delta
        netoutputdelta[patNum][i] = output_delta[i];
        //sumdeltaoutput[i] = output_delta[i];
        //printf("netoutdelta[%d][%d] = %lf\n",patNum,i,output_delta[i]);
        //dc_db[3]+=output_delta[i];
    }


    //hidden2 to output delta weights
    for (i = 0; i < hidden2; ++i)
    {
        weightedsum = 0.0;
        for (j = 0; j < output; ++j)
        {
            weightedsum += output_delta[j] * weights3[i][j];
        }


        hidden2delta[i] = weightedsum * hiddenVal2[i] * (1 - hiddenVal2[i]);
        nethidden2delta[patNum][i] = hidden2delta[i];
        //printf("hiddendelta 2 %lf\n",hidden2delta[i]);


    }

    //hidden1 to hidden2 delta weights
    for (i = 0; i < hidden1; ++i)
    {
        weightedsum = 0.0;
        for (j = 0; j < hidden2; ++j)
        {
            weightedsum += hidden2delta[j] * weights2[i][j];
        }

        hidden1delta[i] = weightedsum * hiddenVal1[i] * (1 - hiddenVal1[i]);
        nethidden1delta[patNum][i] = hidden1delta[i];
        //printf("\nhiddendelta 1 %lf\n",hidden1delta[i]);

    }


    ///dc_dw calculation
    for (i = 0; i < input; ++i)
    {
        for (j = 0; j < hidden1; ++j)
        {
            dc_dw_hidden1[patNum][j] = hidden1delta[j] * trainInputs[patNum][i];
            //printf("%lf \t",weights1[i][j]); //Calculated gradient * Old activation
        }
        //printf("\n");
    }

    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < hidden2; ++j)
        {
            dc_dw_hidden2[patNum][j] = hidden2delta[j] * hiddenVal1[i];
            //printf("%lf \t",weights2[i][j]); //Calculated gradient * Old activation
        }
        //printf("\n");
    }
    for (i = 0; i < hidden2; ++i)
    {
        for (j = 0; j < output; ++j)
        {
            dc_dw_output[patNum][j] = output_delta[j] * hiddenVal2[i];
            //printf("%lf \t",weights3[i][j]);
        }
        //printf("\n");
    }

}

void sum_and_update()
{
    double sum_dc_dw_output[output] = {0.0};
    double sum_dc_dw_hidden2[hidden2] = {0.0};
    double sum_dc_dw_hidden1[hidden1] = {0.0};
    double sumdeltaoutput[output] = {0.0};
    double sumdeltahidden2[hidden2] = {0.0};
    double sumdeltahidden1[hidden1] = {0.0};



    for(i=0; i<output; ++i)
    {
        for (j=0; j<patNum+1; ++j)
        {
            sumdeltaoutput[i] += netoutputdelta[j][i];
            sum_dc_dw_output[i] += dc_dw_output[j][i];

        }
        //printf("\nsum_delta_output %lf\n", sum_dc_dw_output[i]);
    }

    for(i=0; i<hidden2; ++i)
    {
        for (j=0; j<patNum+1; ++j)
        {
            sumdeltahidden2[i] += nethidden2delta[j][i];
            sum_dc_dw_hidden2[i] += dc_dw_hidden2[j][i];


        }
        //printf("\nsum_delta_hidden2 %lf\n", sum_dc_dw_hidden2[i]);
    }

    for(i=0; i<hidden1; ++i)
    {
        for (j=0; j<patNum+1; ++j)
        {
            sumdeltahidden1[i] += nethidden1delta[j][i];
            sum_dc_dw_hidden1[i] += dc_dw_hidden1[j][i];

        }
        //printf("\nsum_delta_hidden1 %lf\n", sum_dc_dw_hidden1[i]);
    }


    /// Bias update.
    for (i = 0; i < output; ++i)
    {
        bias3[i] -= lR*sumdeltaoutput[i] ;
    }
    for (i = 0; i < hidden2; ++i)
    {
        bias2[i] -= lR*sumdeltahidden2[i] ;
        //printf("\nbias 2 %lf\t",bias2[i]);
    }
    for (i = 0; i < hidden1; ++i)
    {
        bias1[i] -= lR*sumdeltahidden1[i] ;
    }

    /// update weights


    //printf("\nweights 1\n");
    for (i = 0; i < input; ++i)
    {
        for (j = 0; j < hidden1; ++j)
        {
            weights1[i][j] -= lR*sum_dc_dw_hidden1[j];
            //printf("%lf \t",weights1[i][j]);
        }
        //printf("\n");
    }

    //printf("\nweights 2\n");
    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < hidden2; ++j)
        {
            weights2[i][j] -= lR*sum_dc_dw_hidden2[j];
            //printf("%lf \t",weights2[i][j]); //Calculated gradient * Old activation
        }
        //printf("\n");
    }
    //printf("\nweights 3\n");
    for (i = 0; i < hidden2; ++i)
    {
        for (j = 0; j < output; ++j)
        {
            weights3[i][j] -= lR*sum_dc_dw_output[j];
            //printf("%lf \t",weights3[i][j]);
        }
        //printf("\n");
    }


}


double randrange()
{

    double r = 4 * sqrt(6.0 / (input+output)); //for Sigmoid
    const double MIN_RAND = 0, MAX_RAND = 1;
    const double range = MAX_RAND - MIN_RAND;
    double random = range * ((((double) rand()) / (double) RAND_MAX)) + MIN_RAND;
    return random;
}

//************************************
// set weights to random numbers
void initWeights(void)
{


    for (j = 0; j < hidden1; ++j)
    {
        for (int i = 0; i < input; ++i)
        {
            weights1[i][j] = ((double) randrange());
            //printf("\t%f\t",weights1[i][j]);
        }
    }
    for (j = 0; j < hidden1; ++j)
    {
        for (int i = 0; i < hidden2; ++i)
        {
            weights2[i][j] = ((double) randrange());
        }
    }

    for (j = 0; j < hidden2; ++j)
    {
        for (int i = 0; i < output; ++i)
        {
            weights3[i][j] = ((double) randrange());
        }
    }
}


//************************************
// display results
void displayResults(void)
{
    for (int i = 0; i < numPatterns; ++i)
    {
        patNum = i;
        calcNet();
        for (j = 0; j < output; ++j)
        {
            printf("pat = %d actual = %lf neural model = %lf\n", patNum + 1, trainOutput[patNum -1][output],outPred[j]);
        }
    }
}


//************************************
// calculate the overall error
void calcOverallError(void)
{
    RMSerror = 0.0;
    for (int i = 0; i < numPatterns; ++i)
    {
        patNum = i;
        calcNet();
        for (j = 0; j < output; ++j)
        {

            RMSerror = (RMSerror + (errThisPat[j] * errThisPat[j])) / output;
        }
    }
    RMSerror = RMSerror / numPatterns;
    RMSerror = sqrt(RMSerror);
}


double Filter( double sample)
{

    static  char oldest = 0;
    static  double sum=0;

    sum -= buffer[oldest];
    sum += sample;
    buffer[oldest] = sample;
    oldest += 1;
    if (oldest >= n) oldest = 0;

    return sum/n;
}




void initData()
{
    FILE *fin,*fout;
    double inp[7],op;
    int count=0,i,j;
    double inputset1[numPatterns],inputset2[numPatterns],inputset3[numPatterns],outputset[numPatterns]={0};
    double *minmax1,*minmax2,*minmax3,*minmaxout;

    fin = fopen("training_input", "r");
    fout = fopen("training_output", "r");

    printf("Init inputs");
    //pFile = fopen ("f1.txt","w");
    while(!feof(fin))
    {

        fscanf(fin,"%lf %lf %lf %lf %lf %lf %lf\n",&inp[0],&inp[1],&inp[2],&inp[3],&inp[4],&inp[5],&inp[6]);
        fscanf(fout,"%lf\n",&op);
//
//        if (count%5 == 0){
//        printf("hel %d\n",count);
//        validationinput[count][0] = inp[0];
//        validationinput[count][1] = inp[1];
//        validationinput[count][2] = inp[2];
//        validationinput[count][3] = inp[3];
//        validationinput[count][4] = inp[4];
//        validationinput[count][5] = inp[5];
//        validationinput[count][6] = inp[6];
//        validationoutput[count][0] = op;
//        //fprintf(fvalidate,"%lf %lf %lf %lf %lf %lf %lf\n",inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6]);
//        }
//
//        else if(count%7 == 0){
//        testinput[count][0] = inp[0];
//        testinput[count][1] = inp[1];
//        testinput[count][2] = inp[2];
//        testinput[count][3] = inp[3];
//        testinput[count][4] = inp[4];
//        testinput[count][5] = inp[5];
//        testinput[count][6] = inp[6];
//        testoutput[count][0] = op;
//        //fprintf(ftest,"%lf %lf %lf %lf %lf %lf %lf\n",inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6]);
//
//
//        }
//
//        else{
        trainInputs[count][0] = inp[0];
        trainInputs[count][1] = inp[1];
        trainInputs[count][2] = inp[2];
        trainInputs[count][3] = inp[3];
        trainInputs[count][4] = inp[4];
        trainInputs[count][5] = inp[5];
        trainInputs[count][6] = inp[6];
        trainOutput[count][0] = op;
        //fprintf(asdf,"%lf %lf %lf %lf %lf %lf %lf\n",inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6]);

        //}

        count++;
        if(count>numPatterns)
        {
            printf("%d\n",count);
            break;
        }

    }

    fclose(fin);
    fclose(fout);
    //fclose(pFile);

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


int main(void)
{
    double moving_avg = 0.0;
    // seed random number function

    //pFile = fopen ("f.txt","w");


    srand(time(NULL));
    // initiate the weights
    initWeights();

    // load in the data
    initData();

    // train the network
    for (int j = 0; j <= numEpochs; ++j)
    {

        for (int i = 0; i < numPatterns; ++i)
        {
            //select a pattern at random

            //patNum = rand()%numPatterns;
            patNum = i;
            //calculate the current network output
            //and error for this pattern
            calcNet();
            //printf("\nAfter % epoch, error = %lf \n",i,errThisPat);
            backprop();
            //printf("net op delta at %d = %lf\n",patNum,netoutputdelta[patNum][0]);


//            printf("------------------------------------------------------------\n");


        }
//        printf("------------------------------------------------------------\n");
//        printf("\nsum error\n");
//        printf("------------------------------------------------------------\n");
        sum_and_update();
        //printf("sum output delta %lf ",sumdeltaoutput[0]);
        //display the overall network error after each epoch
        calcOverallError();
        moving_avg = Filter(RMSerror);
        //float truncated = trunc(moving_avg*10000)/10000;
        printf("epoch %d error = %lf \n",j,RMSerror);

//        if(RMSerror > moving_avg && j>1000 )
//        {
//            if (RMSerror>moving_avg+0.3){
//                continue;
//            }
//            else{
//                printf("Moving avg = %lf\n",moving_avg);
//                printf("%lf\n",RMSerror);
//                displayResults();
//                exit(0);
//            }
//        }

        //fprintf(pFile,"%f\n", RMSerror);

    }

    //training has finished
    //display the results
    printf("\n------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");

    displayResults();
    //fclose(pFile);


    //system("PAUSE");
    return 0;
}



//double * minmax(double v[])
//{
//
//    double min1=9999999,max1 = -9999,len=0;
//    static double out[2] = {0,0};
//    int i=0;
//    for(int i=0; v[i]; i++)
//        len++;
//
//    for(i=0; i<len; i++)
//    {
//
//        if(v[i]>max1)
//        {
//            max1=v[i];
//        }
//        if(v[i]<min1)
//        {
//            min1=v[i];
//        }
//    }
//    out[0] = min1;
//    out[1] = max1;
//    printf("inside func%lf %lf\n",out[0],out[1]);
//    return out;
//}
