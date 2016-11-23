#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define input  7
#define hidden1  20
#define hidden2  10
#define output 1
#define numPatterns 1999
#define numEpochs 20000
#define stoping_error 0.03
#define lR 0.009
#define n 3

double buffer[n] = {0};

int x = 0, y = 0, i = 0, j = 0, patNum = 0;
double weights1[input][hidden1] = {{1, 1},
    {1, 1}
};
double weights2[hidden1][hidden2] = {{1, 1},
    {1, 1}
};
double weights3[hidden2][output] = {{1},
    {1}
};

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
double RMSerrortrain = 0.0, RMSerrorvalidation = 0.0;
double netoutputdelta[numPatterns][output] = {0.0};
double nethidden2delta[numPatterns][hidden2] = {0.0};
double nethidden1delta[numPatterns][hidden1] = {0.0};

double dc_dw_hidden1[numPatterns][hidden1] = {0.0};
double dc_dw_hidden2[numPatterns][hidden2] = {0.0};
double dc_dw_output[numPatterns][output] = {0.0};


double trainInputs[numPatterns][input] = {0};
double trainOutput[numPatterns][output] = {0};

double validationinput[numPatterns][input] = {0};
double validationoutput[numPatterns][output] = {0};

double testinput[numPatterns][input] = {0};
double testoutput[numPatterns][output] = {0};
double inputset1[numPatterns] = {0};
double inputset2[numPatterns] = {0};
double inputset3[numPatterns] = {0};
double outputset[numPatterns] = {0};


FILE *pFile;

//// functions ////
void initWeights();

void forwardPass();

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
void forwardPass(double inputarray[][input], double outputarray[][output])
{
    ///calculate the outputs of the hidden layer 1 neurons
    int i = 0, j = 0;
    for (i = 0; i < hidden1; ++i)
    {
        hiddenVal1[i] = 0.0 + bias1[i];

        for (j = 0; j < input; ++j)
        {
            hiddenVal1[i] = hiddenVal1[i] + (inputarray[patNum][j] * \
                                             weights1[j][i]);

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
        outPred[i] = sigmoid(outPred[i]);
        /// Thats the individual output layer neuron prediction
        //printf("\n output activation %lf\n", outPred[i]);


    }

    for (i = 0; i < output; ++i)
    {
        errThisPat[i] = outputarray[patNum][i] - outPred[i];
        //printf("e %lf\t",errThisPat[i]);
    }


}



void backprop()
{

    /// Backpropagation
    //printf("%lf %lf = error %lf\n",outPred[patNum],\
    trainOutput[patNum],-(outPred[patNum] - trainOutput[patNum]));

    //printf("\nInside backprop at %d pattern \n\n",patNum);
    for (i = 0; i < output; ++i)
{
    output_delta[i] = -(trainOutput[patNum][i] - outPred[i]) * \
                          outPred[i] * (1 - outPred[i]);//output neuron delta
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


        hidden2delta[i] = weightedsum * hiddenVal2[i] * \
                          (1 - hiddenVal2[i]);
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

        hidden1delta[i] = weightedsum * hiddenVal1[i] * \
                          (1 - hiddenVal1[i]);
        nethidden1delta[patNum][i] = hidden1delta[i];
        //printf("\nhiddendelta 1 %lf\n",hidden1delta[i]);

    }


    ///dc_dw calculation
    for (i = 0; i < input; ++i)
{
    for (j = 0; j < hidden1; ++j)
        {
            dc_dw_hidden1[patNum][j] = hidden1delta[j] * \
                                       trainInputs[patNum][i];
            //printf("%lf \t",weights1[i][j]); //Calculated gradient *
            //Old activation
        }
        //printf("\n");
    }

    for (i = 0; i < hidden1; ++i)
{
    for (j = 0; j < hidden2; ++j)
        {
            dc_dw_hidden2[patNum][j] = hidden2delta[j] * hiddenVal1[i];
            //printf("%lf \t",weights2[i][j]); //Calculated gradient *
            //Old activation
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
    /*
    Sums up all the error layerwise

    netxxxxdelta[PatternIndex][NeuronIndex]--- It stores the ouput of each
    neuron for each respective pattern.

    sum_dc_dw_xxxxx - sum of all the error from all the neurons of a layer
                      for each pattern
    */


    double sum_dc_dw_output[output] = {0.0};
    double sum_dc_dw_hidden2[hidden2] = {0.0};
    double sum_dc_dw_hidden1[hidden1] = {0.0};
    double sumdeltaoutput[output] = {0.0};
    double sumdeltahidden2[hidden2] = {0.0};
    double sumdeltahidden1[hidden1] = {0.0};


    for (i = 0; i < output; ++i)
    {
        for (j = 0; j < patNum + 1; ++j)
        {
            sumdeltaoutput[i] += netoutputdelta[j][i];
            sum_dc_dw_output[i] += dc_dw_output[j][i];

        }
        //printf("\nsum_delta_output %lf\n", sum_dc_dw_output[i]);
    }

    for (i = 0; i < hidden2; ++i)
    {
        for (j = 0; j < patNum + 1; ++j)
        {
            sumdeltahidden2[i] += nethidden2delta[j][i];
            sum_dc_dw_hidden2[i] += dc_dw_hidden2[j][i];


        }
        //printf("\nsum_delta_hidden2 %lf\n", sum_dc_dw_hidden2[i]);
    }

    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < patNum + 1; ++j)
        {
            sumdeltahidden1[i] += nethidden1delta[j][i];
            sum_dc_dw_hidden1[i] += dc_dw_hidden1[j][i];

        }
        //printf("\nsum_delta_hidden1 %lf\n", sum_dc_dw_hidden1[i]);
    }


    /// Bias update.
    for (i = 0; i < output; ++i)
    {
        bias3[i] -= lR * sumdeltaoutput[i];
    }
    for (i = 0; i < hidden2; ++i)
    {
        bias2[i] -= lR * sumdeltahidden2[i];
        //printf("\nbias 2 %lf\t",bias2[i]);
    }
    for (i = 0; i < hidden1; ++i)
    {
        bias1[i] -= lR * sumdeltahidden1[i];
    }

    /// update weights


    //printf("\nweights 1\n");
    for (i = 0; i < input; ++i)
    {
        for (j = 0; j < hidden1; ++j)
        {
            weights1[i][j] -= lR * sum_dc_dw_hidden1[j];
            //printf("%lf \t",weights1[i][j]);
        }
        //printf("\n");
    }

    //printf("\nweights 2\n");
    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < hidden2; ++j)
        {
            weights2[i][j] -= lR * sum_dc_dw_hidden2[j];
            //printf("%lf \t",weights2[i][j]); //Calculated gradient * \
            //Old activation
        }
        //printf("\n");
    }
    //printf("\nweights 3\n");
    for (i = 0; i < hidden2; ++i)
    {
        for (j = 0; j < output; ++j)
        {
            weights3[i][j] -= lR * sum_dc_dw_output[j];
            //printf("%lf \t",weights3[i][j]);
        }
        //printf("\n");
    }


}


double randrange()
{

    double r = 4 * sqrt(6.0 / (input + output)); //for Sigmoid
    const double MIN_RAND = 0, MAX_RAND = 1;
    const double range = MAX_RAND - MIN_RAND;
    double random = range * ((((double) rand()) / (double) RAND_MAX)) + \
                    MIN_RAND;
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
    for (int i = 0; i < numPatterns/7; ++i)
    {
        patNum = i;
        forwardPass(testinput,testoutput);
        for (j = 0; j < output; ++j)
        {
            printf("pat = %d actual = %lf neural model = %lf err = %lf\n", \
                   patNum + 1, testoutput[patNum][j],
                   outPred[j], errThisPat[j]);
        }
    }
}


//************************************
// calculate the overall error for triaining set
void calcOverallError(void)
{
    ///Calculate error during traing
    RMSerrortrain = 0.0;
    for (int i = 0; i < numPatterns; ++i)
    {
        patNum = i;
        forwardPass(trainInputs,trainOutput);
        for (j = 0; j < output; ++j)
        {

            RMSerrortrain = (RMSerrortrain + \
                             (errThisPat[j] * errThisPat[j])) / output;
        }
    }
    RMSerrortrain = RMSerrortrain / numPatterns;
    RMSerrortrain = sqrt(RMSerrortrain);
}


void calcValidationError(void)
{
    ///Calculate error during validation
    RMSerrorvalidation = 0.0;
    for (int i = 0; i <= numPatterns / 5; ++i)
    {
        patNum = i * 5;
        forwardPass(validationinput,validationoutput);
        for (j = 0; j < output; ++j)
        {

            RMSerrorvalidation = (RMSerrorvalidation + \
                                  (errThisPat[j] * errThisPat[j])) / output;
        }
    }
    RMSerrorvalidation = RMSerrorvalidation / (numPatterns/5);
    RMSerrorvalidation = sqrt(RMSerrorvalidation);

}

double Filter(double sample)
{

    static char oldest = 0;
    static double sum = 0;

    sum -= buffer[oldest];
    sum += sample;
    buffer[oldest] = sample;
    oldest += 1;
    if (oldest >= n) oldest = 0;

    return sum / n;
}


void initData()
{
    FILE *fin, *fout, *asdf, *ftest, *fvalidate;
    double inp[7], op;
    int count = 0, i, j;
    double *minmax1, *minmax2, *minmax3, *minmaxout;

    fin = fopen("training_input", "r");
    fout = fopen("training_output", "r");
    printf("Init inputs");
    while (!feof(fin))
    {

        fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf\n", &inp[0], \
               &inp[1], &inp[2], &inp[3], &inp[4], &inp[5], &inp[6]);
        fscanf(fout, "%lf\n", &op);

        if (count % 5 == 0)
        {
            validationinput[count][0] = inp[0];
            validationinput[count][1] = inp[1];
            validationinput[count][2] = inp[2];
            validationinput[count][3] = inp[3];
            validationinput[count][4] = inp[4];
            validationinput[count][5] = inp[5];
            validationinput[count][6] = inp[6];
            validationoutput[count][0] = op;
        }
        else if (count % 7 == 0)
        {
            testinput[count][0] = inp[0];
            testinput[count][1] = inp[1];
            testinput[count][2] = inp[2];
            testinput[count][3] = inp[3];
            testinput[count][4] = inp[4];
            testinput[count][5] = inp[5];
            testinput[count][6] = inp[6];
            testoutput[count][0] = op;

        }
        else
        {
            trainInputs[count][0] = inp[0];
            trainInputs[count][1] = inp[1];
            trainInputs[count][2] = inp[2];
            trainInputs[count][3] = inp[3];
            trainInputs[count][4] = inp[4];
            trainInputs[count][5] = inp[5];
            trainInputs[count][6] = inp[6];
            trainOutput[count][0] = op;

        }

        count++;
        if (count > numPatterns)
        {
            printf("%d\n", count);
            break;
        }

    }

    fclose(fin);
    fclose(fout);


}


int main(void)
{
    double moving_avg = 0.0;
    double bestvalidation = 9999;
    int patience = 500;
    // seed random number function
    FILE *validationerrorfile, *test;
    //trainerror = fopen ("err_train.txt","w");
    validationerrorfile = fopen("err_validate.txt","w");
    test = fopen("loopenter.txt","w");
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
            forwardPass(trainInputs,trainOutput);
            backprop();

        }
        sum_and_update();
        ///to do: Add save network.
        calcOverallError();
        calcValidationError();

        if(RMSerrorvalidation<bestvalidation)
        {
            bestvalidation = RMSerrorvalidation;
        }

        //Write to file for graph plotting
        fprintf(validationerrorfile,"%lf %lf \n",RMSerrortrain, \
                RMSerrorvalidation);
        printf("After %d epoch, RmsTrain = %lf RMSvalidate = %lf \
                best =%lf\n",j,RMSerrortrain,RMSerrorvalidation, \
               bestvalidation);


        ///Early Stopping
        moving_avg = Filter(RMSerrorvalidation);


        if(RMSerrorvalidation<stoping_error)
        {
            if(RMSerrorvalidation<bestvalidation*0.99)
            {
                patience = j + patience;
                //            if (RMSerrorvalidation > moving_avg+0.1){
                //                continue;
                //            }
                //            else{
                //printf("\n\n\n\ndifference => %f - %f = %lf\n",\
                moving_avg,RMSerrorvalidation, \
                (moving_avg-RMSerrorvalidation));

                fprintf(test,"\n\n\n\ndifference => %f - %f = %lf\n", \
                        moving_avg,RMSerrorvalidation, \
                        (moving_avg-RMSerrorvalidation));
                //exit(0);
                //printf("Moving avg = %lf\n",moving_avg);
                //printf("%lf\n",RMSerrorvalidation);
                //displayResults();

                //}
            }

//            if(j>= patience){
//                printf("\n\nexit\n");
//                exit(0);
//                }

        }

    }



//        if(RMSerrorvalidation<stoping_error)
//        {
//            if((moving_avg-RMSerrorvalidation)<0.0001 )
//            {
//                //            if (RMSerrorvalidation > moving_avg+0.1){
//                //                continue;
//                //            }
//                //            else{
//                printf("difference => %f - %f = %lf\n",moving_avg,\
    RMSerrorvalidation,(moving_avg-RMSerrorvalidation));
//                //printf("Moving avg = %lf\n",moving_avg);
//                //printf("%lf\n",RMSerrorvalidation);
//                //displayResults();
//                exit(0);
//                //}
//            }
//        }


    fclose(validationerrorfile);
    fclose(test);
    ///training has finished
    ///display the results from training set
    printf("\n------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");

    //displayResults();
    //fclose(pFile);


    //system("PAUSE");
    return 0;
}


