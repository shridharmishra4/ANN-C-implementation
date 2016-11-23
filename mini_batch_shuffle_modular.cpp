#define input  7
#define hidden1  20
#define hidden2  10
#define output 1
#define numPatterns 1999
#define numEpochs 40000
#define stoping_error 0.05
#define lR 0.001
#define train_ratio 0.7
#define validate_ratio 0.2
#define test_ratio 0.1
#define mini_batch_size 20
#define mov_avg_range 3

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "helperfunction.h"

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




//************************************
// set weights to random numbers


//************************************
// display results
void displayResults(void)
{
    printf("Inside testing\n");
    for (int i = numPatterns*(validate_ratio+train_ratio); i < numPatterns; ++i)
    {
        patNum = i;
        forwardPass(testinput,testoutput);
        for (j = 0; j < output; ++j)
        {
            printf("pat = %d actual = %lf neural model = %lf err = %lf\n", \
                   patNum, testoutput[patNum][j],
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
    for (int i = 0; i < numPatterns*train_ratio; ++i)
    {
        patNum = i;
        forwardPass(trainInputs,trainOutput);
        for (j = 0; j < output; ++j)
        {

            RMSerrortrain = (RMSerrortrain + \
                             (errThisPat[j] * errThisPat[j])) / output;
        }
    }
    RMSerrortrain = RMSerrortrain / (numPatterns*train_ratio);
    RMSerrortrain = sqrt(RMSerrortrain);
}


void calcValidationError(void)
{
    ///Calculate error during validation
    RMSerrorvalidation = 0.0;
    for (int i = numPatterns*train_ratio; i <= numPatterns*(1-test_ratio); ++i)
    {
        patNum = i;
        forwardPass(validationinput,validationoutput);
        for (j = 0; j < output; ++j)
        {

            RMSerrorvalidation = (RMSerrorvalidation + \
                                  (errThisPat[j] * errThisPat[j])) / output;
        }
    }
    RMSerrorvalidation = RMSerrorvalidation / (numPatterns*validate_ratio);
    RMSerrorvalidation = sqrt(RMSerrorvalidation);

}

double Calc_moving_average(double sample)
{

    static char oldest = 0;
    static double sum = 0;

    sum -= buffer[oldest];
    sum += sample;
    buffer[oldest] = sample;
    oldest += 1;
    if (oldest >= mov_avg_range) oldest = 0;

    return sum / mov_avg_range;
}

void initData()
{
    FILE *fin, *fout;

    int count = 0, i, j;

    fin = fopen("training_input", "r");
    fout = fopen("training_output", "r");
    printf("Init inputs");

    while (!feof(fin))
    {
        fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf\n", &inp[count][0], \
               &inp[count][1], &inp[count][2], &inp[count][3], &inp[count][4], &inp[count][5], &inp[count][6]);
        fscanf(fout, "%lf\n", &inp[count][7]);
        count++;
    }
    fclose(fin);
    fclose(fout);
}




int main(void)
{
    double moving_avg = 0.0;
    double bestvalidation = 999;
    int patience = 600;
    // seed random number function
    FILE *validationerrorfile, *test;
    validationerrorfile = fopen("err_validate.txt","w");
    srand(time(NULL));
    // initiate the weights
    initWeights();

    // Loads input data for training and validation from file.
    initData();

    // train the network
    for (int j = 0; j <= numEpochs; ++j)
    {
        shuffle_data();
        for (int i = 0; i < train_minibatch_number; ++i)
        {
            ///i represents mini batch index.
            for (int k = 0; k < mini_batch_size; ++k)
            {

                patNum = i * mini_batch_size + k;
                //printf("%d = %d * %d + %d\n",patNum,i,mini_batch_size,k);
                //calculate the current network output
                //and error for this pattern
                forwardPass(trainInputs,trainOutput);
                backprop();

            }
            sum_and_update();
        }
        ///to do: Add save network.
        calcOverallError();
        calcValidationError();


        //Write to file for graph plotting
        fprintf(validationerrorfile,"%lf %lf \n",RMSerrortrain, \
                RMSerrorvalidation);
        printf("After %d epoch, RmsTrain = %lf RMSvalidate = %lf best =%lf patience = %d\n",j,RMSerrortrain,RMSerrorvalidation, \
               bestvalidation,patience);


        ///Early Stopping
        moving_avg = Calc_moving_average(RMSerrorvalidation);


        if(RMSerrorvalidation<stoping_error)
        {
            if(RMSerrorvalidation<bestvalidation*0.95)
            {
                patience = 600;
                //            if (RMSerrorvalidation > moving_avg+0.1){
                //                continue;
                //            }
                //            else{
                //printf("\n\n\n\ndifference => %f - %f = %lf\n",\
                moving_avg,RMSerrorvalidation, \
                (moving_avg-RMSerrorvalidation));

                printf("\n\n\n\ndifference => %f - %f = %lf\n", \
                        moving_avg,RMSerrorvalidation, \
                        (moving_avg-RMSerrorvalidation));
            }



        patience--;

        if(patience<=0){
                printf("\n\nexit training\n");
                break;
                }

        }

        if(RMSerrorvalidation<bestvalidation)
        {
            bestvalidation = RMSerrorvalidation;
        }


    }


    ///training has finished
    ///display the results from training set
    printf("\n------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");

    displayResults();
    fclose(validationerrorfile);



    system("PAUSE");
    return 0;
}
