/**
- initData()
- initWeights(void)
- forwardPass(double inputarray[][input], double outputarray[][output])
- backprop(void)
- sum_and_update(void)
- displayResults(void)
- calcOverallError(void)
- calcValidationError(void)
- Calc_moving_average(double sample)
*/

void initData(void)
{
    ///Read data from files amd store into inp array.
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

void forwardPass(double inputarray[][input], double outputarray[][output])
{
    ///Forward Pass.

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
    }

    ///output of hidden layer 2 neurons
    for (i = 0; i < hidden2; ++i)
    {
        hiddenVal2[i] = 0.0 + bias2[i];

        for (j = 0; j < hidden1; ++j)
        {

            hiddenVal2[i] = hiddenVal2[i] + (hiddenVal1[j] * weights2[j][i]);
        }
        hiddenVal2[i] = sigmoid(hiddenVal2[i]);

    }

    ///calculate the output of the network
    for (i = 0; i < output; ++i)
    {
        outPred[i] = 0.0 + bias3[0];

        for (j = 0; j < hidden2; ++j)
        {
            outPred[i] = outPred[i] + (hiddenVal2[j] * weights3[j][i]);
        }
        outPred[i] = sigmoid(outPred[i]);
        /// Thats the individual output layer neuron prediction


    }

    for (i = 0; i < output; ++i)
    {
        errThisPat[i] = outputarray[patNum][i] - outPred[i];
    }
}


void backprop(void)
{

    /// Backpropagation

    for (i = 0; i < output; ++i)
    {
        output_delta[i] = -(trainOutput[patNum][i] - outPred[i]) * \
                          outPred[i] * (1 - outPred[i]);//output neuron delta
        netoutputdelta[patNum][i] = output_delta[i];
    }


    ///hidden2 to output delta weights
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


    }

    ///hidden1 to hidden2 delta weights
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
        }
        //printf("\n");
    }

    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < hidden2; ++j)
        {
            dc_dw_hidden2[patNum][j] = hidden2delta[j] * hiddenVal1[i];
            //Old activation
        }
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

void sum_and_update(void)
{
    /**
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
    }

    for (i = 0; i < hidden2; ++i)
    {
        for (j = 0; j < patNum + 1; ++j)
        {
            sumdeltahidden2[i] += nethidden2delta[j][i];
            sum_dc_dw_hidden2[i] += dc_dw_hidden2[j][i];


        }
    }

    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < patNum + 1; ++j)
        {
            sumdeltahidden1[i] += nethidden1delta[j][i];
            sum_dc_dw_hidden1[i] += dc_dw_hidden1[j][i];

        }
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


    for (i = 0; i < input; ++i)
    {
        for (j = 0; j < hidden1; ++j)
        {
            weights1[i][j] -= lR * sum_dc_dw_hidden1[j];
            //printf("%lf \t",weights1[i][j]);
        }
    }

    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < hidden2; ++j)
        {
            weights2[i][j] -= lR * sum_dc_dw_hidden2[j];
        }
    }
    for (i = 0; i < hidden2; ++i)
    {
        for (j = 0; j < output; ++j)
        {
            weights3[i][j] -= lR * sum_dc_dw_output[j];
        }
    }


}

//************************************
/// display results of test set forward pass.
void displayResults(void)
{
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
/// calculate the overall error for triaining set
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

