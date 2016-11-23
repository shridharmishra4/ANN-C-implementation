
int x = 0, y = 0, i = 0, j = 0, patNum = 0;
int train_minibatch_number = numPatterns*train_ratio/mini_batch_size;
int validate_minibatch_number = numPatterns* validate_ratio/mini_batch_size;

double buffer[mov_avg_range] = {0};
double inp[numPatterns][8];
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



double sigmoid(double x)
{
//***********************************
// calculates the sigmoid of x

    double exp_value;
    double return_value;

    /*** Exponential calculation ***/
    exp_value = exp((double) -x);

    /*** Final sigmoid value ***/
    return_value = 1 / (1 + exp_value);

    return return_value;
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

void swap (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void randomize ( int arr[], int n )
{
    // Use a different seed value so that we don't get same
    // result each time we run this program
    srand ( time(NULL) );

    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = n-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i+1);

        // Swap arr[i] with the element at random index
        swap(&arr[i], &arr[j]);
    }
}

void shuffle_data()
{
    int i=0,count = 0;
    int index[numPatterns];

    //FILE *a;
    //a = fopen("randowm.txt","w");

    for(i=0; i<numPatterns; i++)
    {
        index[i] = i;
    }
    randomize(index,numPatterns);
    while(count<numPatterns)
    {
        if (count<train_ratio*numPatterns)
        {

            trainInputs[count][0] = inp[index[count]][0];
            trainInputs[count][1] = inp[index[count]][1];
            trainInputs[count][2] = inp[index[count]][2];
            trainInputs[count][3] = inp[index[count]][3];
            trainInputs[count][4] = inp[index[count]][4];
            trainInputs[count][5] = inp[index[count]][5];
            trainInputs[count][6] = inp[index[count]][6];
            trainOutput[count][0] = inp[index[count]][7];
//fprintf(a,"%d %lf %lf %lf %lf %lf %lf %lf\n",index[count],inp[index[count]][0],inp[index[count]][1],inp[index[count]][2],inp[index[count]][3],inp[index[count]][4],inp[index[count]][5],inp[index[count]][6]);

        }

        else if (count>train_ratio*numPatterns && count<(1-test_ratio)*numPatterns)
        {
            validationinput[count][0] = inp[index[count]][0];
            validationinput[count][1] = inp[index[count]][1];
            validationinput[count][2] = inp[index[count]][2];
            validationinput[count][3] = inp[index[count]][3];
            validationinput[count][4] = inp[index[count]][4];
            validationinput[count][5] = inp[index[count]][5];
            validationinput[count][6] = inp[index[count]][6];
            validationoutput[count][0] = inp[index[count]][7];
            //fprintf(a,"%d %lf %lf %lf %lf %lf %lf %lf\n",index[count],inp[index[count]][0],inp[index[count]][1],inp[index[count]][2],inp[index[count]][3],inp[index[count]][4],inp[index[count]][5],inp[index[count]][6]);


        }
        else
        {
            testinput[count][0] = inp[index[count]][0];
            testinput[count][1] = inp[index[count]][1];
            testinput[count][2] = inp[index[count]][2];
            testinput[count][3] = inp[index[count]][3];
            testinput[count][4] = inp[index[count]][4];
            testinput[count][5] = inp[index[count]][5];
            testinput[count][6] = inp[index[count]][6];
            testoutput[count][0] = inp[index[count]][7];
            //fprintf(a,"%d %lf %lf %lf %lf %lf %lf %lf\n",index[count],inp[index[count]][0],inp[index[count]][1],inp[index[count]][2],inp[index[count]][3],inp[index[count]][4],inp[index[count]][5],inp[index[count]][6]);
        }

        count++;
    }
        //fclose(a);

}



