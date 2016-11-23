#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define input  2
#define hidden1  2
#define hidden2  2
#define output 1
#define numPatterns 4
#define numEpochs 5000
#define stoping_error 0.01
#define lR 7.5

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
double sumdeltabias = 0.0;
double dc_db[input+hidden1+hidden2+output] = {0};
double trainInputs[numPatterns][input] = {{0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};
double trainOutput[numPatterns][output] = {{0},
    {1},
    {1},
    {0}
};
FILE * pFile;

//// functions ////
void initWeights();

void calcNet();

void calcOverallError();

void displayResults();

double getRand();



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
    ///output of hidden layer 1
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

        outPred[i] = sigmoid(outPred[i]); /// Thats the individual output layer neuron prediction
        //printf("\n output activation %lf\n", outPred[i]);


    }

    for (i = 0; i < output; ++i)
    {
        errThisPat[i] =  trainOutput[patNum][i] - outPred[i];
        //printf("error = %lf",errThisPat[i]);
    }


}


void backprop()
{

    /// Backpropagation
    //printf("%lf %lf = error %lf\n",outPred[patNum],trainOutput[patNum],-(outPred[patNum] - trainOutput[patNum]));

    //printf("\nInside backprop at %d pattern \n\n",patNum+1);
    for (i = 0; i < output; ++i)
    {
        output_delta[i] = -(trainOutput[patNum][i]-outPred[i]) * outPred[i] *
                          (1 - outPred[i]);//output neuron delta
        //printf("outdelta  %lf\n",output_delta[i]);
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
        //printf("hiddendelta 2 %lf\n",hidden2delta[i]);
        //dc_db[2]+=hidden2delta[i];


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
        //printf("\nhiddendelta 1 %lf\t",hidden1delta[i]);
        //dc_db[1] +=hidden1delta[i];

    }



    for (i = 0; i < input; ++i)
    {
        weightedsum = 0.0;
        for (j = 0; j < hidden1; ++j)
        {
            weightedsum += hidden1delta[j] * weights1[i][j];
        }

        inputdelta[i] = weightedsum * trainInputs[patNum][i] ;
        //printf("\ninputdelta  %lf\t",inputdelta[i]);

    }
    // Bias update.
    for (i = 0; i < output; ++i)
    {
        bias3[i] -= lR*output_delta[output] ;
    }
    for (i = 0; i < hidden2; ++i)
    {
        bias2[i] -= lR*hidden2delta[i] ;
        //printf("\nbias 2 %lf\t",bias2[i]);
    }
    for (i = 0; i < hidden1; ++i)
    {
        bias1[i] -= lR*hidden1delta[i] ;
    }
    // update weights


    //printf("\nweights 1\n");
    for (i = 0; i < input; ++i)
    {
        for (j = 0; j < hidden1; ++j)
        {
            weights1[i][j] -= lR*hidden1delta[j] * trainInputs[patNum][i];
            //printf("%lf \t",weights1[i][j]); //Calculated gradient * Old activation
        }
        //printf("\n");
    }

    //printf("\nweights 2\n");
    for (i = 0; i < hidden1; ++i)
    {
        for (j = 0; j < hidden2; ++j)
        {
            weights2[i][j] -= lR*hidden2delta[j] * hiddenVal1[i];
            //printf("%lf \t",weights2[i][j]); //Calculated gradient * Old activation
        }
        //printf("\n");
    }
    //printf("\nweights 3\n");
    for (i = 0; i < hidden2; ++i)
    {
        for (j = 0; j < output; ++j)
        {
            weights3[i][j] -= lR*output_delta[j] * hiddenVal2[i];
            //printf("%lf \t",weights3[i][j]);
        }
        //printf("\n");
    }
}

float randrange()
{

    float r = 4 * sqrt(6.0 / (input+output)); //for Sigmoid
    const float MIN_RAND = 0, MAX_RAND = 1;
    const float range = MAX_RAND - MIN_RAND;
    float random = range * ((((float) rand()) / (float) RAND_MAX)) + MIN_RAND;
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
            //printf("Weight = %f\n", weightsIH[i][j]);
        }
    }
    for (j = 0; j < hidden1; ++j)
    {
        for (int i = 0; i < hidden2; ++i)
        {
            weights2[i][j] = ((double) randrange());
            //printf("Weight = %f\n", weightsIH[i][j]);
        }
    }

    for (j = 0; j < hidden2; ++j)
    {
        for (int i = 0; i < output; ++i)
        {
            weights3[i][j] = ((double) randrange());
            //printf("Weight = %f\n", weightsIH[i][j]);
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
            printf("pat = %d actual = %lf neural model = %lf\n", patNum + 1, trainOutput[patNum - 1][output],
                   outPred[j]);
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





int main(void)
{
    // seed random number function

    //pFile = fopen ("f.txt","w");


    srand(time(NULL));

    // initiate the weights
    initWeights();

    // load in the data
    //initData();

    // train the network
    for (int j = 0; j <= numEpochs; ++j)
    {

        for (int i = 0; i < numPatterns; ++i)
        {
            //select a pattern at random

            //patNum = rand()%numPatterns;
            patNum = i;
            //printf("------------------------------------------------------------\n");


            //calculate the current network output
            //and error for this pattern
            calcNet();
            //printf("\nAfter % epoch, error = %lf \n",i,errThisPat);

            //change network weights
            backprop();
            //printf("------------------------------------------------------------\n");


        }
        //display the overall network error
        //after each epoch
        calcOverallError();
        printf("epoch %d error = %lf \n",j,RMSerror);
        if(RMSerror<stoping_error)
        {
            printf("%lf\n",RMSerror);
            displayResults();
            exit(0);
        }

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
