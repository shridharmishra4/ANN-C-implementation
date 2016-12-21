#define input  7
#define hidden1  20
#define hidden2  10
#define output 1
#define numPatterns 1999
#define numEpochs 40000
#define stoping_error 0.05
#define lR 0.01
#define train_ratio 0.7
#define validate_ratio 0.2
#define test_ratio 0.1
#define mini_batch_size 30
#define mov_avg_range 3

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "helperfunction.h"
#include "neuralnet.h"





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

    //Loads input data for training and validation from file.
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

            if(patience<=0)
            {
                printf("\n\nexit training\n");
                break;
            }

        }

        if(RMSerrorvalidation<bestvalidation)
        {
            bestvalidation = RMSerrorvalidation;
            copyweights_biases();
//            for(i=0;i<hidden1;i++){
//                for(j=0;j<hidden2;j++){
//                    printf("[%d][%d] = %lf ",i,j,weights2[i][j]);
//                    }
//                printf("\n");
//
//        }
        }


    }
    save_weights();



    ///training has finished
    ///display the results from training set
    printf("\n------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");

    displayResults();
    fclose(validationerrorfile);



    //system("PAUSE");
    return 0;
}
