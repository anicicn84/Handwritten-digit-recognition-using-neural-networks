#pragma once
#include <math.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "logFunctionCall.h"
#include <random>

/*LEARNING PARAMETERS*/
//#define LEARNING_RATE 0.001
#define LEARNING_RATE 0.1
#define MOMENTUM 0.8
#define MAX_EPOCHS 5000
#define DESIRED_ACCURACY 90  
#define DESIRED_MSE 0.001 

using Eigen::MatrixXd; //a matrix of arbitrary size

int nInput = 784; //number of input neurons
int nHidden = 250;//number of hidden layer neurons
int nOutput = 10;//number of output neurons, 1 for each digit
bool useBatch = false;


double trainingSetAccuracy = 0;
double validationSetAccuracy = 0;
double generalizationSetAccuracy = 0;
double trainingSetMSE = 0;
double validationSetMSE = 0;
double generalizationSetMSE = 0;

//learning parameters
double learningRate;					// adjusts the step size of the weight update	
double momentum;						// improves performance of stochastic learning (don't use for batch)

										//epoch counter
long epoch;
long maxEpochs = 0;

//accuracy/MSE required
double desiredAccuracy;


Eigen::VectorXd inputNeurons (nInput + 1);
Eigen::VectorXd hiddenNeurons(nHidden + 1);
Eigen::VectorXd outputNeurons(nOutput); //vectors of the neurons in each layer

Eigen::VectorXd outputErrorGradients(nOutput);
Eigen::VectorXd hiddenErrorGradients(nHidden);

Eigen::VectorXd desiredOutput(nOutput);
MatrixXd wInputHidden(nInput + 1, nHidden);
MatrixXd wHiddenOutput(nHidden + 1, nOutput); //matrices of weights between different layers

MatrixXd deltaHiddenOutput = MatrixXd::Zero(nHidden + 1, nOutput);
MatrixXd deltaInputHidden = MatrixXd::Zero(nInput + 1, nHidden);


//define sigmoid function
double sigmoid_float(double f)
{
	return 1.0 / (1.0 + exp(-f));
}


//apply sigmoid function for each matrix element
MatrixXd sigmoid_matirx(MatrixXd f)
{
	auto sigmoid_matrix_ptr = [](double dbl)
	{
		return sigmoid_float(dbl);
	};
	return f.unaryExpr(sigmoid_matrix_ptr);
}

//define the derivative of sigmoid function
double sigmoid_prime_float(double f)
{
	return sigmoid_float(f)*(1 - sigmoid_float(f));
}


//apply derivative of sigmoid function for each matrix element
MatrixXd sigmoid_prime_matrix(MatrixXd f)
{
	auto sigmoid_prime_matrix_ptr = [](double dbl) 
	{
		return sigmoid_prime_float(dbl);
	};
	f.unaryExpr(sigmoid_prime_matrix_ptr);
	return f;
}

void feedForward(Eigen::VectorXd input)
{
	if (input.rows() != nInput)
	{
		std::cout << "input vector has invalid number of rows " << std::endl;
		return;
	}
	//set input neurons to input values
	inputNeurons << 1,  input;

	//Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	hiddenNeurons << 1, sigmoid_matirx(wInputHidden.transpose() * inputNeurons);

	//Calculating Output Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	outputNeurons << sigmoid_matirx(wHiddenOutput.transpose() * hiddenNeurons);
}

void initializeWeights()
{
	double rH = 1 / sqrt(static_cast<double> (nInput));
	double rO = 1 / sqrt(static_cast<double> (nHidden));

	//set weights to random values between -rH and rH, and -rO and rO

	std::random_device rd;				//used for seeding random generator
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> distrH(-rH, rH);  //between -0.03567 and 0.03567
	std::uniform_real_distribution<double> distrO(-rO, rO);  //between 0.182 and 0.182

    for (int i = 0; i < nInput+1; ++i)
	{
        for (int j = 0; j < nHidden; ++j)
		{
			wInputHidden(i,j) = distrH(mt);
		}
	}

    for (int i = 0; i < nHidden+1; ++i)
	{
        for (int j = 0; j < nOutput; ++j)
		{
			wHiddenOutput(i, j) = distrO(mt);
		}
	}

}

/*******************************************************************
* calculate output error gradient
********************************************************************/
double getOutputErrorGradientOne(double desiredValue, double outputValue)
{
	//return error gradient for element examined
	return outputValue * (1 - outputValue) * (desiredValue - outputValue);
}

double getHiddenErrorGradient(int j)
{
	//get sum of hidden->output weights * output error gradients
	double weightedSum = 0;
    for (int k = 0; k <nOutput; ++k) weightedSum += wHiddenOutput(j, k) * outputErrorGradients(k);

	//return error gradient
	return hiddenNeurons(j) * (1 - hiddenNeurons(j)) * weightedSum;
}

#if 0
Eigen::VectorXd getOutputErrorGradientVector(Eigen::VectorXd desiredValueVector, Eigen::VectorXd outputValueVector)
{
	if (desiredValueVector.rows() != outputValueVector.rows() && desiredValueVector.rows() != nOutput)
	{
		std::cout << "Error: vectors don't have the same size which is " << nOutput << " !!! " << std::endl;
		return;
	}

	//return error gradient as a vector for vector examined
	int i = 0;
	int numberOfVectorRows = desiredValueVector.rows();
	Eigen::VectorXd outputVector (numberOfVectorRows);
	for (i; i < numberOfVectorRows; ++i)
	{
		outputVector << getOutputErrorGradientOne(desiredValueVector(i), outputValueVector(i));
	}
	return outputVector;
}
#endif

/*******************************************************************
* Update weights using delta values
********************************************************************/
void updateWeights()
{
	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i < nInput + 1; ++i)
	{
		for (int j = 0; j < nHidden; ++j)
		{
			//update weight
			wInputHidden(i,j) += deltaInputHidden(i, j);

			//clear delta only if using batch (previous delta is needed for momentum
			if (useBatch) deltaInputHidden(i, j) = 0;
		}
	}

	//hidden -> output weights
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < nHidden + 1; ++j)
	{
		for (int k = 0; k < nOutput; ++k)
		{
			//update weight
			wHiddenOutput(j, k) += deltaHiddenOutput(j, k);

			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch)deltaHiddenOutput(j, k) = 0;
		}
	}
}

/************************************************************************
* Propagate errors back through Neural Network and calculate delta values
*************************************************************************/
void backpropagate(Eigen::VectorXd desiredOutput)
{
	if (desiredOutput.rows() != nOutput)
	{
		std::cout << "desired output vector has invalid number of rows " << std::endl;
		return;
	}

	//modify deltas between hidden and output layers
	//--------------------------------------------------------------------------------------------------------

#pragma omp parallel for
	for (int k = 0; k < nOutput; k++)
	{
		//get error gradient for every output node
		outputErrorGradients(k) = getOutputErrorGradientOne(desiredOutput(k), outputNeurons(k));

		//for all nodes in hidden layer and bias neuron
        for (int j=0; j < nHidden + 1; ++j)
		{
			//calculate change in weight
			if (!useBatch) deltaHiddenOutput(j, k) = LEARNING_RATE * hiddenNeurons(j) * outputErrorGradients(k) + MOMENTUM * deltaHiddenOutput(j, k);
			else deltaHiddenOutput(j, k) += LEARNING_RATE * hiddenNeurons(j) * outputErrorGradients(k);
		}
		hiddenNeurons(1) = 1;
	}

	//modify deltas between input and hidden layers
	//--------------------------------------------------------------------------------------------------------
#pragma omp parallel for
    for (int j = 0; j < nHidden; ++j)
	{
		//get error gradient for every hidden node
		hiddenErrorGradients(j) = getHiddenErrorGradient(j);

		//for all nodes in input layer and bias neuron
        for (int i = 0; i < nInput + 1; ++i)
		{
			//calculate change in weight 
			if (!useBatch) deltaInputHidden(i, j) = LEARNING_RATE * inputNeurons(i) * hiddenErrorGradients(j) + MOMENTUM * deltaInputHidden(i, j);
			else deltaInputHidden(i, j) += LEARNING_RATE * inputNeurons(i) * hiddenErrorGradients(j);

		}
		inputNeurons(1) = 1;
	}

	//if using stochastic learning update the weights immediately

	if (!useBatch) updateWeights();
}

/*******************************************************************
* Output Clamping
********************************************************************/
int clampOutput(double x)
{
	if (x < 0.2) return 0;
	else if (x > 0.8) return 1;
	else return -1;
}


//change of function signature !!!
void runTrainingEpoch(const std::vector<Eigen::VectorXd>& trainingSet, const std::vector<Eigen::VectorXd>& desiredOutput)
{
	//incorrect patterns
	double incorrectPatterns = 0;
	double mse = 0;

	//for every training pattern
    for (int tp = 0; tp < (int) trainingSet.size(); ++tp)
	{
		//feed inputs through network and backpropagate errors
        feedForward(trainingSet[tp]);

		//std::clog << outputNeurons << "\n";

        backpropagate(desiredOutput[tp]);

		//TODO: std::clog << 
		
		//std::cin.get();

		//pattern correct flag
		bool patternCorrect = true;

		//check all outputs from neural network against desired values
        for (int k = 0; k < nOutput; ++k)
		{
			//pattern incorrect if desired and output differ
            if (clampOutput(outputNeurons(k)) != desiredOutput[tp](k)) patternCorrect = false;

			//calculate MSE
            mse += pow((outputNeurons(k) - desiredOutput[tp](k)), 2);
		}

		//if pattern is incorrect add to incorrect count
		if (!patternCorrect) incorrectPatterns++;

	}//end for

	 //if using batch learning - update the weights
	if (useBatch) updateWeights();

	//update training accuracy and MSE
    trainingSetAccuracy = 100 - (incorrectPatterns / trainingSet.size() * 100);
    trainingSetMSE = mse / (nOutput * trainingSet.size());
}


/*******************************************************************
* Save Neuron Weights to a file
********************************************************************/
bool saveWeights(char* filename)
{
	bool saved = false;

	try{
		//
		auto const expectedValueCount = (nInput + 1) * nHidden + (nHidden + 1) * nOutput;

		//
		std::unique_ptr<double[]> buffer(new double[expectedValueCount]);

		Eigen::Map<Eigen::MatrixXd> i_to_h(buffer.get(), nInput + 1, nHidden);
		i_to_h = wInputHidden;

		Eigen::Map<Eigen::MatrixXd> h_to_o(buffer.get() + ((nInput + 1)*nHidden), nHidden + 1, nOutput);
		h_to_o = wHiddenOutput;

		//
		std::ofstream outputFile(filename);
		outputFile << std::setprecision(5);
		std::copy( buffer.get(), buffer.get() + expectedValueCount
			     , std::ostream_iterator<double>(outputFile, " "));

	}
	catch (std::exception const& e) {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": " << e.what() << "\n";
	}
	catch (...) {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": unknown\n";
	}

	return saved;
}

#if 0
bool saveWeights(char* filename)
{
    //open file for reading
    std::fstream outputFile;
    outputFile.open(filename, std::ios::out);

    if ( outputFile.is_open() )
    {
        outputFile.precision(4);

        //output weights
        for ( int i=0; i < nInput; ++i )
        {
            for ( int j=0; j < nHidden; ++j )
            {
                outputFile << wInputHidden(i, j) << ",";
            }
        }

        for ( int i=0; i < nHidden + 1; ++i )
        {
            for ( int j=0; j < nOutput; ++j )
            {
                outputFile << wHiddenOutput(i, j);
                if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput )
                {
                    outputFile << ",";
                }
            }
        }

        //print success
        std::cout << std::endl << "Neuron weights saved to '" << filename << "'" << std::endl;

        //close file
        outputFile.close();

        return true;
    }
    else
    {
        std::cout << std::endl << "Error - Weight output file '" << filename << "' could not be created: " << std::endl;
        return false;
    }
}
#endif

/*******************************************************************
* Load Neuron Weights from a file
********************************************************************/
bool loadWeights(char* filename)
{
	bool valuesLoaded = false;

	try {
		//
		auto const expectedValueCount = (nInput + 1) * nHidden + (nHidden + 1) * nOutput;
		
		//
		std::vector<double> values;
		values.reserve(expectedValueCount);

		std::ifstream inputFile(filename, std::ios::binary);
		
		std::copy(std::istream_iterator<double>(inputFile), std::istream_iterator<double>()
			     , std::back_inserter(values) );

		if (values.size() == expectedValueCount) {
			valuesLoaded = true;

			// input -> hidden
			Eigen::Map<Eigen::MatrixXd> i_to_h_map(values.data(), (nInput + 1), nHidden);
			wInputHidden = i_to_h_map /*.transpose()*/;

			// hidden -> output
			Eigen::Map<Eigen::MatrixXd> h_to_o_map(values.data() + ((nInput + 1) * nHidden), (nHidden + 1), nOutput);
			wHiddenOutput = h_to_o_map /*.transpose()*/;
		}
		else {
			std::stringstream sout;
			sout << "expected " << expectedValueCount << " values, got " << values.size();
			throw std::logic_error(sout.str());
		}
	}
	catch (std::exception const& e) {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": " << e.what() << "\n";
	}
	catch (...) {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": unknown\n";
	}

	return valuesLoaded;
}

#if 0
bool loadWeights(char* filename)
{
    //open file for reading
    std::fstream inputFile;
    inputFile.open(filename, std::ios::in);

    if ( inputFile.is_open() )
    {
		auto expectedWeights = (nInput + 1) * nHidden + (nHidden + 1) * nOutput;
        Eigen::VectorXd weights(expectedWeights);
        std::string line = ""; //just initialization of the empty string which will be populated later on

		std::vector<double> weightsV;
		weightsV.reserve(expectedWeights);

        //read data
        while ( !inputFile.eof() )
        {
            getline(inputFile, line);

            //process line
            if (line.length() > 2 )
            {
                //store inputs
                char* cstr = new char[line.size()+1]; //length of a string in terms of bytes
                char* t;

                //we want to save line in cstr
                strcpy(cstr, line.c_str());

                //tokenise
                //int i = 0;
                t=strtok (cstr,",");

                while (t !=nullptr)
                {
                    //convert token of strings to a float representation
                    //float is then cast to double, for ex. "3.14" will become 3.14
                    weightsV.push_back( atof(t) );

                    //move token onwards
                    t = strtok(nullptr, ",");
                    //i++;
                }

                //free memory
                delete[] cstr;
            }
        }

        //check if sufficient weights were loaded
		if (weightsV.size() == expectedWeights) {
			Eigen::Map<Eigen::VectorXd> weightsMap(weightsV.data(), expectedWeights);
			weights = weightsMap;
		}
		else {
			// TODO:
		}
        //if ( weights.rows() != ( (nInput + 1) * nHidden + (nHidden +  1) * nOutput ) )
        //{
        //    std::cout << std::endl << "Error - Incorrect number of weights in input file: " << filename << std::endl;

        //    //close file
        //    inputFile.close();

        //    return false;
        //}
        //else
        //{
        //    //set weights
        //    int pos = 0;

        //    for ( int i=0; i < nInput; ++i)
        //    {
        //        for ( int j=0; j < nHidden; ++j)
        //        {
        //            wInputHidden(i, j) = weights(pos++);
        //        }
        //    }

        //    //carry on with the pos where we left
        //    for ( int i=0; i < nHidden; ++i)
        //    {
        //        for ( int j=0; j < nOutput; ++j)
        //        {
        //            wHiddenOutput(i, j) = weights(pos++);
        //        }
        //    }

        //    //print success
        //    std::cout << std::endl << "Neuron weights loaded successfuly from '" << filename << "'" << std::endl;

        //    //close file
        //    inputFile.close();

        //    return true;
        //}
    }
    else
    {
        std::cout << std::endl << "Error - Weight input file '" << filename << "' could not be opened: " << std::endl;
        return false;
    }
}
#endif

/*******************************************************************
* Return the NN accuracy on the set
********************************************************************/
double getSetAccuracy(std::vector<Eigen::VectorXd> trainingSet, const std::vector<Eigen::VectorXd>& desiredOutput)
{
    double incorrectResults = 0;

    //for every training input array
    for ( int tp = 0; tp < (int) trainingSet.size(); ++tp)
    {
        //feed inputs through network and backpropagate errors
        feedForward(trainingSet[tp]);

        //correct pattern flag
        bool correctResult = true;

        //check all outputs against desired output values
        for ( int k = 0; k < nOutput; ++k )
        {
            //set flag to false if desired and output differ
            if ( clampOutput(outputNeurons(k)) != desiredOutput[tp](k) ) correctResult = false;
        }

        //inc training error for a incorrect result
        if ( !correctResult ) incorrectResults++;

    }//end for

    //calculate error and return as percentage
    return 100 - (incorrectResults/trainingSet.size() * 100);
}


/*******************************************************************
* Return the NN mean squared error on the set
********************************************************************/
double getSetMSE(const std::vector<Eigen::VectorXd>& trainingSet, const std::vector<Eigen::VectorXd>& desiredOutput)
{
    double mse = 0;

    //for every training input array
    for (int tp = 0; tp < (int) trainingSet.size(); ++tp)
    {
        //feed inputs through network and backpropagate errors
        feedForward(trainingSet[tp]);

        //check all outputs against desired output values
        for ( int k = 0; k < nOutput; ++k )
        {
            //sum all the MSEs together
            mse += pow((outputNeurons(k) - desiredOutput[tp](k)), 2);
        }

    }//end for

    //calculate error and return as percentage
    return mse/(nOutput * trainingSet.size());
}

/*******************************************************************
* Set stopping parameters
********************************************************************/
void setStoppingConditions(int mEpochs, double dAccuracy)
{
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;
}

/*******************************************************************
* Train the NN using gradient descent
********************************************************************/
void trainNetwork(const std::vector<Eigen::VectorXd>& trainingSet, const std::vector<Eigen::VectorXd>& desiredOutput)
{
	std::cout << std::endl << " Neural Network Training Starting: " << std::endl
		<< "==========================================================================" << std::endl
		<< " LR: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << std::endl
		<< " " << nInput << " Input Neurons, " << nHidden << " Hidden Neurons, " << nOutput << " Output Neurons" << std::endl
		<< "==========================================================================" << std::endl << std::endl;

	//reset epoch and log counters
	epoch = 0;

	//train network using training dataset for training and generalization dataset for testing
	//--------------------------------------------------------------------------------------------------------
	while ((trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) && epoch < maxEpochs)
	{
		//store previous accuracy
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		runTrainingEpoch(trainingSet, desiredOutput);

		//get generalization set accuracy and MSE
		generalizationSetAccuracy = getSetAccuracy(trainingSet, desiredOutput);
		generalizationSetMSE = getSetMSE(trainingSet, desiredOutput);


		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		//if (ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy))
		//{
			std::cout << "Epoch :" << epoch;
			std::cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE;
			std::cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << std::endl;
		//}

		//once training set is complete increment epoch
		epoch++;

	}//end while

	 //get validation set accuracy and MSE
	validationSetAccuracy = getSetAccuracy(trainingSet, desiredOutput);
	validationSetMSE = getSetMSE(trainingSet, desiredOutput);

	//out validation accuracy and MSE
	std::cout << std::endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << std::endl;
	std::cout << " Validation Set Accuracy: " << validationSetAccuracy << std::endl;
	std::cout << " Validation Set MSE: " << validationSetMSE << std::endl << std::endl;
}