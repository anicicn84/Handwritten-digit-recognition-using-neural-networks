#include <iostream>
#include "readingmnistfile.h"
#include "networkcomputing.h"
#include <Eigen/Dense>
#include <bitset>
#include <random>

using Eigen::MatrixXd; //a matrix of arbitrary size

int generateRandomNumber(int minValue, int maxValue)
{
	std::random_device rd;				//used for seeding random generator
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> distrH(minValue, maxValue);  //random number between minValue and maxValue
	return distrH(mt);
}

int main()
{

	std::string myImageThree = "C:\\Users\\Nikola\\Desktop\\images\\myFile.idx3-ubyte";
	std::ifstream file(trainingExamplesPath, std::ios::binary);
	std::ifstream fileLabels(trainingLabelsPath, std::ios::binary);
	std::ifstream myImage(myImageThree, std::ios::binary);
	//    std::int32_t image_index;

	bool endian_flag_labels = false; //flag for different endian computers to work in the same manner
	bool endian_flag = false;

	std::int32_t magic_number_labels_temp = read_int_t<std::int32_t>(fileLabels);
	std::int32_t magic_number_temp = read_int_t<std::int32_t>(file);

	//magic number from MNIST database
	//determining this number we should know should we or should we not flip the bytes
	if (flip_bytes(magic_number_labels_temp) == 0x801)
	{
		endian_flag_labels = true;
	}
	if (flip_bytes(magic_number_temp) == 0x803)
	{
		endian_flag = true;
	}

	//this value in mnist is 32bit value
	std::int32_t magic_number_labels = ((endian_flag_labels) ? (flip_bytes(magic_number_labels_temp))
		: magic_number_labels_temp);

	std::int32_t image_count_labels = ((endian_flag_labels) ? flip_bytes(read_int_t<std::int32_t>(fileLabels))
		: read_int_t<std::int32_t>(fileLabels));

	std::cout << "Writing label\'s magic number in decimal format: " << magic_number_labels <<
		"\nand labels image count in decimal format: " << image_count_labels << "\n";


	//      std::ofstream out_file_labels("C:\\Users\\Nikola\\Desktop\\images\\labels.txt");
	//      for(image_index = 0; image_index < image_count_labels; ++ image_index){
	//          auto labels = read_int_t<std::uint8_t>(fileLabels);
	//          out_file_labels << unsigned(labels) <<"\n";
	//      }


	std::int32_t magic_number = (endian_flag) ? flip_bytes(magic_number_temp)
		: magic_number_temp;

	std::int32_t image_count = (endian_flag) ? flip_bytes(read_int_t<std::int32_t>(file))
		: read_int_t<std::int32_t>(file);

	std::int32_t image_height = (endian_flag) ? flip_bytes(read_int_t<std::int32_t>(file))
		: read_int_t<std::int32_t>(file);

	std::int32_t image_width = (endian_flag) ? flip_bytes(read_int_t<std::int32_t>(file))
		: read_int_t<std::int32_t>(file);



	std::cout << "Writing training example\'s magic number and params in decimal format: " << std::dec << magic_number << "\n";

	std::cout << "and image count of training examples in decimal format: " << image_count<< "\n";

	std::cout << "image height in decimal format: " << image_height << "\n";

	std::cout << "and image width in decimal format: " << image_width << "\n";




	/*All this when we want to train our network, no need for this if it's already trained*/


	//int i = 0;
	//std::vector<Eigen::VectorXd> trainingSet;
	//trainingSet.reserve(60000);
	//for ( ; i < 60000; ++i) {
	//	trainingSet.push_back(read_image_to_vector(file, image_height, image_width));
	//}


	/*Desired output will be an array of zeros and one 1 which is on label index place.
	If the label value is for example 5, desired output will be 0000010000*/


	//i = 0;
	//std::vector<Eigen::VectorXd> desiredOutput;
	//desiredOutput.reserve(60000);
	//for (; i < 60000; ++i) {
	//	std::uint8_t label = read_int_t<std::uint8_t>(fileLabels);
	//	//unsigned long label = std::bitset<8>(pixel).to_ulong();
	//	Eigen::VectorXd matrix = Eigen::VectorXd::Zero(10);
	//	matrix(label, 0) = 1;
	//	desiredOutput.push_back(matrix);
	//}



	/*We don't want to work with all the data, we choose 10.000 random samples from 60.000 samples*/
	/*We're going to use those 10.000 elements set to train our network*/


	//i = 0;
	//std::vector<Eigen::VectorXd> trainingSet_10000_Sample;
	//std::vector<Eigen::VectorXd> desiredOutput_10000_Sample;

	//trainingSet_10000_Sample.reserve(10000);
	//desiredOutput_10000_Sample.reserve(10000);


	/*This lambda function is used for scaling the input and to have input values between
	0 and 1 and not between 0-255. Instead of dividing by 255, it's faster to multiply by 0.003921568*/


	//auto scaleInput = [](double matrixElement) {
	//	return matrixElement * 0.003921568;
	//};

	//for (; i < 10000; ++i) {
	//	int randomNum = generateRandomNumber(0, 59999);
	//	trainingSet_10000_Sample.push_back(trainingSet[randomNum].unaryExpr(scaleInput));
	//	desiredOutput_10000_Sample.push_back(desiredOutput[randomNum]);
	//}

	//setStoppingConditions(150, 90);
	//initializeWeights();
	//trainNetwork(trainingSet_10000_Sample, desiredOutput_10000_Sample);
	//saveWeights("Weights");
	
	/********************************************************************************/
	loadWeights("Weights");
	Eigen::VectorXd image(784);
	image = read_image_to_vector(myImage, image_width, image_height);
	feedForward(image);
	int recogizedDigit = -1;
	double max = outputNeurons(0);
	int maxIndex = 0;
	for (int i = 0; i < nOutput; ++i) {
		if (outputNeurons(i) > max) {
			max = outputNeurons(i);
			maxIndex = i;
		}
	}

	std::cout << "Recognized number is: " << maxIndex << std::endl;

	/****************************************************************************/
	/*******************VECTOR OF LABELS IN UINT8 FORMAT*************************/

	/*std::vector<std::uint8_t> desiderOutput;
	desiderOutput.reserve(60000);
	for (int i = 0; i < 60000; ++i) {
		desiderOutput.push_back(read_int_t<std::uint8_t>(fileLabels));
	}*/


	/****************************************************************************/
	/*******************PRINT HANDWRITTEN NUMBERS TO TXT FILE********************/

	//std::ofstream out_file("C:\\Users\\Nikola\\Desktop\\images\\images.txt");
	//for(int image_index = 0; image_index < image_count; ++ image_index){
	//    auto pixels = read_image(myImage, image_height, image_width);
	//    print_pixels(out_file, pixels, image_height, image_width);
	//}


	/****************************************************************************/
	/*******************PRINT LABELS TO TXT FILE IN BINARY FORMAT****************/

	/*std::ofstream out_file("C:\\Users\\Nikola\\Desktop\\images.txt");
	for (int image_index = 0; image_index < image_count; ++image_index) {
		auto pixels = read_int_t<std::uint8_t>(fileLabels);
		unsigned long label = std::bitset<8>(pixels).to_ulong();
		Eigen::RowVectorXd matrix = Eigen::RowVectorXd::Zero(10);
		matrix(0, label) = 1;
		out_file << matrix;
		out_file << "\n";
	}*/

	//initializeWeights();
	//std::cout << wInputHidden << "\n";

	system("pause");
	return 0;
}

