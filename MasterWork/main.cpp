#include <iostream>
#include "readingmnistfile.h"
#include "networkcomputing.h"
#include <Eigen/Dense>

using Eigen::MatrixXd; //a matrix of arbitrary size

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



	std::ofstream out_file("C:\\Users\\Nikola\\Desktop\\images\\images.txt");
	//for(int image_index = 0; image_index < image_count; ++ image_index){
	    auto pixels = read_image(myImage, image_height, image_width);
	    print_pixels(out_file, pixels, image_height, image_width);
	//}

	//initializeWeights();
	//std::cout << wInputHidden << "\n";

	system("pause");
	return 0;
}

