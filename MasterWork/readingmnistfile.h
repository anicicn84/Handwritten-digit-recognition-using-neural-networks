#pragma once
#include <Eigen/Dense>
#include "logFunctionCall.h"

using Eigen::MatrixXd; //a matrix of arbitrary size

std::string trainingExamplesPath = "C:\\Users\\Nikola\\Documents\\MasterWork2016QT\\train-images.idx3-ubyte";
std::string trainingLabelsPath = "C:\\Users\\Nikola\\Documents\\MasterWork2016QT\\train-labels.idx1-ubyte";

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>

#include <cstdint>

template<class IntT>

IntT flip_bytes(IntT in_x)

{
	IntT out_x;

	//pointing to first 8bit chunk of in_x memory
	std::int8_t* pin_x = reinterpret_cast<std::int8_t*>(&in_x);

	//pointing to the last 8bit chunk of out_x memory
	std::int8_t* pout_x = reinterpret_cast<std::int8_t*>(&out_x) + (sizeof(IntT) - 1);


	//reversing the 8bit memory chunks of in_x, saving to out_x
	for (std::size_t i = 0; i < sizeof(IntT); ++i)

	{
		*pout_x = *pin_x;

		++pin_x;

		--pout_x;
	}

	//returning the modified endian(big to little or little to big)
	return out_x;

}


//reading sizeof(IntT) bytes to istream

template<class IntT>
IntT read_int_t(std::istream& str)
{
	IntT x;

	//extracting sizeof(IntT) characters and store into array pointed by &x (pointer)
	//if IntT is 32bit integer then sizeof(IntT) == 4 and 4 bytes of memory is stored to address od x
	str.read(reinterpret_cast<char*>(&x), sizeof(IntT));
	return x;
}

MatrixXd read_image_to_matrix(std::istream& str, int width, int height)
{
	MatrixXd image_matrix(width*height, 1);
	for (int i = 0; i < width*height; ++i)
	{
		image_matrix(i, 0) = read_int_t<std::uint8_t>(str);
	}
	return image_matrix;
}


//read an image 28x28 pixels to std::vector
std::vector<std::uint8_t> read_image(std::istream& str, int width, int height)

{
	std::vector<std::uint8_t> pixels;

	pixels.reserve(width*height);

	for (int i = 0; i < width*height; ++i) {
		auto pixel = read_int_t<std::uint8_t>(str);
		pixels.push_back(pixel);
	}
	return pixels;
}

//read an image 28x28 pixels to Eigen vector, there's alternate solution above to MatrixXd
Eigen::VectorXd read_image_to_vector(std::istream& str, int width, int height)
{
	Eigen::VectorXd pixels;

	for (int i = 0; i < width*height; ++i) {
		auto pixel = read_int_t<std::uint8_t>(str);
		pixels << pixel;
	}
	return pixels;
}


//just for visual purposes only - printin in txt file
void print_pixels(std::ostream& str, std::vector<std::uint8_t> const& pixels, int width, int height)

{
	for (int row = 0; row < height; ++row) {

		for (int col = 0; col < width; ++col) {

			auto pixel = pixels[row*width + col];

			char to_write = (pixel == 0u) ? ' ' : '#';

			str << to_write;
		}
		str << "\n";
	}
	str << "\n";

}
