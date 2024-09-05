// Parse command line
// -i model_2.onnx - o model_2.onnx - input_name input_4 input_5 - input_shape 1 1 512, 1 2 128 2 - input_type 1 1
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "utility.hpp"


class commandline_input{
public:
	void parse(const int argc,char *argv[]);
	std::string get_model_name()const ;
	std::string get_output_name() const;
	std::vector<std::vector<int64_t>> get_input_shape() const;
	std::vector<std::string> get_input_name() const;
	std::vector<int64_t> get_Input_type() const;
	void print();

private:

	std::string input_file_name_;
	std::string output_file_name_;
	std::vector<std::string> input_tensor_name_;
	std::vector<int64_t> input_type_;
	std::vector<std::vector<int64_t>> input_tensor_shape_;

};