#pragma once
#include "onnxruntime_cxx_api.h"

#include <iostream>
#include <vector>
// This have to be the first thing called
struct OnnxENV {
	Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
};

// Common functions
class OnnxInferenceBase {
public:
	// Settings

	// Create session
	bool LoadWeights(OnnxENV* Env, const wchar_t* ModelPath);
	void SetInputNodeNames(std::vector<const char*> input_node_names);
	void SetInputDemensions(std::vector<std::vector<int64_t>> input_node_dims);
	void SetOutputNodeNames(std::vector<const char*> input_node_names);
protected:
	Ort::Session session = Ort::Session(nullptr);
	Ort::SessionOptions sessionOptions;
	Ort::MemoryInfo memory_info{ nullptr };					// Used to allocate memory for input
	std::vector<const char*> output_node_names;	// output node names
	std::vector<const char*> input_node_names;	// Input node names
	std::vector<std::vector<int64_t>> input_node_dims;					// Input node dimension
	
};

// model specifics
class model1 :public OnnxInferenceBase {
public:
	model1(OnnxENV* Env);
	int Inference(std::vector < std::vector<float>*>& input, std::vector<std::vector<float>*>& OutputTensor);

private:
	const wchar_t* ModelPath = L"model_1_simp.onnx";
	// Inputs 
	std::string input_2 = "input_2";
	std::string input_3 = "input_3";
	std::vector<int64_t>input_2_dim = { 1,1,257 };
	std::vector<int64_t>input_3_dim = { 1,2,128,2 };
	// Outputs
	std::string activation_2 = "activation_2";
	std::string tf_op_layer_stack_2 = "tf_op_layer_stack_2";
};
	
class model2 :public OnnxInferenceBase {
public:
	model2(OnnxENV* Env);
	int Inference(std::vector<std::vector<float>*>& input, std::vector<std::vector<float>*>& OutputTensor);
private:
	const wchar_t* ModelPath = L"model_2_simp.onnx";
	// Inputs
	std::string input_4 = "input_4";
	std::string input_5 = "input_5";
	std::vector<int64_t>input_4_dim = { 1,1,512};
	std::vector<int64_t>input_5_dim = { 1,2,128,2 };
	// Outputs
	std::string conv1d_3 = "conv1d_3";
	std::string tf_op_layer_stack_5 = "tf_op_layer_stack_5";
	std::string conv1d_2 = "conv1d_2";
};