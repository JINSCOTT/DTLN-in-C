#include"onnxruntime.hpp"


bool OnnxInferenceBase::LoadWeights(OnnxENV* Env, const wchar_t* ModelPath) {
	//Set  session options
	sessionOptions.SetInterOpNumThreads(1);
	sessionOptions.SetIntraOpNumThreads(1);
	// Optimization will take time and memory during startup
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

	try {
		// Model path is const wchar_t*
		session = Ort::Session(Env->env, ModelPath, sessionOptions);
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	try {	// For allocating memory for input tensors
		memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	return true;
}

void OnnxInferenceBase::SetInputNodeNames(std::vector<const char*> names) {
	input_node_names = names;
}

void OnnxInferenceBase::SetOutputNodeNames(std::vector<const char*> names) {
	output_node_names = names;
}

void OnnxInferenceBase::SetInputDemensions(std::vector<std::vector<int64_t>> Dims) {
	input_node_dims = Dims;
}

model1::model1(OnnxENV* Env) {

	if (!LoadWeights(Env, ModelPath)) {
		throw std::runtime_error("Model 1 can not load weight");
	}

	std::vector<const char*> input_names{ input_2.c_str(), input_3.c_str() };
	SetInputNodeNames(input_names);
	std::vector<std::vector<int64_t>> Input_dims{ input_2_dim,input_3_dim };
	SetInputDemensions(Input_dims);
	std::vector<const char*> output_names{ activation_2.c_str(),tf_op_layer_stack_2.c_str() };
	SetOutputNodeNames(output_names);
}

int model1::Inference(std::vector < std::vector<float>*>& input, std::vector<std::vector<float >*>& output) {
	std::vector<Ort::Value>InputTensor, OutputTensor;
	for (int i = 0; i < input.size(); i++) {	// push inputs
		try {

			InputTensor.emplace_back(Ort::Value::CreateTensor<float>(this->memory_info, (float*)input[i]->data(), input[i]->size(), this->input_node_dims.at(i).data(), input_node_dims.at(i).size()));
		}
		catch (Ort::Exception oe) {
			std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
			return false;
		}
	}
	try { // execute
		OutputTensor = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), InputTensor.data(), InputTensor.size(), output_node_names.data(), output_node_names.size());
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}

	// push outputs
	for (int i = 0; i < OutputTensor.size(); i++) {
		memcpy(output[i]->data(), OutputTensor.at(i).GetTensorMutableData<float>(), OutputTensor.at(i).GetTensorTypeAndShapeInfo().GetElementCount());
	}
	return true;
}

model2::model2(OnnxENV* Env) {
	if (!LoadWeights(Env, ModelPath)) {
		throw std::runtime_error("Model 2 can not load weight");
	}

	std::vector<const char*> input_names{ input_4.c_str(), input_5.c_str() };
	SetInputNodeNames(input_names);
	std::vector<std::vector<int64_t>> Input_dims{ input_4_dim,input_5_dim };
	SetInputDemensions(Input_dims);
	std::vector<const char*> output_names{ conv1d_3.c_str(),tf_op_layer_stack_5.c_str() };
	SetOutputNodeNames(output_names);

}

int model2::Inference(std::vector<std::vector<float>*>& input, std::vector< std::vector<float>*>& output) {
	std::vector<Ort::Value>InputTensor, OutputTensor;
	for (int i = 0; i < input.size(); i++) {	// push inputs
		try {

			InputTensor.emplace_back(Ort::Value::CreateTensor<float>(this->memory_info, (float*)input[i]->data(), input[i]->size(), this->input_node_dims.at(i).data(), input_node_dims.at(i).size()));
		}
		catch (Ort::Exception oe) {
			std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
			return false;
		}
	}

	try { // execute
		OutputTensor = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), InputTensor.data(), InputTensor.size(), output_node_names.data(), output_node_names.size());
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}


	// push outputs
	for (int i = 0; i < OutputTensor.size(); i++) {
		memcpy(output[i]->data(), OutputTensor.at(i).GetTensorMutableData<float>(), OutputTensor.at(i).GetTensorTypeAndShapeInfo().GetElementCount());
	}
	return true;
}