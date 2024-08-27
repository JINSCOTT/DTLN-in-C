#include "dtln.hpp"
DTLN::DTLN()
{
	int error = 0;
	status = DftiCreateDescriptor(&handle, DFTI_SINGLE,
		DFTI_REAL, 1, 512);
	if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
	{
		printf("Error: %s\n", DftiErrorMessage(status));
	}

	status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	status = DftiCommitDescriptor(handle);
	if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
	{
		printf("Error: %s\n", DftiErrorMessage(status));
	}
	//home brew

	m2 = (struct model*)calloc(1, sizeof(struct model));
	if (m2 == NULL) {
		printf("allocate for model fail\n");
		throw std::runtime_error("model 2 allocate fail");
	}

	error = create_model2_onnx_tensor();
	if (error == NULL) {
		printf("Create tensor fail\n");
		throw std::runtime_error("model 2 tensorc create fail");
	}
	else {
		printf("create tensor success\n");
	}

	error = create_model2_onnx_attributes();
	error = setup_model2_onnx(m2);
	if (error == 0) {
		printf("set up mode error\n");
		throw std::runtime_error("model 2 setup fail");
	}
	else {
		printf("create model success\n");
	}
}

// Should incorporate more error checks
bool DTLN::inference(std::vector<float>& input, std::vector<float>& output)
{
	int error = 0;
	std::vector<std::vector<float>*> input_vec, output_vec;
	if (model_1 == NULL) {
		try {
			model_1 = new model1(&Env);
		}
		catch (std::exception& e) {
			return 0;
		}
	}
	if (model_2 == NULL) {
		try {
			model_2 = new model2(&Env);
		}
		catch (std::exception& e) {
			return 0;
		}
	}
	// check IO size
	if (input.size() < BLOCK_SHIFT || output.size() < BLOCK_LENGTH)
	{
		return false;
	}
	std::rotate(in_buffer.begin(), in_buffer.begin() + BLOCK_SHIFT, in_buffer.end());                   // shift
	std::memcpy(in_buffer.data() + BLOCK_LENGTH - BLOCK_SHIFT, input.data(), BLOCK_SHIFT * sizeof(float)); // copy to back

	// np.fft.rfft
	status = DftiComputeForward(handle, in_buffer.data(), fd.data());
	if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
	{
		printf("Error: %s\n", DftiErrorMessage(status));
	}
	for (int i = 0; i < BLOCK_FFT; i++)
	{
		in_mag[i] = std::abs(fd[i]);   // np.abs
		in_phase[i] = std::arg(fd[i]); // np.angle
	}
	// set input
	input_vec.push_back(&in_mag);
	input_vec.push_back(&state_1);
	output_vec.push_back(&out_mask);
	output_vec.push_back(&state_1);
	//execute
	error = model_1->Inference(input_vec, output_vec);
	// estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
	for (int i = 0; i < BLOCK_FFT; i++)
	{
		fd[i] = std::complex<float>(0, 1);
		fd[i] *= in_phase[i];
		fd[i] = std::exp(fd[i]);
		fd[i] *= in_mag[i] * out_mask[i];
	}
	status = DftiComputeBackward(handle, fd.data(), outblock.data());
	if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
	{
		printf("Error: %s\n", DftiErrorMessage(status));
	}


	// Normalize
	for (int i = 0; i < outblock.size(); i++)
	{
		outblock[i] /= (float)BLOCK_LENGTH;
	}
	//input_vec.clear();
	//output_vec.clear();
	//// set input
	//input_vec.push_back(&outblock);
	//input_vec.push_back(&state_2);
	//output_vec.push_back(&outblock);
	//output_vec.push_back(&state_2);
	// execute
	//model_2->Inference(input_vec, output_vec);

	memcpy(input_4_array, outblock.data(), 512 * sizeof(float));
	memcpy(input_5_array, state_2.data(), 512 * sizeof(float));
	inference_model(m2);
	memcpy(outblock.data(), conv1d_3_array, 512 * sizeof(float));
	memcpy(state_2.data(), tf_op_layer_stack_5_array, 512 * sizeof(float));

	std::rotate(out_buffer.begin(), out_buffer.begin() + BLOCK_SHIFT, out_buffer.end());
	for (int i = BLOCK_LENGTH - BLOCK_SHIFT; i < BLOCK_LENGTH; i++)
	{
		out_buffer[i] = 0.0f;
	}
	for (int i = 0; i < BLOCK_LENGTH; i++)
	{
		out_buffer[i] += outblock[i];
	}
	memcpy(output.data(), out_buffer.data(), output.size() * sizeof(float));
	return true;
}