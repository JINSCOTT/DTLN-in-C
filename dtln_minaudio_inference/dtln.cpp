#include "dtln.hpp"
DTLN::DTLN()
{
	int error = 0;
	// Set up dft 
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
	// Create model 1
	m1 = (struct model*)calloc(1, sizeof(struct model));
	if (m1 == NULL) {
		printf("allocate for model fail\n");
		throw std::runtime_error("model 1 allocate fail");
	}
	error = create_model1_onnx_tensor();
	if (error == NULL) {
		printf("Create tensor fail\n");
		throw std::runtime_error("model 1 tensorc create fail");
	}
	else {
		printf("create tensor success\n");
	}
	error = create_model1_onnx_attributes();
	error = setup_model1_onnx(m1);
	if (error == 0) {
		printf("set up mode error\n");
		throw std::runtime_error("model 1 setup fail");
	}
	else {
		printf("create model success\n");
	}
	// Create model 2
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
	// Inference model 1
	memcpy(input_2_array, in_mag.data(), 257 * sizeof(float));
	memcpy(input_3_array, state_1.data(), 512 * sizeof(float));
	inference_model(m1);
	memcpy(out_mask.data(), activation_2_array, 257 * sizeof(float));
	memcpy(state_1.data(), tf_op_layer_stack_2_array, 512 * sizeof(float));

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
	// Inference model 2
	printf("inference model 2\n");
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