#include "linkedlist.h"
#include "tensor.h"
#include "ops.h"
#include "node.h"
void list_test() {
	struct list* test_list = calloc(1, sizeof(struct list));
	int64_t a = 10;
	float   b = 100.0f;
	char* c = "hello";
	push_back_list(test_list, &a);
	push_back_list(test_list, &b);
	push_back_list(test_list, c);
	printf("a: %lld\n", *(int64_t*)get_data_list(test_list, 0));
	printf("b: %f\n", *(float*)get_data_list(test_list, 1));
	printf("c: %s\n", (char*)get_data_list(test_list, 2));

}

int16_t tensor_Test() {
	float array1[100];
	for (int i = 0; i < 100; i++) {
		array1[i] = i;
	}
	printf("%f\n", array1[2]);
	int64_t dims[3] = { 10,5,2 };

	int64_t dims1[3] = { 2,5,10 };

	int64_t dims2[3] = { 1,2 };
	struct tensor* tensor = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensor_copy = create_tensor_copy(tensor);
	struct tensor* tensor1 = create_tensor(array1, 100, dims1, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensor2 = create_tensor(array1, 2, dims2, 2, DATATYPE_FLOAT32, true);
	if (tensor_copy == NULL) {
		printf("tensor copy failed\n");
		return 0;
	}
	int64_t* larger_dim = NULL;
	if (is_shape_compatible_tensor(tensor, tensor_copy)) {
		printf("Is compatible\n");
	}
	else {
		printf("Not compatible\n");
	}

	if (is_shape_broadcastable_tensor(tensor, tensor_copy, &larger_dim) > 0) {
		printf("Is broadcastable\n");
		for (int i = 0; i < 3; i++) {
			printf("%lld ", larger_dim[i]);
		}
		printf("\n");
	}
	else {
		printf("Not broadcastable\n");
	}

	if (is_shape_compatible_tensor(tensor, tensor1)) {
		printf("Is compatible\n");

	}
	else {

		printf("Not compatible\n");
	}

	if (is_shape_broadcastable_tensor(tensor, tensor1, &larger_dim) > 0) {
		printf("Is broadcastable\n");
		for (int i = 0; i < 3; i++) {
			printf("%lld ", larger_dim[i]);
		}
		printf("\n");
	}
	else {
		printf("Not broadcastable\n");
	}

	if (is_shape_compatible_tensor(tensor, tensor2)) {
		printf("Is compatible\n");

	}
	else {

		printf("Not compatible\n");
	}

	if (is_shape_broadcastable_tensor(tensor, tensor2, &larger_dim) > 0) {
		printf("Is broadcastable\n");
		for (int i = 0; i < 3; i++) {
			printf("%lld ", larger_dim[i]);
		}
		printf("\n");
	}
	else {
		printf("Not broadcastable\n");
	}

	struct tensor_iterator* it = create_tensor_iterator(tensor);
	// original
	printf("0 point\n");
	print_tensor_iter(it);
	// 50
	printf("\n\nmove to 50\n");
	int error = goto_1d_tensor_iter(it, 50);
	print_tensor_iter(it);
	if (error <= 0) {
		printf("goto_1d fail\n");
	}
	printf("\n\nmove to 51\n");
	next_tensor_iter(it);
	print_tensor_iter(it);
	printf("\n\nmove to 52\n");
	next_tensor_iter(it);
	print_tensor_iter(it);
	int64_t last[3] = { 9,4,1 };
	printf("\n\nmove to 9,4,1\n");
	goto_tensor_iter(it, last);
	print_tensor_iter(it);
	printf("\n\nreset\n");
	reset_tensor_iter(it);
	print_tensor_iter(it);

	for (int i = 0; i < 102; i++) {
		if (is_not_done_tensor_iter(it)) {
			next_tensor_iter(it);
			printf("%d: not done\n", i);
		}
		else {
			printf("%d: is done\n", i);
			break;
		}
	}
	release_tensor_iterator(it);
	release_tensor(tensor);
	release_tensor(tensor_copy);
	release_tensor(tensor1);
	release_tensor(tensor2);
	return 1;
}



void broadcast_test() {
	int error = 0;
	float array1[100];
	float array2[1] = { 10 };
	for (int i = 0; i < 100; i++) {
		array1[i] = i;
	}
	int64_t dims[3] = { 10,5,2 };
	int64_t dimsb[1] = { 1 };


	struct tensor* tensorA = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 1, dimsb, 1, DATATYPE_FLOAT32, true);
	struct tensor* tensorC = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	print_tensor(tensorA);

	error = broadcast_function(tensorA, tensorB, tensorC, Add);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	print_tensor(tensorC);

	error = broadcast_function(tensorA, tensorA, tensorC, Add);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	print_tensor(tensorC);

	error = broadcast_function(tensorA, tensorB, tensorC, Sub);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	print_tensor(tensorC);

	error = broadcast_function(tensorA, tensorB, tensorC, Mul);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	print_tensor(tensorC);

	error = broadcast_function(tensorA, tensorB, tensorC, Div);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	print_tensor(tensorC);
}

void tanh_test() {
	printf("\n\ntanh test\n");
	int error = 0;
	float array1[100], array2[100] = { 0 };
	for (int i = 0; i < 100; i++) {
		array1[i] = (float)i;
	}
	int64_t dims[3] = { 10,5,2 };

	struct tensor* tensorA = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 100, dims, 3, DATATYPE_FLOAT32, true);
	printf("A tensor: \n");
	print_tensor(tensorA);
	printf("B tensor: \n");
	print_tensor(tensorB);

	error = tanh_function(tensorA, tensorB);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else return;
	printf("result: \n");
	print_tensor(tensorB);
	printf("tanh test done \n\n");
}

void sqrt_test() {
	printf("\n\nsqrt test\n");
	int error = 0;
	float array1[100], array2[100] = { 0 };
	for (int i = 0; i < 100; i++) {
		array1[i] = (float)i;
	}
	int64_t dims[3] = { 10,5,2 };

	struct tensor* tensorA = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 100, dims, 3, DATATYPE_FLOAT32, true);
	printf("A tensor: \n");
	print_tensor(tensorA);
	printf("B tensor: \n");
	print_tensor(tensorB);

	error = sqrt_function(tensorA, tensorB);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else return;
	printf("result: \n");
	print_tensor(tensorB);
	printf("sqrt test done \n\n");
}
void relu_test() {
	printf("\n\relu test\n");
	int error = 0;
	float array1[100], array2[100] = { 0 };
	for (int i = 0; i < 100; i++) {
		array1[i] = (float)i;
	}
	int64_t dims[3] = { 10,5,2 };

	struct tensor* tensorA = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 100, dims, 3, DATATYPE_FLOAT32, true);
	printf("A tensor: \n");
	print_tensor(tensorA);
	printf("B tensor: \n");
	print_tensor(tensorB);

	error = relu_function(tensorA, tensorB);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else return;
	printf("result: \n");
	print_tensor(tensorB);
	printf("relu test done \n\n");
}
void sigmoid_test() {
	printf("\n\nsigmoid test\n");
	int error = 0;
	float array1[100], array2[100] = { 0 };
	for (int i = 0; i < 100; i++) {
		array1[i] = (float)i;
	}
	int64_t dims[3] = { 10,5,2 };

	struct tensor* tensorA = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 100, dims, 3, DATATYPE_FLOAT32, true);
	printf("A tensor: \n");
	print_tensor(tensorA);
	printf("B tensor: \n");
	print_tensor(tensorB);

	error = sigmoid_function(tensorA, tensorB);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else return;
	printf("result: \n");
	print_tensor(tensorB);
	printf("sigmoid test done \n\n");
}
void copy_test() {
	printf("\n\copy test\n");
	int error = 0;
	float array1[100], array2[100] = { 0 };
	for (int i = 0; i < 100; i++) {
		array1[i] = (float)i;
	}
	int64_t dims[3] = { 10,5,2 };

	struct tensor* tensorA = create_tensor(array1, 100, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 100, dims, 3, DATATYPE_FLOAT32, true);
	printf("A tensor: \n");
	print_tensor(tensorA);
	printf("B tensor: \n");
	print_tensor(tensorB);

	error = copy_function(tensorA, tensorB);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else return;
	printf("result: \n");
	print_tensor(tensorB);
	printf("vopy test done \n\n");
}
void slice_test() {
	printf("\n\nslice test\n");
	int error = 0;
	float array1[8] = { 1,2,3,4,5,6,7,8 }, array2[3] = { 0 };

	int64_t dims_1[4] = { 2,4 };
	int64_t dims_2[4] = { 1,3 };
	struct tensor* tensorA = create_tensor(array1, 8, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 3, dims_2, 2, DATATYPE_FLOAT32, true);

	int64_t starts[2] = { 1,0 };

	int64_t ends[2] = { 2,3 };

	int64_t steps[2] = { 1,2 };
	int64_t axis[2] = { 0,1 };
	error = slice_function(tensorA, tensorB, starts, ends, axis, steps);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else {
		if (error == OPS_UNDEFINED) {
			printf("OPS_UNDEFINED\n");
		}
		else if (error == OPS_INPUT_IS_NULL) {
			printf("OPS_INPUT_IS_NULL\n");
		}
		else if (error == OPS_NOT_BROADCASTABLE) {
			printf("OPS_NOT_BROADCASTABLE\n");
		}
		else if (error == OPS_DIMENSION_MISMATCH) {
			printf("OPS_DIMENSION_MISMATCH\n");
		}
		else if (error == OPS_ALLOCATION_FAIL) {
			printf("OPS_ALLOCATION_FAIL\n");
		}
		else if (error == OPS_TYPE_UNIMPLEMENTED) {
			printf("OPS_TYPE_UNIMPLEMENTED\n");
		}
		else if (error == OPS_TYPE_NOT_SUPPORTED) {
			printf("OPS_TYPE_NOT_SUPPORTED\n");
		}
		else {
			printf("Something is very wrong\n");
		}

		printf("OPS fail\n");
		return;
	}
	printf("result: \n");
	print_tensor(tensorB);
	printf("slice test done \n\n");
}

void transpose_f_test() {
	printf("\n\ntranspose test\n");
	int error = 0;
	float array1[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 }, array2[12] = { 0 };

	int64_t dims_1[4] = { 2,2,3 };
	int64_t dims_2[4] = { 2,3,2 };
	struct tensor* tensorA = create_tensor(array1, 12, dims_1, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 12, dims_2, 3, DATATYPE_FLOAT32, true);

	int64_t perm[3] = { 0,2,1 };


	error = transpose_function(tensorA, tensorB, perm);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else {
		if (error == OPS_UNDEFINED) {
			printf("OPS_UNDEFINED\n");
		}
		else if (error == OPS_INPUT_IS_NULL) {
			printf("OPS_INPUT_IS_NULL\n");
		}
		else if (error == OPS_NOT_BROADCASTABLE) {
			printf("OPS_NOT_BROADCASTABLE\n");
		}
		else if (error == OPS_DIMENSION_MISMATCH) {
			printf("OPS_DIMENSION_MISMATCH\n");
		}
		else if (error == OPS_ALLOCATION_FAIL) {
			printf("OPS_ALLOCATION_FAIL\n");
		}
		else if (error == OPS_TYPE_UNIMPLEMENTED) {
			printf("OPS_TYPE_UNIMPLEMENTED\n");
		}
		else if (error == OPS_TYPE_NOT_SUPPORTED) {
			printf("OPS_TYPE_NOT_SUPPORTED\n");
		}
		else {
			printf("Something is very wrong\n");
		}

		printf("OPS fail\n");
		return;
	}
	printf("result: \n");
	print_tensor(tensorB);
	printf("transpose test done \n\n");
}

void matmul_test() {
	printf("\n\n matmul test\n");
	int error = 0;
	float array1[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array2[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array3[8] = { 4,1,2,2 };

	int64_t dims_1[3] = { 2,2,4 };
	int64_t dims_2[3] = { 2,4,2 };
	int64_t dims_3[3] = { 2,2 ,2 };
	struct tensor* tensorA = create_tensor(array1, 16, dims_1, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 16, dims_2, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensorC = create_tensor(array3, 8, dims_3, 3, DATATYPE_FLOAT32, true);



	error = matmul_function(tensorA, tensorB, tensorC);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else {
		if (error == OPS_UNDEFINED) {
			printf("OPS_UNDEFINED\n");
		}
		else if (error == OPS_INPUT_IS_NULL) {
			printf("OPS_INPUT_IS_NULL\n");
		}
		else if (error == OPS_NOT_BROADCASTABLE) {
			printf("OPS_NOT_BROADCASTABLE\n");
		}
		else if (error == OPS_DIMENSION_MISMATCH) {
			printf("OPS_DIMENSION_MISMATCH\n");
		}
		else if (error == OPS_ALLOCATION_FAIL) {
			printf("OPS_ALLOCATION_FAIL\n");
		}
		else if (error == OPS_TYPE_UNIMPLEMENTED) {
			printf("OPS_TYPE_UNIMPLEMENTED\n");
		}
		else if (error == OPS_TYPE_NOT_SUPPORTED) {
			printf("OPS_TYPE_NOT_SUPPORTED\n");
		}
		else {
			printf("Something is very wrong\n");
		}

		printf("OPS fail\n");
		return;
	}
	printf("result: \n");
	print_tensor(tensorC);
	printf("matmul test done \n\n");
}

void gemm_test() {
	printf("\n\n gemm test\n");
	int error = 0;
	float array1[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array2[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array3[16] = { 4,1,2,2 };

	int64_t dims_1[2] = { 4,4 };
	int64_t dims_2[2] = { 4,4 };
	int64_t dims_3[2] = { 4,4 };
	struct tensor* tensorA = create_tensor(array1, 16, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 16, dims_2, 2, DATATYPE_FLOAT32, true);
	struct tensor* output = create_tensor(array3, 16, dims_3, 2, DATATYPE_FLOAT32, true);



	error = gemm_function(tensorA, tensorB, NULL, output, 1.0f, 1.0f, 0, 0);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else {
		if (error == OPS_UNDEFINED) {
			printf("OPS_UNDEFINED\n");
		}
		else if (error == OPS_INPUT_IS_NULL) {
			printf("OPS_INPUT_IS_NULL\n");
		}
		else if (error == OPS_NOT_BROADCASTABLE) {
			printf("OPS_NOT_BROADCASTABLE\n");
		}
		else if (error == OPS_DIMENSION_MISMATCH) {
			printf("OPS_DIMENSION_MISMATCH\n");
		}
		else if (error == OPS_ALLOCATION_FAIL) {
			printf("OPS_ALLOCATION_FAIL\n");
		}
		else if (error == OPS_TYPE_UNIMPLEMENTED) {
			printf("OPS_TYPE_UNIMPLEMENTED\n");
		}
		else if (error == OPS_TYPE_NOT_SUPPORTED) {
			printf("OPS_TYPE_NOT_SUPPORTED\n");
		}
		else {
			printf("Something is very wrong\n");
		}

		printf("OPS fail\n");
		return;
	}
	printf("result: \n");
	print_tensor(output);
	printf("gemm test done \n\n");
}



void paddin_test() {
	printf("\n\padding test\n");
	int error = 0;
	float array1[9] = { 1,2,3,4,5,6,7,8,9 }, array2[49] = { 0 };

	int64_t dims_1[2] = { 3,3 };
	int64_t dims_2[2] = { 7,7 };
	struct tensor* tensorA = create_tensor(array1, 9, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 49, dims_2, 2, DATATYPE_FLOAT32, true);
	print_tensor(tensorA);
	int64_t padding[4] = { 2,1,2,3 };

	float val = 13.0f;
	char c[] = "constant";
	error = pad_function_simple(tensorA, tensorB, padding, c, &val);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else {
		if (error == OPS_UNDEFINED) {
			printf("OPS_UNDEFINED\n");
		}
		else if (error == OPS_INPUT_IS_NULL) {
			printf("OPS_INPUT_IS_NULL\n");
		}
		else if (error == OPS_NOT_BROADCASTABLE) {
			printf("OPS_NOT_BROADCASTABLE\n");
		}
		else if (error == OPS_DIMENSION_MISMATCH) {
			printf("OPS_DIMENSION_MISMATCH\n");
		}
		else if (error == OPS_ALLOCATION_FAIL) {
			printf("OPS_ALLOCATION_FAIL\n");
		}
		else if (error == OPS_TYPE_UNIMPLEMENTED) {
			printf("OPS_TYPE_UNIMPLEMENTED\n");
		}
		else if (error == OPS_TYPE_NOT_SUPPORTED) {
			printf("OPS_TYPE_NOT_SUPPORTED\n");
		}
		else {
			printf("Something is very wrong\n");
		}

		printf("OPS fail\n");
		return;
	}
	printf("result: \n");
	print_tensor(tensorA);
	print_tensor(tensorB);
	printf("transpose test done \n\n");
}

void fgemm_test() {
	printf("\n\n fgemm test\n");
	int error = 0;
	float array1[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	float array2[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	float array3[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	int64_t m = 4, n = 4, k = 4;
	fgemm(0, 0, 4, 4, 4, 1.0f, array1, m, array2, n, 1.0f, array3, k);


	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%f ", array3[i * 4 + j]);
		}
		printf("\n");

	}
	printf("fgemm test done \n\n");
}

//1, 2, 3
//4, 2, 1
int conv1d_test() {
	float array1[8] = { 0,1,2,3,4,5,6,7 };
	float array2[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	float array3[16] = { 0 };

	conv1df(array1, array2, NULL, array3, 2, 4, 4, 1, 4, 1, 1);
	for (int i = 0; i < 16; i++) {
		printf("%f ", array3[i]);

	}
}

void lstm_Test() {
	float activation_a[3] = { 1.0f,1.0f,1.0f };
	float activation_b[3] = { 1.0f,1.0f,1.0f };
	char sigmoid[] = "Sigmoid";
	char tanh[] = "Tanh";
	char forward[] = "forward";
	struct list* act_list = malloc(sizeof(struct list));
	push_back_list(act_list, sigmoid);
	push_back_list(act_list, sigmoid);
	push_back_list(act_list, tanh);
	struct list* direction_list = malloc(sizeof(struct list));
	float clip = 0.0f;

	push_back_list(direction_list, forward);
	int64_t hidden_size = 4;
	int64_t input_forget = 0;
	//int64_t layout = 0;
	//float x_array[8] = { 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 }, w_array[32] = { 0 }, r_array[4] = {}, b_array[6] = {}, b_array[6] = {}, b_array[6] = {}, b_array[6] = {}, b_array[6] = {}, b_array[6] = {}, b_array[6] = {};

	//int64_t x_dims[3] = { 4,1,2 }, w_dims[3] = { 1,4*4,2 }, dims_1[2] = { 3,3 }, dims_1[2] = { 3,3 }, dims_1[2] = { 3,3 }, dims_1[2] = { 3,3 }, dims_1[2] = { 3,3 }, dims_1[2] = { 3,3 };





	//int64_t dims_1[2] = { 3,3 };
	//int64_t dims_2[2] = { 7,7 };
	//struct tensor* tensorA = create_tensor(array1, 9, dims_1, 2, DATATYPE_FLOAT32, true);
	//int lstm(float* activation_alpha, float* activation_beta, struct list* activations, float clip, struct list* direction, int64_t hidden_size, int64_t input_forget, int64_t layout,
	//	struct tensor* x, struct tensor* w, struct tensor* r, struct tensor* b, int64_t * seq_length, struct tensor* initial_h, struct tensor* initial_c, struct tensor* P, struct tensor* Y, struct tensor* Y_h, struct tensor* Y_c) {
}

int16_t broadcast_Test() {
	float array1[100], array2[] = { 2,2,2,2,2,2,2,2 };
	for (int i = 0; i < 100; i++) {
		array1[i] = i;
	}

	int64_t dims[3] = { 2,2,4 };

	int64_t dims1[3] = { 2,4 };

	struct tensor* tensor1 = create_tensor(array1, 16, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensor2 = create_tensor(array2, 8, dims1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensor3 = create_empty_tensor();



	struct node* newnode = create_add_node(tensor1, tensor2, tensor3);
	int error = inference_add_node(newnode);
	printf("Result %lld:\n", tensor3->data_size);
	for (int i = 0; i < tensor3->data_size; i++) {
		if (tensor3->type == DATATYPE_FLOAT32) {
			printf("%f ", ((float*)tensor3->data)[i]);
		}
	}
	printf("\n");
}

void dynamic_arry_Test() {
	struct dynamic_array* arr = create_array(sizeof(int64_t));
	for (int64_t i = 0; i < 10; i++) {
		pushback_array(arr, &i);
		printf("%lld ", *(int64_t*)get_item_array(arr, i));
	}
	printf("\n");
	int64_t i = 0;

	popfront_array(arr);

	popback_array(arr);


	for (int64_t i = 0; i < arr->size; i++) {
		printf("%lld ", *(int64_t*)get_item_array(arr, i));
	}
	printf("\n");
	pushfront_array(arr, &i);
	delete_item_array(arr, 8);
	for (int64_t i = 0; i < arr->size; i++) {
		printf("%ld ", *(int64_t*)get_item_array(arr, i));
	}

	printf("\n");
	printf("%ld ", *(int64_t*)front_array(arr));
	printf("%ld ", *(int64_t*)back_array(arr));
	release_array(arr);
}

void Squeeze_Unsqueeze_test() {

	float array1[60];
	for (int i = 0; i < 60; i++) {
		array1[i] = i;
	}
	int64_t dims[5] = { 10,1,2,1,3 };
	struct tensor* a = NULL, * b = NULL, * c = NULL;
	struct node* squeeze = NULL, * unsqueeze = NULL;
	struct tensor* squeeze_axes = NULL, * unsqueeze_axes = NULL;
	a = create_tensor(array1, 60, dims, 5, DATATYPE_FLOAT32, true);
	b = create_empty_tensor();
	c = create_empty_tensor();

	int64_t array_2[2] = { 1,3 };
	int64_t dims_2[1] = { 2 };
	int64_t array_3[2] = { 2,3 };
	int64_t dims_3[1] = { 2 };
	squeeze_axes = create_tensor(array_2, 2, dims_2, 1, DATATYPE_INT64, true);
	unsqueeze_axes = create_tensor(array_3, 2, dims_3, 1, DATATYPE_INT64, true);

	squeeze = create_squeeze_node(a, squeeze_axes, b);
	inference_squeeze_node(squeeze);
	printf("squeeze result dims: ");
	for (int i = 0; i < b->dimension_size; i++) {

		printf("%lld ", b->dimension[i]);
	}
	printf("\nsqueeze result: ");
	for (int i = 0; i < b->data_size; i++) {
		printf("%f ", ((float*)b->data)[i]);
	}
	printf("\n");


	unsqueeze = create_unsqueeze_node(b, unsqueeze_axes, c);
	inference_unsqueeze_node(unsqueeze);
	printf("unsqueeze result dims: ");
	for (int i = 0; i < c->dimension_size; i++) {

		printf("%ld ", c->dimension[i]);
	}
	printf("\nunsqueeze result: ");
	for (int i = 0; i < c->data_size; i++) {
		printf("%f ", ((float*)c->data)[i]);
	}
	printf("\n");
}

void transpose_test() {
	struct tensor* a = NULL, * b = NULL;
	struct list* PERM = NULL;
	float array1[60];
	for (int i = 0; i < 60; i++) {
		array1[i] = i;
	}
	int64_t dims[3] = { 2,5,6 };
	a = create_tensor(array1, 60, dims, 3, DATATYPE_FLOAT32, true);
	b = create_empty_tensor();

	int64_t transpose[3] = { 2,0,1 };
	PERM = create_list_from_array(transpose, 3, 8);

	struct node* transpose_node = create_transpose_node(transpose, a, b);
	inference_transpose_node(transpose_node);

	printf("transpose result dims: ");
	print_tensor(b);
}

int16_t matmul_node_test() {
	float array1[20], array2[] = { 2,2,2,2,2, };
	for (int i = 0; i < 20; i++) {
		array1[i] = i;
	}

	int64_t dims[3] = { 2,2,5 };

	int64_t dims1[2] = { 5,1 };

	struct tensor* tensor1 = create_tensor(array1, 20, dims, 3, DATATYPE_FLOAT32, true);
	struct tensor* tensor2 = create_tensor(array2, 5, dims1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensor3 = create_empty_tensor();

	struct node* newnode = create_matmul_node(tensor1, tensor2, tensor3);
	int error = inference_matmul_node(newnode);
	if (error != OPS_SUCCESS) {
		printf("Ops fail\n");
	}
	print_tensor(tensor3);
}

int16_t slice_node_test() {
	float array1[432], array2[] = { 2,2,2,2,2, };
	for (int i = 0; i < 432; i++) {
		array1[i] = i;
	}

	int64_t dims[4] = { 2,6,6,6 };

	struct tensor* tensor1 = create_tensor(array1, 432, dims, 4, DATATYPE_FLOAT32, true);
	struct tensor* tensor2 = create_empty_tensor();

	int64_t starts[4] = { 0,1,1,1 }, ends[4] = { 1,2,4,4 };
	int64_t sdim[1] = { 4 };
	struct tensor* tensor_starts = create_tensor(starts, 4, sdim, 1, DATATYPE_FLOAT32, true);
	struct tensor* tensor_ends = create_tensor(ends, 4, sdim, 1, DATATYPE_FLOAT32, true);

	struct node* newnode = create_slice_node(tensor1, tensor_starts, tensor_ends, NULL, NULL, tensor2);
	int error = inference_slice_node(newnode);
	if (error != OPS_SUCCESS) {
		printf("Ops fail\n");
	}
	print_tensor(tensor2);
}

int16_t gemm_node_Test() {
	printf("\n\n gemm test\n");
	int error = 0;
	float array1[] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array2[] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array3[25] = { 0,0,0,0, };

	int64_t dims_1[2] = { 3,4 };
	int64_t dims_2[2] = { 4,3 };
	int64_t dims_3[2] = { 3,3 };
	int64_t asize = dims_1[1] * dims_1[0];
	int64_t bsize = dims_2[1] * dims_2[0];
	struct tensor* tensorA = create_tensor(array1, asize, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, bsize, dims_2, 2, DATATYPE_FLOAT32, true);
	print_tensor(tensorA);
	struct tensor* output = create_empty_tensor();
	float alpha = 1, beta = 1;
	int64_t transposea = 1, transposeb = 1
		;
	struct node* newnode = create_gemm_node(&alpha, &beta, &transposea, &transposeb, tensorA, tensorB, NULL, output);
	error = inference_gemm_node(newnode);
	if (error != OPS_SUCCESS) {
		printf("Ops fail\n");
	}
	print_tensor(output);
}

int16_t concat_node_Test() {
	printf("\n\n concat test\n");
	int error = 0;
	int64_t AXIS = 0;
	float array1[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array2[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 }, array3[16] = { 4,1,2,2 };

	int64_t dims_1[2] = { 4,4 };
	int64_t dims_2[2] = { 4,4 };
	int64_t dims_3[2] = { 4,4 };
	struct tensor* tensorA = create_tensor(array1, 16, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_tensor(array2, 16, dims_2, 2, DATATYPE_FLOAT32, true);

	struct tensor* output = create_empty_tensor();

	struct node* newnode = create_concat_node(&AXIS, output, 2, tensorA, tensorB);
	error = inference_concat_node(newnode);
	if (error != OPS_SUCCESS) {
		printf("Ops fail\n");
	}
	print_tensor(output);
}

int16_t split_node_test() {
	printf("\n\n split node test test\n");
	int error = 0;
	int64_t num_outputs = 2;



	float array1[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };

	int64_t dims_1[2] = { 4,4 };

	struct tensor* tensorA = create_tensor(array1, 16, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_empty_tensor();
	struct tensor* tensorC = create_empty_tensor();


	struct node* newnode = create_split_node(NULL, &num_outputs, tensorA, NULL, tensorB, tensorC);
	error = inference_split_node(newnode);
	if (error != OPS_SUCCESS) {
		printf("Ops fail\n");
	}
	print_tensor(tensorB);
	print_tensor(tensorC);

	return error;
}

int16_t reshaped_node_test() {
	float array1[50];
	for (int i = 0; i < 50; i++) {
		array1[i] = i;
	}

	int64_t dims_1[2] = { 1,50 };

	struct tensor* input = create_tensor(array1, 50, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* output = create_empty_tensor();

	int64_t new_dims[] = { 5,5,2 }, dim[] = { 3 };
	int64_t allow = 0;


	struct tensor* new_shape = create_tensor(new_dims, 3, dim, 1, DATATYPE_INT64, true);
	struct node* new_node = create_reshape_node(&allow, input, new_shape, output);
	inference_reshape_node(new_node);

	print_tensor(output);


}


void pad_node_test() {
	printf("\n\pad node test\n");
	int error = 0;
	float array1[9] = { 1,2,3,4,5,6,7,8,9 }, array2[49] = { 0 };

	int64_t dims_1[2] = { 3,3 };

	struct tensor* tensorA = create_tensor(array1, 9, dims_1, 2, DATATYPE_FLOAT32, true);
	struct tensor* tensorB = create_empty_tensor();
	int64_t padding[4] = { 2,1,2,3 };
	int64_t pad_DIM[1] = { 4 };
	struct tensor* padtensor = create_tensor(padding, 4, pad_DIM, 1, DATATYPE_INT64, true);
	float val = 13.0f;
	int64_t val_dims[1] = { 1 };
	struct tensor* valtensor = create_tensor(&val, 1, val_dims, 1, DATATYPE_FLOAT32, true);
	struct node* new_node = create_pad_node(NULL, tensorA, padtensor, NULL, NULL, tensorB);
	error = inference_pad_node(new_node);
	if (error == OPS_SUCCESS) {
		printf("OPS SUCCESS\n");
	}
	else {
		if (error == OPS_UNDEFINED) {
			printf("OPS_UNDEFINED\n");
		}
		else if (error == OPS_INPUT_IS_NULL) {
			printf("OPS_INPUT_IS_NULL\n");
		}
		else if (error == OPS_NOT_BROADCASTABLE) {
			printf("OPS_NOT_BROADCASTABLE\n");
		}
		else if (error == OPS_DIMENSION_MISMATCH) {
			printf("OPS_DIMENSION_MISMATCH\n");
		}
		else if (error == OPS_ALLOCATION_FAIL) {
			printf("OPS_ALLOCATION_FAIL\n");
		}
		else if (error == OPS_TYPE_UNIMPLEMENTED) {
			printf("OPS_TYPE_UNIMPLEMENTED\n");
		}
		else if (error == OPS_TYPE_NOT_SUPPORTED) {
			printf("OPS_TYPE_NOT_SUPPORTED\n");
		}
		else {
			printf("Something is very wrong\n");
		}

		printf("OPS fail\n");
		return;
	}
	printf("result: \n");
	print_tensor(tensorA);
	print_tensor(tensorB);

}

int16_t conv_node_test() {
	float array1[15], array2[108];
	for (int i = 0; i < 15; i++) {
		array1[i] = i;
	}
	for (int i = 0; i < 108; i++) {
		array2[i] = i;

	}
	int64_t dims_1[4] = { 1,3, 5 };
	int64_t dims_2[4] = { 12,3,3 };
	char autopad[] = "VALID";
	struct tensor* x = create_tensor(array1, 15, dims_1, 3, DATATYPE_FLOAT32, true);
	struct tensor* w = create_tensor(array2, 108, dims_2, 3, DATATYPE_FLOAT32, true);
	struct tensor* output = create_empty_tensor();

	struct node* new_node = create_conv_node(&autopad, NULL, NULL, NULL, NULL, NULL, x, w, NULL, output);
	inference_conv_node(new_node);

	print_tensor(output);


}



void lstm_node_test() {
	float array1[128], array2[128], array3[257];
	/*str*/
}