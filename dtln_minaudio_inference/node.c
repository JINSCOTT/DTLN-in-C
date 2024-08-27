#include "node.h"

struct node* create_node() {
	struct node* NewNode = (struct node*)calloc(1, sizeof(struct node));
	if (NewNode == NULL) return NULL;
	NewNode->type = UNDEFINED;
	return NewNode;
}

struct node* create_add_node(struct tensor* A, struct tensor* B, struct tensor* C) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Add;
	pushback_list(&new_node->input, A);
	pushback_list(&new_node->input, B);
	pushback_list(&new_node->output, C);
	return new_node;
}

struct node* create_sub_node(struct tensor* A, struct tensor* B, struct tensor* C) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Sub;
	pushback_list(&new_node->input, A);
	pushback_list(&new_node->input, B);
	pushback_list(&new_node->output, C);
	return new_node;
}

struct node* create_mul_node(struct tensor* A, struct tensor* B, struct tensor* C) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Mul;
	pushback_list(&new_node->input, A);
	pushback_list(&new_node->input, B);
	pushback_list(&new_node->output, C);
	return new_node;
}

struct node* create_div_node(struct tensor* A, struct tensor* B, struct tensor* C) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Div;
	pushback_list(&new_node->input, A);
	pushback_list(&new_node->input, B);
	pushback_list(&new_node->output, C);
	return new_node;
}

struct node* create_tanh_node(struct tensor* input, struct tensor* output) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Tanh;
	pushback_list(&new_node->input, input);
	pushback_list(&new_node->output, output);
	return new_node;
}

struct node* create_sigmoid_node(struct tensor* X, struct tensor* Y) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Sigmoid;
	pushback_list(&new_node->input, X);
	pushback_list(&new_node->output, Y);
	return new_node;
}

struct node* create_sqrt_node(struct tensor* X, struct tensor* Y) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Sqrt;
	pushback_list(&new_node->input, X);
	pushback_list(&new_node->output, Y);
	return new_node;
}

struct node* create_squeeze_node(struct tensor* data, struct tensor* axes, struct tensor* squeezed) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Squeeze;
	pushback_list(&new_node->input, data);
	pushback_list(&new_node->input, axes);
	pushback_list(&new_node->output, squeezed);
	return new_node;
}

struct node* create_unsqueeze_node(struct tensor* data, struct tensor* axes, struct tensor* expanded) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Unsqueeze;
	pushback_list(&new_node->input, data);
	pushback_list(&new_node->input, axes);
	pushback_list(&new_node->output, expanded);
	return new_node;

}
struct node* create_transpose_node(int64_t* perm, struct tensor* data, struct tensor* transposed) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Transpose;
	pushback_list(&new_node->attribute, perm);
	pushback_list(&new_node->input, data);
	pushback_list(&new_node->output, transposed);
	return new_node;
}

struct node* create_matmul_node(struct tensor* a, struct tensor* b, struct tensor* c) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = MatMul;
	pushback_list(&new_node->input, a);
	pushback_list(&new_node->input, b);
	pushback_list(&new_node->output, c);
	return new_node;
}

struct node* create_slice_node(struct tensor* data, struct tensor* starts, struct tensor* ends, struct tensor* axes, struct tensor* steps, struct tensor* output) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Slice;
	pushback_list(&new_node->input, data);
	pushback_list(&new_node->input, starts);
	pushback_list(&new_node->input, ends);
	pushback_list(&new_node->input, axes);
	pushback_list(&new_node->input, steps);
	pushback_list(&new_node->output, output);
	return new_node;
}

struct node* create_gemm_node(float* alpha, float* beta, int64_t* transA, int64_t* transB, struct tensor* A, struct tensor* B, struct tensor* C, struct tensor* Y) {
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Gemm;
	pushback_list(&new_node->attribute, alpha);
	pushback_list(&new_node->attribute, beta);
	pushback_list(&new_node->attribute, transA);
	pushback_list(&new_node->attribute, transB);
	pushback_list(&new_node->input, A);
	pushback_list(&new_node->input, B);
	pushback_list(&new_node->input, C);
	pushback_list(&new_node->output, Y);
	return new_node;
}

struct node* create_concat_node(int64_t* axis, struct tensor* concat_result, int64_t num_input, ...) {
	va_list ptr;
	int64_t i = 0;
	struct list* input_list = NULL;
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Concat;
	input_list = calloc(1, sizeof(struct list));
	if (input_list == NULL) return NULL;	// fail to create list to hold inputs

	pushback_list(&new_node->attribute, axis);
	// Push inputs from variadic
	va_start(ptr, num_input);
	for (i = 0; i < num_input; i++) {
		struct tensor* temp = va_arg(ptr, struct tensor*);
		pushback_list(input_list, temp);
	}
	va_end(ptr);
	pushback_list(&new_node->input, input_list);
	pushback_list(&new_node->output, concat_result);
	return new_node;
}

struct node* create_split_node(int64_t* axis, int64_t* num_outputs, struct tensor* input, int64_t* split, ...) {
	;
	va_list ptr;
	int64_t i = 0;
	struct list* output_list = NULL;
	struct node* new_node = create_node();
	if (new_node == NULL) return NULL;
	new_node->type = Split;
	pushback_list(&new_node->attribute, axis);
	pushback_list(&new_node->attribute, num_outputs);
	pushback_list(&new_node->input, input);
	pushback_list(&new_node->input, split);

	output_list = calloc(1, sizeof(struct list));
	if (output_list == NULL) return NULL;	// Fail to create list for outputs
	va_start(ptr, split);
	for (i = 0; i < *num_outputs; i++) {
		struct tensor* temp = va_arg(ptr, int64_t*);
		pushback_list(output_list, temp);
	}
	va_end(ptr);
	pushback_list(&new_node->output, output_list);
	return new_node;
}
struct node* create_reshape_node(int64_t* allowzero, struct tensor* data, struct tensor* shape, struct tensor* reshaped) {
	struct node* new_node = create_node();
	new_node->type = Reshape;
	pushback_list(&new_node->attribute, allowzero);
	pushback_list(&new_node->input, data);
	pushback_list(&new_node->input, shape);
	pushback_list(&new_node->output, reshaped);
	return new_node;
}

struct node* create_pad_node(char* mode, struct tensor* data, struct tensor* pads, struct tensor* constant_value, struct tensor* axes, struct tensor* output) {
	struct node* new_node = create_node();
	new_node->type = Pad;
	pushback_list(&new_node->attribute, mode);
	pushback_list(&new_node->input, data);
	pushback_list(&new_node->input, pads);
	pushback_list(&new_node->input, constant_value);
	pushback_list(&new_node->input, axes);
	pushback_list(&new_node->output, output);
	return new_node;
}

struct node* create_conv_node(char* auto_pad, int64_t* dilations, int64_t* group, int64_t* kernel_shape, int64_t* pads, int64_t* strides, struct tensor* x, struct tensor* w, struct tensor* b, struct tensor* y) {
	struct node* new_node = create_node();
	new_node->type = Conv;
	pushback_list(&new_node->attribute, auto_pad);
	pushback_list(&new_node->attribute, dilations);
	pushback_list(&new_node->attribute, group);
	pushback_list(&new_node->attribute, kernel_shape);
	pushback_list(&new_node->attribute, pads);
	pushback_list(&new_node->attribute, strides);
	pushback_list(&new_node->input, x);
	pushback_list(&new_node->input, w);
	pushback_list(&new_node->input, b);
	pushback_list(&new_node->output, y);
	return new_node;
}

struct node* create_lstm_node(float* activation_alpha, float* activation_beta, struct list* activations, float* clip, char* direction, int64_t* hidden_size, int64_t* input_forget, int64_t* layout,
	struct tensor* x, struct tensor* w, struct tensor* r, struct tensor* b, int64_t* sequence_lens, struct tensor* initial_h, struct tensor* initial_c, struct tensor* p,
	struct tensor* y, struct tensor* y_h, struct tensor* y_c) {
	struct node* new_node = create_node();
	new_node->type = LSTM;
	pushback_list(&new_node->attribute, activation_alpha);
	pushback_list(&new_node->attribute, activation_beta);
	pushback_list(&new_node->attribute, activations);
	pushback_list(&new_node->attribute, clip);
	pushback_list(&new_node->attribute, hidden_size);
	pushback_list(&new_node->attribute, input_forget);
	pushback_list(&new_node->attribute, layout);
	pushback_list(&new_node->input, x);
	pushback_list(&new_node->input, w);
	pushback_list(&new_node->input, r);
	pushback_list(&new_node->input, b);
	pushback_list(&new_node->input, sequence_lens);
	pushback_list(&new_node->input, initial_h);
	pushback_list(&new_node->input, initial_c);
	pushback_list(&new_node->input, p);
	pushback_list(&new_node->output, y);
	pushback_list(&new_node->output, y_h);
	pushback_list(&new_node->output, y_c);
	return new_node;
}

struct node* create_reducemean_node(int64_t* keepdim, int64_t* noop_with_empty_axes, struct tensor* data, struct tensor* axes, struct tensor* reduced) {
	struct node* new_node = create_node();
	new_node->type = ReduceMean;
	pushback_list(&new_node->attribute, keepdim);
	pushback_list(&new_node->attribute, noop_with_empty_axes);
	pushback_list(&new_node->input, data);
	pushback_list(&new_node->input, axes);
	pushback_list(&new_node->output, reduced);
	return new_node;
}

struct node* create_constant_node(struct tensor* value, float* value_float, float* value_floats, int64_t* value_int, int64_t* value_ints, struct tensor* output) {
	struct node* new_node = create_node();
	new_node->type = Constant;
	pushback_list(&new_node->attribute, value);
	pushback_list(&new_node->attribute, value_float);
	pushback_list(&new_node->attribute, value_floats);
	pushback_list(&new_node->attribute, value_int);
	pushback_list(&new_node->attribute, value_ints);
	pushback_list(&new_node->output, output);
	return new_node;


}
void print_ops_error(int i) {
	if (i == OPS_UNDEFINED) {
		printf("OPS result undefined\n");
	}
	else if (i == OPS_SUCCESS) {
		printf("OPS result sucess\n");
	}
	else if (i == OPS_INPUT_IS_NULL) {
		printf("OPS result IS null\n");
	}
	else if (i == OPS_NOT_BROADCASTABLE) {
		printf("OPS result not broadcastable\n");
	}
	else if (i == OPS_DIMENSION_MISMATCH) {
		printf("OPS result mismatch\n");
	}
	else if (i == OPS_ALLOCATION_FAIL) {
		printf("OPS result allocation fail\n");
	}
	else if (i == OPS_TYPE_UNIMPLEMENTED) {
		printf("OPS result unimplemented\n");
	}
	else if (i == OPS_INVALID_ARGUMENT) {
		printf("OPS result invalid arguments\n");
	}
	else if (i == OPS_NO_OUTPUT_SHAPE) {
		printf("OPS result no output shape\n");
	}
	else {
		printf("OPS RESULT UNKNOWN\n");
	}
}
int inference_node(struct node* node) {
	int result = 0;
	//system("pause");
	switch (node->type)
	{
	case UNDEFINED: {
		// undefined node. Fail.
		printf("Undefined node\n");
		system("pause");
		return 0;
	}
	case Add: {
		//printf("Inference add\n");
		result = inference_add_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Mul: {
		//printf("Inference mul\n");
		result = inference_mul_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Div: {
		//printf("inference div\n");
		result = inference_div_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Sub: {
		//printf("inference sub\n");
		result = inference_sub_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Sigmoid: {
		//printf("inference sigmoid\n");
		result = inference_sigmoid_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Tanh: {
		//printf("inference tanh\n");
		result = inference_tanh_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Sqrt: {
		//printf("inference sqrt\n");
		result = inference_sqrt_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Squeeze: {
		//printf("inference squeeze\n");
		result = inference_squeeze_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Unsqueeze: {
		//printf("inference unsqueeze\n");
		result = inference_unsqueeze_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Transpose: {
		//printf("inference transpose\n");
		result = inference_transpose_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Slice: {
		//printf("Inference slice\n");
		result = inference_slice_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case MatMul: {
		//printf("inference matmul\n");
		result = inference_matmul_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Gemm: {
		//printf("inference gemm\n");
		result = inference_gemm_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Reshape: {
		//printf("inference reshape\n");
		result = inference_reshape_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Concat: {
		//printf("inference concat\n");
		result = inference_concat_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Split: {
		//printf("inference split\n");
		result = inference_split_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Pad: {
		//printf("inference pad\n");
		result = inference_pad_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Constant: {
		//printf("inference consant\n");
		result = inference_constant_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case Conv: {
		//printf("inference conv\n");
		result = inference_conv_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case ReduceMean: {
		//printf("inference reducemean\n");
		result = inference_reducemean_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	case LSTM: {
		//printf("inference lstm\n");
		result = inference_lstm_node(node);
		//print_ops_error(result);
		if (result != OPS_SUCCESS)return 0;
		return 1;
	}
	default:
		printf("unknown\n");
		system("pause");
		break;
	}
	return 0;
}


int inference_add_node(struct node* n) {
	//printf("Add inference\n");
	int error = 0;
	struct tensor* A = NULL, * B = NULL, * C = NULL;
	A = (struct tensor*)get_list(&n->input, 0);
	B = (struct tensor*)get_list(&n->input, 1);
	C = (struct tensor*)get_list(&n->output, 0);
	if (A == NULL || B == NULL || C == NULL) return OPS_INPUT_IS_NULL;
	// Calculate C shape
	if (C->is_size_unknown == true) {
		error = set_broadcast_shape(A, B, C);
		if (error != OPS_SUCCESS) return error;
	}
	//calculate_result
	error = broadcast_function(A, B, C, Add);
#ifdef DEBUG
	printf("Add node result\n");
	printf("print tensor A:\n");
	print_tensor(A);
	printf("print tensor B:\n");
	print_tensor(B);
	printf("print tensor C:\n");
	print_tensor(C);
	printf("\n\n\n");
#endif // DEBUG
	return error;
}

int inference_sub_node(struct node* n) {
	int error = 0;
	struct tensor* A = NULL, * B = NULL, * C = NULL;
	A = (struct tensor*)get_list(&n->input, 0);
	B = (struct tensor*)get_list(&n->input, 1);
	C = (struct tensor*)get_list(&n->output, 0);
	if (A == NULL || B == NULL || C == NULL)return OPS_INPUT_IS_NULL;
	// Calculate C shape is it doesn't have a shape
	if (C->is_size_unknown == true) {
		error = set_broadcast_shape(A, B, C);
		if (error != OPS_SUCCESS) return error;
	}
	error = broadcast_function(A, B, C, Sub);
#ifdef DEBUG
	printf("Sub node result\n");
	printf("print tensor A:\n");
	print_tensor(A);
	printf("print tensor B:\n");
	print_tensor(B);
	printf("print tensor C:\n");
	print_tensor(C);
	printf("\n\n\n");
#endif // DEBUG
	return error;
}

int inference_mul_node(struct node* n) {
	int error = 0;
	struct tensor* A = NULL, * B = NULL, * C = NULL;
	A = (struct tensor*)get_list(&n->input, 0);
	B = (struct tensor*)get_list(&n->input, 1);
	C = (struct tensor*)get_list(&n->output, 0);
	if (A == NULL || B == NULL || C == NULL) return OPS_INPUT_IS_NULL;
	// Calculate C shape is it doesn't have a shape
	if (C->is_size_unknown == true) {
		error = set_broadcast_shape(A, B, C);
		if (error != OPS_SUCCESS) return error;
	}
	error = broadcast_function(A, B, C, Mul);
#ifdef DEBUG
	printf("Mul node result\n");
	printf("print tensor A:\n");
	print_tensor(A);
	printf("print tensor B:\n");
	print_tensor(B);
	printf("print tensor C:\n");
	print_tensor(C);
	printf("\n\n\n");
#endif // DEBUG
	return error;
}


int inference_div_node(struct node* n) {
	int error = 0;
	struct tensor* A = NULL, * B = NULL, * C = NULL;
	A = (struct tensor*)get_list(&n->input, 0);
	B = (struct tensor*)get_list(&n->input, 1);
	C = (struct tensor*)get_list(&n->output, 0);
	if (A == NULL || B == NULL || C == NULL) return OPS_INPUT_IS_NULL;
	// Calculate C shape is it doesn't have a shape
	if (C->is_size_unknown == true) {
		error = set_broadcast_shape(A, B, C);
		if (error != OPS_SUCCESS) return error;
	}
	error = broadcast_function(A, B, C, Div);
#ifdef DEBUG
	printf("Div node result\n");
	printf("print tensor A:\n");
	print_tensor(A);
	printf("print tensor B:\n");
	print_tensor(B);
	printf("print tensor C:\n");
	print_tensor(C);
	printf("\n\n\n");
#endif // DEBUG
	return error;
}

int inference_tanh_node(struct node* n) {
	int error = 0;
	struct tensor* X = NULL, * Y = NULL;
	X = (struct tensor*)get_list(&n->input, 0);
	Y = (struct tensor*)get_list(&n->output, 0);
	if (X == NULL || Y == NULL)	return OPS_INPUT_IS_NULL;
	// Calculate Y shape is it doesn't have a shape
	if (Y->is_size_unknown == true) {
		error = resize_tensor(Y, X->dimension, X->dimension_size, X->type);
		if (error < 1) {
			return OPS_ALLOCATION_FAIL;
		}
		Y->is_size_unknown = 0;
	}
	error = tanh_function(X, Y);
#ifdef DEBUG
	printf("Tanh node result\n");
	printf("print tensor X:\n");
	print_tensor(X);
	printf("print tensor Y:\n");
	print_tensor(Y);
	printf("\n\n\n");
#endif // DEBUG
	return error;
}
int inference_sigmoid_node(struct node* n) {
	int error = 0;
	struct tensor* X = NULL, * Y = NULL;
	X = (struct tensor*)get_list(&n->input, 0);
	Y = (struct tensor*)get_list(&n->output, 0);
	if (X == NULL || Y == NULL)	return OPS_INPUT_IS_NULL;
	// Calculate Y shape is it doesn't have a shape
	if (Y->is_size_unknown == true) {
		error = resize_tensor(Y, X->dimension, X->dimension_size, X->type);
		if (error < 1) {
			return OPS_ALLOCATION_FAIL;
		}
		Y->is_size_unknown = 0;
	}
	error = sigmoid_function(X, Y);
#ifdef DEBUG
	printf("Sigmoid node result\n");
	printf("print tensor X:\n");
	print_tensor(X);
	printf("print tensor Y:\n");
	print_tensor(Y);
	printf("\n\n\n");
#endif // DEBUG
	return error;
}
int inference_sqrt_node(struct node* n) {
	int error = 0;
	struct tensor* X = NULL, * Y = NULL;
	X = (struct tensor*)get_list(&n->input, 0);
	Y = (struct tensor*)get_list(&n->output, 0);
	if (X == NULL || Y == NULL) return OPS_INPUT_IS_NULL;
	// Calculate Y shape is it doesn't have a shape
	if (Y->is_size_unknown == true) {
		error = resize_tensor(Y, X->dimension, X->dimension_size, X->type);
		if (error < 1) {
			return OPS_ALLOCATION_FAIL;
		}
		Y->is_size_unknown = 0;
	}
	error = sqrt_function(X, Y);
#ifdef DEBUG
	printf("Sqrt node result\n");
	printf("print tensor X:\n");
	print_tensor(X);
	printf("print tensor Y:\n");
	print_tensor(Y);
	printf("\n\n\n");
#endif // DEBUG
	return error;
}


int inference_squeeze_node(struct node* n) {
	int error = OPS_SUCCESS;
	struct tensor* data = NULL, * squeezed = NULL, * axes = NULL;
	int64_t* new_axes = NULL, new_axes_size = 0;
	data = (struct tensor*)get_list(&n->input, 0);
	axes = (struct tensor*)get_list(&n->input, 1);
	squeezed = (struct tensor*)get_list(&n->output, 0);
	if (data == NULL || squeezed == NULL) return OPS_INPUT_IS_NULL;
	// Calculate Y shape
	if (squeezed->is_size_unknown) {
		error = set_squeeze_shape(data, axes, squeezed);
		if(error != OPS_SUCCESS) goto finally;
	}
	memcpy_s(squeezed->data, squeezed->data_size * squeezed->item_size, data->data, data->data_size * data->item_size);
#ifdef DEBUG
	printf("Squeeze node result\n");
	printf("print tensor data:\n");
	print_tensor(data);
	printf("print tensor axes:\n");
	print_tensor(axes);
	printf("print tensor squeezed:\n");
	print_tensor(squeezed);
	printf("\n\n\n");
#endif
finally:
	return error;
}

int inference_unsqueeze_node(struct node* n) {
	int error = OPS_SUCCESS;
	struct tensor* data = NULL, * expanded = NULL, * axes = NULL;
	data = (struct tensor*)get_list(&n->input, 0);
	axes = (struct tensor*)get_list(&n->input, 1);
	expanded = (struct tensor*)get_list(&n->output, 0);
	if (data == NULL || axes == NULL|| expanded == NULL)return OPS_INPUT_IS_NULL;
	// Calculate Y shape
	if (expanded->is_size_unknown) {
		error = set_unsqueeze_shape(data, axes, expanded);
		if (error != OPS_SUCCESS) goto finally;
	}
	memcpy_s(expanded->data, expanded->data_size * expanded->item_size, data->data, data->data_size * data->item_size);
#ifdef DEBUG
	printf("Unsqueeze node result\n");
	printf("print tensor data:\n");
	print_tensor(data);
	printf("print tensor axes:\n");
	print_tensor(axes);
	printf("print tensor squeezed:\n");
	print_tensor(expanded);
	printf("\n\n\n");
#endif // DEBUG
finally:
	return error;
}

int inference_transpose_node(struct node* n) {
	int error = 0;
	int64_t* perm = NULL, i = 0;
	struct tensor* data = NULL, * transposed = NULL;
	perm = (int64_t*)get_list(&n->attribute, 0);
	data = (struct tensor*)get_list(&n->input, 0);
	transposed = (struct tensor*)get_list(&n->output, 0);
	if (data == NULL || transposed == NULL) return OPS_INPUT_IS_NULL;
	if (perm == NULL) {
		perm = malloc(data->dimension_size * sizeof(int64_t));
		if (perm == NULL) return OPS_ALLOCATION_FAIL;
		for (i = 0; i < data->dimension_size; i++) {
			perm[data->dimension_size - 1 - i] = i;
		}
	}
	if (transposed->is_size_unknown) {
		error = set_transpose_shape(data, perm, transposed);
		if (error != OPS_SUCCESS) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
	}
	error = transpose_function(data, transposed, perm);
#ifdef DEBUG
	printf("tRANSPOSE node result\n");
	printf("permutation: ");
	print_int64_t(perm, data->dimension_size);
	printf("\nprint tensor data:\n");
	print_tensor(data);
	printf("print tensor squeezed:\n");
	print_tensor(transposed);
	printf("\n\n\n");
#endif // DEBUG
cleanup:
	return error;
}

int inference_matmul_node(struct node* n) {
	int error = 0;
	struct tensor* a = NULL, * b = NULL, * c = NULL;

	a = (struct tensor*)get_list(&n->input, 0);
	b = (struct tensor*)get_list(&n->input, 1);
	c = (struct tensor*)get_list(&n->output, 0);

	if (a == NULL || b == NULL || c == NULL) {
		return OPS_INPUT_IS_NULL;
	}

	if (c->is_size_unknown) {
		error = set_matmul_shape(a, b, c);
		if (error != OPS_SUCCESS) {
			return OPS_DIMENSION_MISMATCH;
		}
	}
	error = matmul_function(a, b, c);
	/*printf("matmul node result\n");
	printf("\nprint tensor A:\n");
	print_tensor(a);
	printf("\nprint tensor B:\n");
	print_tensor(b);
	printf("\nprint tensor C:\n");
	print_tensor(c);
	printf("\n\n\n");*/
	return error;
}


int inference_slice_node(struct node* n) {
	//printf("slice node inference\n");
	int error = 0;
	int64_t i = 0, push_value = 0, * axes_arr = NULL, * steps_arr = NULL, * starts_arr = NULL, * ends_arr = NULL; ;
	struct tensor* data = NULL, * output = NULL, * starts = NULL, * ends = NULL, * axes = NULL, * steps = NULL;;


	data = (struct tensor*)get_list(&n->input, 0);
	starts = (struct tensor*)get_list(&n->input, 1);
	ends = (struct tensor*)get_list(&n->input, 2);
	axes = (struct tensor*)get_list(&n->input, 3);
	steps = (struct tensor*)get_list(&n->input, 4);
	output = (struct tensor*)get_list(&n->output, 0);
	// Set defualt value for missing axes and steps
	if (axes == NULL) {
		axes_arr = malloc(data->dimension_size * sizeof(int64_t));
		if (axes_arr == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		for (i = 0; i < data->dimension_size; i++) {
			axes_arr[i] = i;
		}
	}
	else {
		axes_arr = axes->data;
	}
	if (steps == NULL) {
		steps_arr = malloc(data->dimension_size * sizeof(int64_t));
		if (steps_arr == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		for (i = 0; i < data->dimension_size; i++) {
			steps_arr[i] = 1;
		}
	}
	else {
		steps_arr = steps->data;
	}
	if (output->is_size_unknown) {
		error = set_slice_shape(data, starts->data, ends->data, axes_arr, steps_arr, output);
		if (error != OPS_SUCCESS) {
			return OPS_DIMENSION_MISMATCH;
			replace_list(&n->input, NULL, 3);
		}
		output->is_size_unknown = false;
	}
	error = slice_function(data, output, starts->data, ends->data, axes_arr, steps_arr);
	if (error = OPS_SUCCESS) print_tensor(output);
cleanup:
	if (axes == NULL)safe_free(&axes_arr);
	if (steps == NULL) safe_free(&steps_arr);
	//printf("Slice node result\n");
	//printf("\nprint tensor data:\n");
	//print_tensor(data);
	//printf("\nprint tensor output:\n");
	//print_tensor(output);
	//printf("\n\n\n");
	return error;
}

int inference_gemm_node(struct node* n) {
	//printf("gemm inference\n");
	int error = 0;
	int64_t* transA = NULL, * transB = NULL, insert_int64 = 0;
	float* alpha = NULL, * beta = NULL, insert_float = 0.f;
	struct tensor* A = NULL, * B = NULL, * C = NULL, * Y = NULL;
	alpha = (float*)get_list(&n->attribute, 0);
	beta = (float*)get_list(&n->attribute, 1);
	transA = (int64_t*)get_list(&n->attribute, 2);
	transB = (int64_t*)get_list(&n->attribute, 3);
	A = (struct tensor*)get_list(&n->input, 0);
	B = (struct tensor*)get_list(&n->input, 1);
	C = (struct tensor*)get_list(&n->input, 2);
	Y = (struct tensor*)get_list(&n->output, 0);
	if (alpha == NULL) {
		alpha = malloc(sizeof(float));
		if (alpha == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		insert_float = 1.0f;
		memcpy_s(alpha, sizeof(float), &insert_float, sizeof(float));
		replace_list(&n->attribute, alpha, 0);
	}
	if (beta == NULL) {
		beta = malloc(sizeof(float));
		if (beta == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		insert_float = 1.0f;
		memcpy_s(beta, sizeof(float), &insert_float, sizeof(float));
		replace_list(&n->attribute, beta, 1);
	}
	if (transA == NULL) {
		transA = malloc(sizeof(int64_t));
		if (transA == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		insert_int64 = 0;
		memcpy_s(transA, sizeof(int64_t), &insert_int64, sizeof(int64_t));
		replace_list(&n->attribute, transA, 2);
	}
	if (transB == NULL) {
		transB = malloc(sizeof(int64_t));
		if (transB == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		insert_int64 = 0;
		memcpy_s(transB, sizeof(int64_t), &insert_int64, sizeof(int64_t));
		replace_list(&n->attribute, transB, 3);
	}
	if (Y->is_size_unknown) {
		error = set_gemm_shape(A, B, transA, transB, Y);
		if (error != OPS_SUCCESS) {
			error = OPS_DIMENSION_MISMATCH;
			goto cleanup;
		}
	}

	error = gemm_function(A, B, C, Y, *alpha, *beta, *transA, *transB);

	/*printf("gemm node result\n");
	printf("\nprint tensor A:\n");
	print_tensor(A);
	printf("\nprint tensor B:\n");
	print_tensor(B);
	printf("\nprint tensor C:\n");
	print_tensor(C);
	printf("\nprint tensor Y:\n");
	print_tensor(Y);
	printf("\n\n\n");*/


cleanup:

	if (get_list(&n->attribute, 0) == 0)safe_free(&alpha);
	if (get_list(&n->attribute, 1) == 0)safe_free(&beta);
	if (get_list(&n->attribute, 2) == 0)safe_free(&transA);
	if (get_list(&n->attribute, 3) == 0)safe_free(&transB);

	return error;
}



int inference_concat_node(struct node* n) {
	//printf("concat inference\n");
	int error = 0;
	int64_t* axis = NULL;
	struct  list* inputs = NULL;
	struct tensor* concat_result = NULL;
	axis = (int64_t*)get_list(&n->attribute, 0);
	inputs = (struct list*)get_list(&n->input, 0);
	concat_result = (struct tensor*)get_list(&n->output, 0);
	if (concat_result->is_size_unknown) {
		error = set_concat_shape(axis, inputs, concat_result);
		if (error != OPS_SUCCESS) {
			error = OPS_DIMENSION_MISMATCH;
			goto finally;
		}
		concat_result->is_size_unknown = false;
	}
	error = concat_function(axis, inputs, concat_result);
	finally:
	//printf("Concat node result\n");
	//printf("\nprint inputs:\n");
	//for (int i = 0; i < inputs->size; i++) {
	//	printf("\nprint input tensor %d:\n",i);
	//	print_tensor(get_list(inputs, i ));
	//}
	//printf("\nprint tensor cocnat result\n");
	//print_tensor(concat_result);
	//printf("\n\n\n");
	return error;
}

int inference_split_node(struct node* n) {
	int error = 0;
	int64_t* axis = NULL, * num_outputs = NULL, * split = NULL, insert_int64 = 0;
	struct tensor* input = NULL;
	struct list* outputs = NULL;
	axis = (int64_t*)get_list(&n->attribute, 0);
	num_outputs = (int64_t*)get_list(&n->attribute, 1);
	input = (struct tensor*)get_list(&n->input, 0);
	split = (int64_t*)get_list(&n->input, 1);
	outputs = (struct list*)get_list(&n->output, 0);

	if (axis == NULL) {
		axis = malloc(sizeof(int64_t));
		if (axis == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		insert_int64 = 0;
		memcpy_s(axis, sizeof(int64_t), &insert_int64, sizeof(int64_t));
		replace_list(&n->attribute, axis, 0);
	}
	if (((struct tensor*)outputs->first->data)->is_size_unknown) {
		error = set_split_shape(*axis, *num_outputs, input, split, outputs);
		if (error != OPS_SUCCESS) {
			error = OPS_DIMENSION_MISMATCH;
			goto cleanup;
		}
	}
	error = split_function(*axis, *num_outputs, input, split, outputs);

cleanup:
	if (get_list(&n->attribute, 0) == NULL) safe_free(&axis);

	/*printf("Concat node result\n");
	printf("\nprint tensor input\n");
	print_tensor(input);
	printf("\nprint inputs:\n");
	for (int i = 0; i < outputs->size; i++) {
		printf("\nprint output tensor %d:\n", i);
		print_tensor(get_list(outputs, i));
	}
	printf("\n\n\n");*/


	return error;
}

int inference_reshape_node(struct node* n) {
	int error = 0;
	int64_t* allowzero = NULL;
	struct tensor* input = NULL, * reshaped = NULL, * shape = NULL;
	allowzero = (int64_t*)get_list(&n->attribute, 0);
	input = (struct tensor*)get_list(&n->input, 0);
	shape = (struct tensor*)get_list(&n->input, 1);
	reshaped = (struct tensor*)get_list(&n->output, 0);
	if (reshaped->is_size_unknown) {
		error = set_reshaped_shape(input, shape, reshaped);
		if (error != OPS_SUCCESS) {
			error = OPS_DIMENSION_MISMATCH;
			goto cleanup;
		}
	}
	memcpy(reshaped->data, input->data, reshaped->data_size * reshaped->item_size);
	error = OPS_SUCCESS;
cleanup:
	/*printf("gemm node result\n");
	printf("new shape: ");
	print_int64_t(shape, reshaped->dimension_size);
	printf("\nprint tensor input:\n");
	print_tensor(input);
	printf("\nprint tensor reshaped:\n");
	print_tensor(reshaped);
	printf("\n\n\n");*/
	return error;
}

int inference_pad_node(struct node* n) {
	int error = 0;
	struct tensor* data = NULL, * output = NULL, * value = NULL, * axes = NULL, * pads = NULL;

	char* mode = NULL, default_mode[] = "constant";
	int64_t i = 0, * temp_dim = NULL, * temp_data = NULL;


	mode = (char*)get_list(&n->attribute, 0);
	data = (struct tensor*)get_list(&n->input, 0);
	pads = (struct tensor*)get_list(&n->input, 1);
	value = (struct tensor*)get_list(&n->input, 2);
	axes = (struct tensor*)get_list(&n->input, 3);
	output = (struct tensor*)get_list(&n->output, 0);


	if (mode == NULL) {
		mode = malloc(sizeof(default_mode));
		if (mode == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		memcpy_s(mode, sizeof(default_mode), default_mode, sizeof(default_mode));
		replace_list(&n->attribute, mode, 0);
	}
	if (axes == NULL) {	// reorder with axes
		temp_dim = malloc(1 * sizeof(int64_t));
		if (temp_dim == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		temp_data = malloc(data->dimension_size * sizeof(int64_t));
		if (temp_data == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		*temp_dim = data->dimension_size;
		for (i = 0; i < data->dimension_size; i++) {
			temp_data[i] = i;
		}
		axes = create_tensor(temp_data, data->dimension_size, temp_dim, 1, DATATYPE_INT64, false);
		if (axes == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		// change to null to not trigger again
		replace_list(&n->input, NULL, 3);
	}
	if (output->is_size_unknown) {
		error = set_pad_shape(data, pads, axes, output);
		if (error != OPS_SUCCESS) {
			error = OPS_DIMENSION_MISMATCH;
			goto cleanup;
		}
		output->is_size_unknown = 0;
	}
	error = pad_function(mode, data, pads, value, axes, output);
cleanup:
	if (get_list(&n->attribute, 0) == NULL) safe_free(&mode);
	safe_free(&temp_dim);
	safe_free(&temp_data);


	//printf("gemm node result\n");
	//printf("\nprint tensor data:\n");
	//print_tensor(data);
	//printf("\nprint tensor output:\n");
	//print_tensor(output);
	//printf("\n\n\n");

	return error;
}

int inference_conv_node(struct node* n) {
	//printf("inference conv\n");
	int error = 0;
	char* autopad = NULL, default_autopad[] = "NOTSET";
	int64_t* dilations = NULL, * group = NULL, * kernel_shape = NULL, * pads = NULL, * strides = NULL, i = 0, total_pad = 0;
	struct tensor* x = NULL, * w = NULL, * b = NULL, * y = NULL;
	autopad = (char*)get_list(&n->attribute, 0);
	dilations = (int64_t*)get_list(&n->attribute, 1);
	group = (int64_t*)get_list(&n->attribute, 2);
	kernel_shape = (int64_t*)get_list(&n->attribute, 3);
	pads = (int64_t*)get_list(&n->attribute, 4);
	strides = (int64_t*)get_list(&n->attribute, 5);
	x = (struct tensor*)get_list(&n->input, 0);
	w = (struct tensor*)get_list(&n->input, 1);
	b = (struct tensor*)get_list(&n->input, 2);
	y = (struct tensor*)get_list(&n->output, 0);
	if (x == NULL || w == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	// Set defualts
	if (autopad == NULL) {
		autopad = malloc(sizeof(default_autopad));
		if (autopad == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		memcpy_s(autopad, sizeof(default_autopad), default_autopad, sizeof(default_autopad));
		replace_list(&n->attribute, autopad, 0);
	}
	if (dilations == NULL) {
		dilations = malloc((x->dimension_size - 2) * sizeof(int64_t));
		if (dilations == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		for (i = 0; i < x->dimension_size - 2; i++) {
			dilations[i] = 1;
		}
		replace_list(&n->attribute, dilations, 1);
	}

	if (group == NULL) {
		group = malloc(sizeof(int64_t));
		*group = 1;
		replace_list(&n->attribute, group, 2);
	}
	if (kernel_shape == NULL) {
		kernel_shape = malloc((w->dimension_size - 2) * sizeof(int64_t));
		for (i = 2; i < w->dimension_size; i++) {
			kernel_shape[i - 2] = w->dimension[i];
		}
		replace_list(&n->attribute, kernel_shape, 3);
	}
	if (strides == NULL) {
		strides = malloc((w->dimension_size - 2) * sizeof(int64_t));
		if (strides == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		for (i = 0; i < w->dimension_size - 2; i++) {
			strides[i] = 1;
		}
		replace_list(&n->attribute, strides, 5);
	}
	if (pads == NULL) {
		pads = malloc(2 * (w->dimension_size - 2) * sizeof(int64_t));
		if (pads == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		if (strcmp(autopad, "VALID") == 0) {
			;
			for (i = 0; i < (w->dimension_size - 2) * 2; i++) {
				pads[i] = 0;
			}
		}
		else if (strcmp(autopad, "SAME_UPPER") == 0) {
			for (i = 0; i < (w->dimension_size - 2) * 2; i++) {
				total_pad = strides[i] * (w->dimension[i + 2] - 1) - w->dimension[i + 2] + kernel_shape[i];
				pads[i] = total_pad / 2;
				pads[i + w->dimension_size - 2] = total_pad / 2;
				if (total_pad % 2 != 0)pads[i]++;
			}
		}
		else if (strcmp(autopad, "SAME_LOWER") == 0) {
			for (i = 0; i < (w->dimension_size - 2) * 2; i++) {
				total_pad = strides[i] * (w->dimension[i + 2] - 1) - w->dimension[i + 2] + kernel_shape[i];
				pads[i] = total_pad / 2;
				pads[i + w->dimension_size - 2] = total_pad / 2;
				if (total_pad % 2 != 0)pads[i + w->dimension_size - 2]++;
			}

		}
		else if (autopad == NULL || strcmp(autopad, "NOT_SET")) {	// re
			for (i = 0; i < 2 * (w->dimension_size - 2); i++)
			{
				pads[i] = 0;
			}

		}
		else {
			error = OPS_INVALID_ARGUMENT;
			goto cleanup;
		}
		replace_list(&n->attribute, pads, 4);
	}
	// Calculate size
	if (y->is_size_unknown) {
		error = set_conv_shape(x, w, y, pads, strides);
		if (error != OPS_SUCCESS) {
			goto cleanup;
		}
		y->is_size_unknown = 0;
	}
	error = conv_function(x, w, b, y, dilations, *group, kernel_shape, pads, strides);
cleanup:
	if (get_list(&n->attribute, 0) == NULL) safe_free(&autopad);
	if (get_list(&n->attribute, 1) == NULL) safe_free(&dilations);
	if (get_list(&n->attribute, 2) == NULL) safe_free(&group);
	if (get_list(&n->attribute, 3) == NULL) safe_free(&kernel_shape);
	if (get_list(&n->attribute, 4) == NULL) safe_free(&pads);
	if (get_list(&n->attribute, 5) == NULL) safe_free(&strides);
	return error;
}

int inference_lstm_node(struct node* n) {
	int error = 0;
	float* activation_alpha = NULL, * activation_beta = NULL, * clip = NULL;
	struct list* activations = NULL;
	char* direction = NULL, default_direction[8] = "forward";
	int64_t* hidden_size = NULL, * input_forget = NULL, * layout = NULL, i = 0, j = 0, num_direction;
	struct tensor* x = NULL, * w = NULL, * r = NULL, * b = NULL, * sequence_length = NULL, * initial_h = NULL, * initial_c = NULL, * p = NULL, * y = NULL, * y_h = NULL, * y_c = NULL;
	struct dynamic_array* temp_dim = NULL;
	activation_alpha = (float*)get_list(&n->attribute, 0);
	activation_alpha = (float*)get_list(&n->attribute, 1);
	activations = (struct list*)get_list(&n->attribute, 2);
	clip = (float*)get_list(&n->attribute, 3);
	direction = (char*)get_list(&n->attribute, 4);
	hidden_size = (int64_t*)get_list(&n->attribute, 5);
	input_forget = (int64_t*)get_list(&n->attribute, 6);
	layout = (int64_t*)get_list(&n->attribute, 7);
	x = (struct tensor*)get_list(&n->input, 0);
	w = (struct tensor*)get_list(&n->input, 1);
	r = (struct tensor*)get_list(&n->input, 2);
	b = (struct tensor*)get_list(&n->input, 3);
	sequence_length = (struct tensor*)get_list(&n->input, 4);
	initial_h = (struct tensor*)get_list(&n->input, 5);
	initial_c = (struct tensor*)get_list(&n->input, 6);
	p = (struct tensor*)get_list(&n->input, 7);
	y = (struct tensor*)get_list(&n->output, 0);
	y_h = (struct tensor*)get_list(&n->output, 1);
	y_c = (struct tensor*)get_list(&n->output, 2);

	if (direction == NULL) {
		direction = malloc(sizeof(default_direction));
		if (direction == NULL)goto cleanup;
		memcpy(direction, default_direction, sizeof(default_direction));
		replace_list(&n->attribute, direction, 5);
	}
	num_direction = 1;
	if (strcmp(direction, "bidirectional") == 0) {
		num_direction = 2;
	}
	if (initial_h == NULL) {
		temp_dim = create_darray(sizeof(int64_t));
		if (temp_dim == NULL) goto cleanup;
		pushback_darray(temp_dim, &num_direction);
		pushback_darray(temp_dim, &x->dimension[1]);
		pushback_darray(temp_dim, hidden_size);
		j = 1;
		for (i = 0; i < temp_dim->size; i++) {
			j *= *(int64_t*)get_darray(temp_dim, i);
		}
		initial_h = create_tensor(NULL, j, temp_dim->data, temp_dim->size, x->type, false);
		if (initial_h == NULL) goto cleanup;
		replace_list(&n->input, initial_h, 5);
		release_darray(&temp_dim);
	}
	if (initial_c == NULL) {
		temp_dim = create_darray(sizeof(int64_t));
		if (temp_dim == NULL) goto cleanup;

		pushback_darray(temp_dim, &num_direction);
		pushback_darray(temp_dim, x->dimension[1]);
		pushback_darray(temp_dim, hidden_size);
		j = 1;
		for (i = 0; i < temp_dim->size; i++) {
			j *= *(int64_t*)get_darray(temp_dim, i);
		}
		initial_c = create_tensor(NULL, j, temp_dim->data, temp_dim->size, x->type, false);
		if (initial_c == NULL) goto cleanup;
		replace_list(&n->input, initial_c, 6);
		release_darray(&temp_dim);
	}
	if (p == NULL) {
		temp_dim = create_darray(sizeof(int64_t));
		if (temp_dim == NULL) goto cleanup;

		pushback_darray(temp_dim, &num_direction);
		j = *hidden_size;
		j *= 3;
		pushback_darray(temp_dim, &j);
		j = 1;
		for (i = 0; i < temp_dim->size; i++) {
			j *= *(int64_t*)get_darray(temp_dim, i);
		}
		p = create_tensor(NULL, j, temp_dim->data, temp_dim->size, x->type, false);
		if (p == NULL) goto cleanup;
		replace_list(&n->input, p, 7);
	}
	if (y == NULL) {
		y = create_empty_tensor();
		if (y == NULL) goto cleanup;
		replace_list(&n->output, y, 0);
	}
	if (y_h == NULL) {
		y_h = create_empty_tensor();
		if (y_h == NULL) goto cleanup;
		replace_list(&n->output, y_h, 1);
	}
	if (y_c == NULL) {
		y_c = create_empty_tensor();
		if (y_c == NULL) goto cleanup;
		replace_list(&n->output, y_c, 2);
	}
	if (y_c->is_size_unknown || y->is_size_unknown || y_c->is_size_unknown) {

		error = set_lstm_shape(x, num_direction, *hidden_size, y, y_h, y_c);
		if (error != OPS_SUCCESS) goto cleanup;
	}
	error = lstm_function(activation_alpha, activation_alpha, activations, *clip, direction, *hidden_size, *input_forget, *layout
		, x, w, r, b, sequence_length, initial_h, initial_c, p, y, y_h, y_c);

cleanup:
	if (get_list(&n->attribute, 4) == NULL) safe_free(&direction);
	if (get_list(&n->input, 5) == NULL) safe_free(&initial_h);
	if (get_list(&n->input, 6) == NULL) safe_free(&initial_c);
	if (get_list(&n->input, 7) == NULL) safe_free(&p);
	if (get_list(&n->output, 0) == NULL) safe_free(&y);
	if (get_list(&n->output, 1) == NULL) safe_free(&y_h);
	if (get_list(&n->output, 2) == NULL) safe_free(&y_c);
	release_darray(&temp_dim);
}

int inference_reducemean_node(struct node* n) {
	int error = 0;

	int64_t* keepdims = NULL, * noop_with_empty_axes = NULL, * temp = NULL, i = 0;
	struct tensor* data = NULL, * reduced = NULL, * axes = NULL;

	keepdims = (int64_t*)get_list(&n->attribute, 0);
	noop_with_empty_axes = (int64_t*)get_list(&n->attribute, 1);
	data = (struct tensor*)get_list(&n->input, 0);
	axes = (struct tensor*)get_list(&n->input, 1);
	reduced = (struct tensor*)get_list(&n->output, 0);
	if (keepdims == NULL) {
		temp = malloc(sizeof(int64_t));
		if (temp == NULL)goto cleanup;
		*temp = 1;
		keepdims = temp;
		replace_list(&n->attribute, temp, 0);
	}
	if (noop_with_empty_axes == NULL) {
		temp = malloc(sizeof(int64_t));
		if (temp == NULL)goto cleanup;
		*temp = 0;
		noop_with_empty_axes = temp;
		replace_list(&n->attribute, temp, 1);
	}
	if (reduced->is_size_unknown) {
		error = set_reducemean_shape(axes, noop_with_empty_axes, *keepdims, data, reduced);
		if (error != OPS_SUCCESS)goto cleanup;
	}
	error = reducemean_function(*keepdims, *noop_with_empty_axes, data, axes, reduced);
cleanup:
	//printf("reduce_mean result\n");
	//printf("\nprint tensor data:\n");
	//print_tensor(data);
	//printf("\nprint tensor reduced:\n");
	//print_tensor(reduced);
	//printf("\n\n\n");
	//system("pause");
	if (get_list(&n->attribute, 0) == NULL) safe_free(&keepdims);
	if (get_list(&n->attribute, 1) == NULL) safe_free(&noop_with_empty_axes);
	safe_free(&temp);
	return error;
}

int inference_constant_node(struct node* n) {
	// constant node should have their output dimension specified to simplify input 
	int error = 0;
	struct tensor* value = NULL, * output = NULL;
	float* value_float = NULL;
	float* value_floats = NULL;
	int64_t* value_int = NULL;
	int64_t* value_ints = NULL;
	value = (struct tensor*)get_list(&n->attribute, 0);
	value_float = (float*)get_list(&n->attribute, 1);
	value_floats = (float*)get_list(&n->attribute, 2);
	value_int = (int64_t*)get_list(&n->attribute, 3);
	value_ints = (int64_t*)get_list(&n->attribute, 4);
	output = (struct tensor*)get_list(&n->output, 0);
	if (output == NULL) return OPS_INPUT_IS_NULL;
	if (output->is_size_unknown) {
		printf("output size unknown\n");
		return OPS_DIMENSION_MISMATCH;
	}
	if (value != NULL) {
		memcpy(output->data, value, output->data_size * output->item_size);
	}
	else if (value_float != NULL) {
		memcpy(output->data, value_float, output->data_size * output->item_size);
	}
	else if (value_floats != NULL) {
		memcpy(output->data, value_floats, output->data_size * output->item_size);
	}
	else if (value_int != NULL) {
		memcpy(output->data, value_int, output->data_size * output->item_size);
	}
	else if (value_ints != NULL) {
		memcpy(output->data, value_ints, output->data_size * output->item_size);
	}
	else {
		return OPS_UNDEFINED;
	}
	return OPS_SUCCESS;

}