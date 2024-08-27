#include "shape_calc.h"
int set_broadcast_shape(struct tensor* A, struct tensor* B, struct tensor* C) {
	int64_t* dims = NULL, i = 0, j = 0, k = 0, c_dimsize = 0;
	if (A == NULL || B == NULL || C == NULL) return OPS_INPUT_IS_NULL;

	if (A->dimension_size > B->dimension_size) c_dimsize = A->dimension_size;
	else c_dimsize = B->dimension_size;
	dims = calloc(c_dimsize , sizeof(int64_t));
	if (dims == NULL) return OPS_ALLOCATION_FAIL;
	i = A->dimension_size - 1;
	j = B->dimension_size - 1;
	k = c_dimsize - 1;
	// assign the larger on from right to left
	while (i >= 0 || j >= 0 || k >= 0) {
		if (i >= 0) {
			dims[k] = A->dimension[i];
		}
		if (j >= 0) {
			if (B->dimension[j] > dims[k]) {
				dims[k] = B->dimension[j];
			}
		}
		i--; j--; k--;
	}

	resize_tensor(C, dims, c_dimsize, A->type);
	C->is_size_unknown = 0;
	free(dims);
	return OPS_SUCCESS;
}

int set_squeeze_shape(struct tensor* data, struct tensor* axes, struct tensor* squeezed) {
	int error = 0;
	int64_t  i = 0, * cur = NULL;
	struct dynamic_array* new_dims = NULL;
	struct tensor_iterator* axes_it = NULL;

	if (data == NULL || squeezed == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	new_dims = create_darray(sizeof(int64_t));
	if (new_dims == NULL) return OPS_ALLOCATION_FAIL;

	if (axes == NULL) {
		// remove every dimension with size 1
		for (i = 0; i < data->dimension_size; i++) {
			if (data->dimension[i] != 1) {
				pushback_darray(new_dims, &data->dimension[i]);
			}
		}
	}
	else {
		axes_it = create_tensor_iterator(axes);
		if (axes_it == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		for (i = 0; i < data->dimension_size; i++) {
			pushback_darray(new_dims, &data->dimension[i]);
		}
		while (1) {
			cur = get_data_tensor_iter(axes_it);
			if (*cur < 0) {	// is negative
				*(int64_t*)get_darray(new_dims, *cur + data->dimension_size) = -1;
			}
			else {
				*(int64_t*)get_darray(new_dims, *cur) = -1;
			}
			if (!is_not_done_tensor_iter(axes_it)) break;
			next_tensor_iter(axes_it);
		}
		i = 0;
		while (i < new_dims->size) {
			if (*(int64_t*)get_darray(new_dims, i) == -1) {
				delete_darray(new_dims, i);
			}
			else {
				i++;
			}
		}
	}
	error = resize_tensor(squeezed, (int64_t*)new_dims->data, new_dims->size, data->type);
	if (error == 1) error = OPS_SUCCESS;

cleanup:
	release_darray(&new_dims);
	release_tensor_iterator(&axes_it);
	return error;
}

int set_unsqueeze_shape(struct tensor* data, struct tensor* axes, struct tensor* expanded) {
	int error = 0;
	int64_t  i = 0, j = 0, output_index = 0, value = -1;
	struct dynamic_array* new_dims = NULL;
	if (data == NULL || axes == NULL || expanded == NULL) {
		return OPS_INPUT_IS_NULL;
	}

	new_dims = create_darray(sizeof(int64_t));
	if (new_dims == NULL) return OPS_ALLOCATION_FAIL;
	for (i = 0; i < data->dimension_size + axes->data_size; i++) {	// new dimension size
		pushback_darray(new_dims, &value);
	}
	for (i = 0; i < axes->data_size; i++) {
		if (((int64_t*)axes->data)[i] < 0) {		// change new added axes to 1
			*(int64_t*)get_darray(new_dims, ((int64_t*)axes->data)[i] + new_dims->size) = 1;
		}
		else {
			*(int64_t*)get_darray(new_dims, ((int64_t*)axes->data)[i]) = 1;
		}
	}
	for (i = 0; i < new_dims->size; i++) {			// change original axes to it's own value
		if (*(int64_t*)get_darray(new_dims, i) == -1) {
			*(int64_t*)get_darray(new_dims, i) = data->dimension[j];
			j++;
		}
	}
	if (j != data->dimension_size) return  OPS_INVALID_ARGUMENT;

	error = resize_tensor(expanded, (int64_t*)new_dims->data, new_dims->size, data->type);
	release_darray(&new_dims);

	if (error != 1) {
		return OPS_ALLOCATION_FAIL;
	}
	return OPS_SUCCESS;
}

int set_transpose_shape(struct tensor* data, int64_t* perm, struct tensor* transposed) {
	int error = 0;
	int64_t  i = 0, j = 0, output_index = 0, value = -1;
	struct dynamic_array* new_dims = NULL;
	if (data == NULL || perm == NULL || transposed == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	new_dims = create_darray(sizeof(int64_t));
	for (i = 0; i < data->dimension_size; i++) {
		pushback_darray(new_dims, &data->dimension[perm[i]]);
	}
	if (new_dims->size != data->dimension_size) {
		release_darray(&new_dims);
		return OPS_INVALID_ARGUMENT;
	}
	error = resize_tensor(transposed, (int64_t*)new_dims->data, new_dims->size, data->type);
	release_darray(&new_dims);
	if (error != 1) {
		return OPS_ALLOCATION_FAIL;
	}
	return OPS_SUCCESS;
}

int set_matmul_shape(struct tensor* a, struct tensor* b, struct tensor* c) {
	int64_t insert_val = 1, i = 0;
	int error = 0;
	if (a == NULL || b == NULL || c == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	struct dynamic_array* a_dim = NULL, * b_dim = NULL, * c_dim = NULL;
	a_dim = create_darray_array(a->dimension, a->dimension_size, sizeof(int64_t));
	b_dim = create_darray_array(b->dimension, b->dimension_size, sizeof(int64_t));
	c_dim = create_darray(sizeof(int64_t));
	if (a->dimension_size == 1) { // prepend if dim is 1
		pushfront_darray(a_dim, &insert_val);
	}
	if (b->dimension_size == 1) { // append if dim is 1 &insert_val);
		pushback_darray(b_dim, &insert_val);
	}

	if (*(int64_t*)back_darray(a_dim) != *(int64_t*)get_darray(b_dim, b_dim->size - 2)) {
		error = OPS_DIMENSION_MISMATCH;
		goto cleanup;
	}
	// push until they both have same dimension size
	while (a_dim->size < b_dim->size) {
		pushfront_darray(a_dim, &insert_val);
	}
	while (a_dim->size > b_dim->size) {
		pushfront_darray(b_dim, &insert_val);
	}

	for (i = 0; i < a_dim->size - 2; i++) {
		if (*(int64_t*)get_darray(a_dim, i) == *(int64_t*)get_darray(b_dim, i))pushback_darray(c_dim, get_darray(a_dim, i));
		else {
			if (*(int64_t*)get_darray(a_dim, i) == 1)pushback_darray(c_dim, (int64_t*)get_darray(b_dim, i));
			else if (*(int64_t*)get_darray(b_dim, i) == 1) pushback_darray(c_dim, (int64_t*)get_darray(a_dim, i));
			else {
				error = OPS_NOT_BROADCASTABLE;

				goto cleanup;
			}

		};
	}
	pushback_darray(c_dim, (int64_t*)get_darray(a_dim, a_dim->size - 2));
	pushback_darray(c_dim, (int64_t*)back_darray(b_dim));
	resize_tensor(c, (int64_t*)c_dim->data, c_dim->size, a->type);
	error = OPS_SUCCESS;
cleanup:
	release_darray(&a_dim);
	release_darray(&b_dim);
	release_darray(&c_dim);
	return error;

}

int set_slice_shape(struct tensor* data, int64_t* starts, int64_t* ends, int64_t* axes, int64_t* steps, struct tensor* output) {
	int64_t  i = 0, push_back_value = 0;
	int error = 0;
	struct dynamic_array* ouput_dim = NULL;
	if (data == NULL || starts == NULL || ends == NULL || steps == NULL || output == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	for (i = 0; i < data->dimension_size; i++) {
		if (axes[i] < 0)axes[i] += data->dimension_size;
	}
	ouput_dim = create_darray(sizeof(int64_t));
	if (ouput_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	for (i = 0; i < data->dimension_size; i++) {
		push_back_value = ends[axes[i]] - starts[axes[i]] / steps[axes[i]];
		if (push_back_value <= 0) {
			error = OPS_INVALID_ARGUMENT;
			goto cleanup;
		}
		pushback_darray(ouput_dim, &push_back_value);
	}
	resize_tensor(output, (int64_t*)ouput_dim->data, ouput_dim->size, data->type);
	error = OPS_SUCCESS;
cleanup:
	release_darray(&ouput_dim);
	return error;
}

int set_gemm_shape(struct tensor* A, struct tensor* B, int64_t* transA, int64_t* transB, struct tensor* Y) {
	int64_t temp = 0;
	int error = 0;
	struct dynamic_array* y_dim = NULL, * a_dim = NULL, * b_dim = NULL;
	if (A == NULL || B == NULL || transA == NULL || transB == NULL || Y == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	if (A->dimension_size != 2 || B->dimension_size != 2) {
		return OPS_DIMENSION_MISMATCH;
	}
	a_dim = create_darray_array(A->dimension, A->dimension_size, sizeof(int64_t));
	if (a_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	b_dim = create_darray_array(B->dimension, B->dimension_size, sizeof(int64_t));
	if (b_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	// apply transpose
	if (*transA == 1) {
		temp = *(int64_t*)get_darray(a_dim, 0);
		*(int64_t*)get_darray(a_dim, 0) = *(int64_t*)get_darray(a_dim, 1);
		*(int64_t*)get_darray(a_dim, 1) = temp;
	}
	if (*transB == 1) {
		temp = *(int64_t*)get_darray(b_dim, 0);
		*(int64_t*)get_darray(b_dim, 0) = *(int64_t*)get_darray(b_dim, 1);
		*(int64_t*)get_darray(b_dim, 1) = temp;
	}
	y_dim = create_darray(sizeof(int64_t));
	if (y_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	pushback_darray(y_dim, get_darray(a_dim, 0));
	pushback_darray(y_dim, get_darray(b_dim, 1));

	resize_tensor(Y, y_dim->data, y_dim->size, A->type);
	print_tensor(Y);
	error = OPS_SUCCESS;
cleanup:
	release_darray(&a_dim);
	release_darray(&b_dim);
	release_darray(&y_dim);

	return error;
}

int set_concat_shape(int64_t* axis, struct list* inputs, struct tensor* concat_result) {
	int error = 0;
	int64_t rank = 0, i = 0, j = 0;;
	struct dynamic_array* y_dim = NULL;
	struct tensor* cur_tensor = NULL;
	if (axis == NULL || inputs == NULL || concat_result == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	if (inputs->size == 0) {
		return OPS_INPUT_IS_NULL;
	}
	rank = ((struct tensor*)get_list(inputs, 0))->dimension_size;

	for (i = 0; i < inputs->size; i++) {
		if (((struct tensor*)get_list(inputs, i))->dimension_size != rank) {
			return OPS_DIMENSION_MISMATCH;
		}
	}
	if (*axis < 0) axis += rank;
	y_dim = create_darray_array(((struct tensor*)get_list(inputs, 0))->dimension, ((struct tensor*)get_list(inputs, 0))->dimension_size, sizeof(int64_t));
	if (y_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	for (i = 1; i < inputs->size; i++) {
		cur_tensor = get_list(inputs, i);
		for (j = 0; j < rank; j++) {
			if (j == *axis) {
				*(int64_t*)get_darray(y_dim, j) += cur_tensor->dimension[j];
			}
			else {
				if (*(int64_t*)get_darray(y_dim, j) != cur_tensor->dimension[j]) {
					printf("here\n");
					error = OPS_DIMENSION_MISMATCH;
					goto cleanup;
				}
			}
		}
	}
	resize_tensor(concat_result, y_dim->data, y_dim->size, ((struct tensor*)get_list(inputs, 0))->type);
	error = OPS_SUCCESS;
cleanup:
	release_darray(&y_dim);
	return error;
}

int set_split_shape(int64_t axis, int64_t num_outputs, struct tensor* input, int64_t* split, struct list* outputs) {
	int error = 0;
	int64_t i = 0, j = 0, axis_value = 0, rank = 0, * y_dim = NULL;
	struct dynamic_array* output_dim = NULL;
	if (input == NULL || outputs == NULL) return NULL;
	y_dim = malloc(input->dimension_size * sizeof(int64_t));
	if (y_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	memcpy_s(y_dim, input->dimension_size * sizeof(int64_t), input->dimension, input->dimension_size * sizeof(int64_t));
	for (i = 0; i < num_outputs; i++) {
		if (split == NULL) {
			if (i == num_outputs - 1) {
				if (input->dimension[axis] % num_outputs != 0) {
					axis_value = input->dimension[axis] % num_outputs;
				}
				else {
					axis_value = input->dimension[axis] / num_outputs;
				}
			}
			else {
				axis_value = input->dimension[axis] / num_outputs;
			}
		}
		else {
			axis_value = split[i];
		}
		y_dim[axis] = axis_value;
		resize_tensor(get_list(outputs, i), y_dim, input->dimension_size, input->type);
	}
	error = OPS_SUCCESS;
cleanup:
	safe_free(&y_dim);
	return error;
}

int set_reshaped_shape(struct tensor* input, struct tensor* shape, struct tensor* reshaped) {
	int error = 0;
	int64_t i = 0,total= 1;

	if (input == NULL || shape == NULL || reshaped == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	for (i = 0; i < shape->data_size; i++) {
		total *= ((int64_t*)shape->data)[i];
	}
	if (input->data_size != total) return OPS_DIMENSION_MISMATCH;
	resize_tensor(reshaped, shape->data, shape->data_size, input->type);
	return OPS_SUCCESS;
}

int set_pad_shape(struct tensor* data, struct tensor* pads, struct tensor* axes, struct tensor* output){
	int error = 0;
	int64_t i = 0;
	struct dynamic_array* output_dim = NULL;
	if (data == NULL || pads == NULL || axes== NULL||output == NULL) return OPS_INPUT_IS_NULL;

	output_dim = create_darray_array(data->dimension, data->dimension_size, sizeof(int64_t));
	if (output_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	for (i = 0; i < axes->data_size; i++) {
		*(int64_t*)get_darray(output_dim, ((int64_t*)axes->data)[i]) += ((int64_t*)pads->data)[i] + ((int64_t*)pads->data)[i + axes->data_size];
	}
	resize_tensor(output, output_dim->data, output_dim->size, data->type);

	error = OPS_SUCCESS;
cleanup:
	release_darray(&output_dim);
	return error;
}

int set_conv_shape(struct tensor* x, struct tensor* w, struct tensor* y, int64_t* pads, int64_t* strides) {
	int error = 0;
	int64_t i = 0, temp = 0; ;
	struct dynamic_array* y_dim = NULL;
	if (x == NULL || w == NULL || y == NULL || pads == NULL || strides == NULL) return OPS_INPUT_IS_NULL;
	y_dim = create_darray(sizeof(int64_t));
	// batch size
	pushback_darray(y_dim, &x->dimension[0]);
	// new channel
	pushback_darray(y_dim, &w->dimension[0]);
	// features
	for (i = 2; i < x->dimension_size; i++) {
		temp = (x->dimension[i] - w->dimension[i] + pads[i - 2] + pads[i - 2 + x->dimension_size - 2]) / strides[i - 2] + 1;
		pushback_darray(y_dim, &temp);
	}
	resize_tensor(y, y_dim->data, y_dim->size, x->type);
	error = OPS_SUCCESS;
	release_darray(&y_dim);
	return error;
}

int set_lstm_shape(struct tensor* x, int64_t num_direction, int64_t hidden_size, struct tensor* y, struct tensor* y_h, struct tensor* y_c) {
	int error = 0;
	struct dynamic_array* dims = NULL;
	if (x == NULL || y == NULL || y_h == NULL || y_c == NULL) return OPS_INPUT_IS_NULL;

	dims = create_darray(sizeof(int64_t));
	if (dims == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	// Set y dim
	pushback_darray(dims, &x[0]);
	pushback_darray(dims, &num_direction);
	pushback_darray(dims, &x[1]);
	pushback_darray(dims, &hidden_size);
	resize_tensor(y, (int64_t*)dims->data, dims->size, x->type);
	popfront_darray(dims);
	resize_tensor(y_h, (int64_t*)dims->data, dims->size, x->type);
	resize_tensor(y_c, (int64_t*)dims->data, dims->size, x->type);
	error = OPS_SUCCESS;
cleanup:
	release_darray(&dims);
	return error;

}

int set_reducemean_shape(struct tensor* axes, int64_t* noop_with_empty_axes, int64_t keepdims, struct tensor* data, struct tensor* reduced) {
	int error = 0;
	int64_t num = 0, i = 0;
	struct dynamic_array* reduced_dim = NULL;
	if (axes == NULL || data == NULL || reduced == NULL) return OPS_INPUT_IS_NULL;
	if (axes == NULL) {
		if (*noop_with_empty_axes == 1) {
			resize_tensor(reduced, data->dimension, data->dimension_size, data->type);
			error = OPS_SUCCESS;
			goto cleanup;
		}
	}
	printf("\n\n\ncalculate reduce shape\n\n\n");
	reduced_dim = create_darray_array(data->dimension, data->dimension_size, sizeof(int64_t));
	if (reduced_dim == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	for (i = 0; i < axes->data_size; i++) {
		num = ((int64_t*)axes)[i];
		if (keepdims) {
			*(int64_t*)get_darray(reduced_dim, num) = 1;
		}
		else {
			*(int64_t*)get_darray(reduced_dim, num) = 0;
		}
	}
	if (keepdims == 0) {
		for (i = 0; i < reduced_dim->size; i++) {
			if (*(int64_t*)get_darray(reduced_dim, num) == 0) {
				delete_darray(reduced_dim, i);
				i--;
			}
		}
	}
	resize_tensor(reduced, (int64_t*)reduced_dim->data, reduced_dim->size, data->type);
	error = OPS_SUCCESS;
cleanup:
	release_darray(&reduced_dim);
	return error;
}