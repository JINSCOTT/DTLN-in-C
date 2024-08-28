#include "ops.h"

int16_t broadcast_function(struct tensor* A, struct tensor* B, struct tensor* C, NODE_TYPE op_type) {
	int error = 0;
	int64_t* c_dim = NULL, c_dim_length = 0;
	struct tensor_iterator* A_iter = NULL, * B_iter = NULL, * C_iter = NULL;
	// Check if input is NULL;
	if (A == NULL || B == NULL || C == NULL) return OPS_INPUT_IS_NULL;
	// Check BROADCASTABLE;
	c_dim_length = is_shape_broadcastable_tensor(A, B, &c_dim);
	if (c_dim_length <= 0) return OPS_NOT_BROADCASTABLE;
	// Check if C dimension match
	if (!is_dimension_tensor(C, c_dim, c_dim_length))return OPS_DIMENSION_MISMATCH;

	A_iter = create_tensor_iterator(A);
	if (A_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	B_iter = create_tensor_iterator(B);
	if (B_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	C_iter = create_tensor_iterator(C);
	if (C_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	// Main loop
	while (1) {
		if (A->type == DATATYPE_FLOAT32) {
			if (op_type == Add) {
				*(float*)get_data_tensor_iter(C_iter) = *(float*)get_data_tensor_iter(A_iter) + *(float*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Sub) {
				*(float*)get_data_tensor_iter(C_iter) = *(float*)get_data_tensor_iter(A_iter) - *(float*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Mul) {
				*(float*)get_data_tensor_iter(C_iter) = *(float*)get_data_tensor_iter(A_iter) * *(float*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Div) {
				*(float*)get_data_tensor_iter(C_iter) = *(float*)get_data_tensor_iter(A_iter) / *(float*)get_data_tensor_iter(B_iter);
			}
			else {
				error =  OPS_UNDEFINED;
				goto cleanup;
			}

		}
		else if (A->type == DATATYPE_INT32) {
			if (op_type == Add) {
				*(int32_t*)get_data_tensor_iter(C_iter) = *(int32_t*)get_data_tensor_iter(A_iter) + *(int32_t*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Sub) {
				*(int32_t*)get_data_tensor_iter(C_iter) = *(int32_t*)get_data_tensor_iter(A_iter) - *(int32_t*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Mul) {
				*(int32_t*)get_data_tensor_iter(C_iter) = *(int32_t*)get_data_tensor_iter(A_iter) * *(int32_t*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Div) {
				*(int32_t*)get_data_tensor_iter(C_iter) = *(int32_t*)get_data_tensor_iter(A_iter) / *(int32_t*)get_data_tensor_iter(B_iter);
			}
			else {
				error = OPS_UNDEFINED;
				goto cleanup;
			}

		}
		else if (A->type == DATATYPE_INT64) {
			if (op_type == Add) {
				*(int64_t*)get_data_tensor_iter(C_iter) = *(int64_t*)get_data_tensor_iter(A_iter) + *(int64_t*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Sub) {
				*(int64_t*)get_data_tensor_iter(C_iter) = *(int64_t*)get_data_tensor_iter(A_iter) - *(int64_t*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Mul) {
				*(int64_t*)get_data_tensor_iter(C_iter) = *(int64_t*)get_data_tensor_iter(A_iter) * *(int64_t*)get_data_tensor_iter(B_iter);
			}
			else if (op_type == Div) {
				*(int64_t*)get_data_tensor_iter(C_iter) = *(int64_t*)get_data_tensor_iter(A_iter) / *(int64_t*)get_data_tensor_iter(B_iter);
			}
			else {
				error = OPS_UNDEFINED;
				goto cleanup;
			}
		}
		else {	// type not supported
			error =  OPS_TYPE_NOT_SUPPORTED;
			goto cleanup;
		}

		if (!is_not_done_tensor_iter(C_iter)) {
			break;
		}
		// iterate
		next_tensor_iter(C_iter);
		if (is_not_done_tensor_iter(B_iter)) {
			next_tensor_iter(B_iter);
		}
		else {
			reset_tensor_iter(B_iter);
		}
		if (is_not_done_tensor_iter(A_iter)) {
			next_tensor_iter(A_iter);
		}
		else {
			reset_tensor_iter(A_iter);
		}
	}
	error = OPS_SUCCESS;
cleanup:
	release_tensor_iterator(&A_iter);
	release_tensor_iterator(&B_iter);
	release_tensor_iterator(&C_iter);
	return error ;
}
int tanh_function(struct tensor* A, struct tensor* B) {
	int error = 0;
	struct tensor_iterator* A_iter = NULL, * B_iter = NULL;
	if (A == NULL || B == NULL) return OPS_INPUT_IS_NULL;
	if (!is_shape_compatible_tensor(A, B)) {
		return OPS_DIMENSION_MISMATCH;
	}
	A_iter = create_tensor_iterator(A);
	if (A_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	B_iter = create_tensor_iterator(B);
	if (B_iter <= 0){
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	while (1) {
		if (A->type == DATATYPE_FLOAT32) {
			*(float*)get_data_tensor_iter(B_iter) = tanhf(*(float*)get_data_tensor_iter(A_iter));
		}
		else {
			error  = OPS_TYPE_NOT_SUPPORTED;
			goto cleanup;
		}
		if (!is_not_done_tensor_iter(B_iter)) {
			break;
		}
		next_tensor_iter(A_iter);
		next_tensor_iter(B_iter);
	}
	error = OPS_SUCCESS;
cleanup:
	release_tensor_iterator(&A_iter);
	release_tensor_iterator(&B_iter);
	return error;
}
int sigmoid_function(struct tensor* A, struct tensor* B) {
	int error = 0;
	struct tensor_iterator* A_iter = NULL, * B_iter = NULL;
	if (A == NULL || B == NULL) return OPS_INPUT_IS_NULL;
	if (!is_shape_compatible_tensor(A, B)) {
		return OPS_DIMENSION_MISMATCH;
	}
	A_iter = create_tensor_iterator(A);
	if (A_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	B_iter = create_tensor_iterator(B);
	if (B_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}

	while (1) {
		if (A->type == DATATYPE_FLOAT32) {
			*(float*)get_data_tensor_iter(B_iter) = 1.f / (1.0f + expf(-*(float*)get_data_tensor_iter(A_iter)));
		}
		else {
			error = OPS_TYPE_NOT_SUPPORTED;
			goto cleanup;
		}
		if (!is_not_done_tensor_iter(B_iter)) {
			break;
		}
		next_tensor_iter(A_iter);
		next_tensor_iter(B_iter);
	}
	error = OPS_SUCCESS;
cleanup:
	release_tensor_iterator(&A_iter);
	release_tensor_iterator(&B_iter);
	return error;
}
int relu_function(struct tensor* A, struct tensor* B) {
	int error = 0;
	struct tensor_iterator* A_iter = NULL, * B_iter = NULL;
	if (A == NULL || B == NULL) return OPS_INPUT_IS_NULL;
	if (!is_shape_compatible_tensor(A, B)) {
		return OPS_DIMENSION_MISMATCH;
	}
	A_iter = create_tensor_iterator(A);
	if (A_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	B_iter = create_tensor_iterator(B);
	if (B_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}

	while (1) {
		if (A->type == DATATYPE_FLOAT32) {
			*(float*)get_data_tensor_iter(B_iter) = max(0.0f, *(float*)get_data_tensor_iter(A_iter));
		}
		else {
			error = OPS_TYPE_NOT_SUPPORTED;
			goto cleanup;
		}
		if (!is_not_done_tensor_iter(B_iter)) {
			break;
		}
		next_tensor_iter(A_iter);
		next_tensor_iter(B_iter);
	}
	error = OPS_SUCCESS;
cleanup:
	release_tensor_iterator(&A_iter);
	release_tensor_iterator(&B_iter);
	return error;
}
int sqrt_function(struct tensor* A, struct tensor* B) {
	int error = 0;
	struct tensor_iterator* A_iter = NULL, * B_iter = NULL;
	if (A == NULL || B == NULL) return OPS_INPUT_IS_NULL;
	if (!is_shape_compatible_tensor(A, B)) {
		return OPS_DIMENSION_MISMATCH;
	}
	A_iter = create_tensor_iterator(A);
	if (A_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	B_iter = create_tensor_iterator(B);
	if (B_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	while (1) {
		if (A->type == DATATYPE_FLOAT32) {
			*(float*)get_data_tensor_iter(B_iter) = sqrtf(*(float*)get_data_tensor_iter(A_iter));
		}
		else {
			error = OPS_TYPE_NOT_SUPPORTED;
			goto cleanup;
		}
		if (!is_not_done_tensor_iter(B_iter)) {
			break;
		}
		next_tensor_iter(A_iter);
		next_tensor_iter(B_iter);
	}
	error = OPS_SUCCESS;
cleanup:
	release_tensor_iterator(&A_iter);
	release_tensor_iterator(&B_iter);
	return error;
}
int copy_function(struct tensor* A, struct tensor* B) {
	int error = 0;
	struct tensor_iterator* A_iter = NULL, * B_iter = NULL;
	if (A == NULL || B == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	if (!is_shape_compatible_tensor(A, B)) {
		return OPS_DIMENSION_MISMATCH;
	}
	A_iter = create_tensor_iterator(A);
	if (A_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	B_iter = create_tensor_iterator(B);
	if (B_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}

	while (1) {
		// Assign
		memcpy(get_data_tensor_iter(B_iter), get_data_tensor_iter(A_iter), A->item_size);
		if (!is_not_done_tensor_iter(B_iter)) {
			break;
		}
		next_tensor_iter(A_iter);
		next_tensor_iter(B_iter);
	}
	error = OPS_SUCCESS;
cleanup:
	release_tensor_iterator(&A_iter);
	release_tensor_iterator(&B_iter);
	return error;
}
int slice_function(struct tensor* data, struct tensor* output, int64_t* starts, int64_t* ends, int64_t* axes, int64_t* steps) {
	int error = 0;
	int64_t i = 0, item_size = 0;
	struct tensor_iterator* data_iter = NULL, * output_iter = NULL;
	int64_t* data_index = NULL;	
	if (data == NULL || output == NULL) return OPS_INPUT_IS_NULL;
	data_iter = create_tensor_iterator(data);
	if (data_iter <= 0) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	output_iter = create_tensor_iterator(output);
	if (output_iter <= 0) return OPS_ALLOCATION_FAIL;
	item_size = data_iter->stride[data_iter->dimension_size - 1];
	data_index = (int64_t*)malloc(data_iter->dimension_size * sizeof(int64_t));

	for (i = 0; i < data_iter->dimension_size; i++) {		// starts and end inside dimension bounds
		if (starts[axes[i]] < 0)starts[axes[i]] += data_iter->dimension_size;
		if (starts[axes[i]] > data_iter->dimension[i])starts[axes[i]] = data_iter->dimension[i];
		if (ends[axes[i]] < 0)ends[axes[i]] += data_iter->dimension_size;
		if (ends[axes[i]] > data_iter->dimension[i])ends[axes[i]] = data_iter->dimension[i];
		data_index[i] = starts[axes[i]];
	}
	goto_tensor_iter(data_iter, data_index);
	while (data_index[0] < ends[0])
	{
		// assign
		memcpy(get_data_tensor_iter(output_iter), get_data_tensor_iter(data_iter), data->item_size);
		// Calculate next index
		data_index[data_iter->dimension_size - 1] += steps[axes[data_iter->dimension_size - 1]];
		i = data_iter->dimension_size - 1;
		while (data_index[i] >= ends[axes[i]] && i >= 1) {
			data_index[i] = starts[axes[i]];
			data_index[i - 1] += steps[axes[i - 1]];
			i--;
		}
		goto_tensor_iter(data_iter, data_index);
		next_tensor_iter(output_iter);
	}
cleanup:
	free(data_index);
	release_tensor_iterator(&data_iter);
	release_tensor_iterator(&output_iter);
	return OPS_SUCCESS;
}

int transpose_function(struct tensor* data, struct tensor* transposed, int64_t* perm) {
	int64_t i = 0, item_size = 0, j = 0;;
	struct tensor_iterator* data_iter = NULL, * transposed_iter = NULL;
	int64_t* transposed_coordinate = NULL;
	if (data == NULL || perm == NULL || transposed == NULL) return OPS_INPUT_IS_NULL;
	data_iter = create_tensor_iterator(data);
	if (data_iter <= 0) return OPS_ALLOCATION_FAIL;
	transposed_iter = create_tensor_iterator(transposed);
	if (transposed_iter <= 0) return OPS_ALLOCATION_FAIL;
	transposed_coordinate = malloc(data_iter->dimension_size * sizeof(int64_t));
	if (transposed_coordinate == NULL) return OPS_ALLOCATION_FAIL;
	item_size = data_iter->stride[data_iter->dimension_size - 1];
	memcpy_s(get_data_tensor_iter(transposed_iter), item_size, get_data_tensor_iter(data_iter), item_size);
	while (1) { 
		// calculate next
		if (!is_not_done_tensor_iter(data_iter)) {
			break;
		}
		next_tensor_iter(data_iter);
		for (i = 0; i < data_iter->dimension_size; i++) {
			transposed_coordinate[i] = data_iter->coordinate[perm[i]];
		}
		j++;
		goto_tensor_iter(transposed_iter, transposed_coordinate);
		// assign
		memcpy_s(get_data_tensor_iter(transposed_iter), item_size, get_data_tensor_iter(data_iter), item_size);
	}
	release_tensor_iterator(&transposed_iter);
	release_tensor_iterator(&data_iter);
	free(transposed_coordinate);
	return OPS_SUCCESS;
}
int matmul_array(void* A, void* B, void* C, int64_t n, int64_t m, int64_t p, int type) {
	int64_t i = 0, j = 0, k = 0;
	if (type == DATATYPE_FLOAT32) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < p; j++) {
				((float*)C)[i * p + j] = 0;
				for (k = 0; k < m; k++) {
					((float*)C)[i * p + j] += ((float*)A)[i * m + k] * ((float*)B)[k * p + j];
				}
			}
		}
	}
	else if (type == DATATYPE_FLOAT32) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < p; j++) {
				((int32_t*)C)[i * p + j] = 0;
				for (k = 0; k < m; k++) {
					((int32_t*)C)[i * p + j] += ((int32_t*)A)[i * m + k] * ((int32_t*)B)[k * p + j];
				}
			}
		}
	}
	else if (type == DATATYPE_INT64) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < p; j++) {
				((int64_t*)C)[i * p + j] = 0;
				for (k = 0; k < m; k++) {
					((int64_t*)C)[i * p + j] += ((int64_t*)A)[i * m + k] * ((int64_t*)B)[k * p + j];
				}
			}
		}
	}
	else {
		return 0;
	}
	return 1;
}

int mean_array(void* array, void* result, int64_t num_elements, int type) {
	int64_t i = 0;
	if (array == NULL || result == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT32) {
		*(float*)result = 0;
		for (i = 0; i < num_elements; i++) {
			*(float*)result += ((float*)array)[i];
		}
		*(float*)result /= (float)num_elements;
	}
	else if (type == DATATYPE_INT32) {
		*(int32_t*)result = 0;
		for (i = 0; i < num_elements; i++) {
			*(int32_t*)result += ((int32_t*)array)[i];
		}
		*(int32_t*)result /= (int32_t)num_elements;
	}
	else if (type == DATATYPE_INT64) {
		*(int64_t*)result = 0;
		for (i = 0; i < num_elements; i++) {
			*(int64_t*)result += ((int64_t*)array)[i];
		}
		*(int64_t*)result /= (int64_t)num_elements;
	}
	return OPS_SUCCESS;
}
int matmul_function(struct tensor* a, struct tensor* b, struct tensor* c) {
	int error = 0;
	int64_t m = 0, n = 0, p = 0, a_index = 0, b_index = 0, c_index = 0, a_matrix_size = 0, b_matrix_size = 0, c_matrix_size = 0;
	struct tensor_iterator* a_iter = NULL, * b_iter = NULL, * c_iter = NULL;
	if (a == NULL || b == NULL || c == NULL)  return OPS_INPUT_IS_NULL;

	if (a->dimension[a->dimension_size - 1] != b->dimension[b->dimension_size - 2] ||
		a->dimension[a->dimension_size - 2] != c->dimension[c->dimension_size - 2] ||
		b->dimension[b->dimension_size - 1] != c->dimension[c->dimension_size - 1]) {
		return OPS_DIMENSION_MISMATCH;
	}
	a_iter = create_tensor_iterator(a);
	if (a_iter == NULL) return OPS_ALLOCATION_FAIL;
	b_iter = create_tensor_iterator(b);
	if (b_iter == NULL) return OPS_ALLOCATION_FAIL;
	c_iter = create_tensor_iterator(c);
	if (c_iter == NULL) return OPS_ALLOCATION_FAIL;


	n = a->dimension[a->dimension_size - 2];
	m = a->dimension[a->dimension_size - 1];
	p = b->dimension[b->dimension_size - 1];

	a_matrix_size = n * m;
	b_matrix_size = m * p;
	c_matrix_size = n * p;

	while (1) {
		error = matmul_array(a_iter->data, b_iter->data, c_iter->data, n, m, p, a->type);
		if (error == 0) return OPS_TYPE_NOT_SUPPORTED;


		c_index += c_matrix_size;
		error = goto_1d_tensor_iter(c_iter, c_index);
		if (error <= 0) break;

		a_index += a_matrix_size;
		goto_1d_tensor_iter(a_iter, a_index);
		if (!is_not_done_tensor_iter(a_iter)) {
			reset_tensor_iter(a_iter);
			a_index = 0;
		}
		b_index += b_matrix_size;
		goto_1d_tensor_iter(b_iter, b_index);
		if (!is_not_done_tensor_iter(b_iter)) {
			reset_tensor_iter(b_iter);
			b_index = 0;
		}
	}
	release_tensor_iterator(&a_iter);
	release_tensor_iterator(&b_iter);
	release_tensor_iterator(&c_iter);
	return OPS_SUCCESS;
}

int pad_function(char* mode, struct tensor* data, struct tensor* pads, struct tensor* constant_value, struct tensor* axes, struct tensor* output) {
	struct tensor_iterator* data_iter = NULL, * output_iter = NULL;
	int error = 0;
	int64_t i = 0, is_pad = false, * pad_array = NULL;
	if (data == NULL || output == NULL)  return OPS_INPUT_IS_NULL;
	data_iter = create_tensor_iterator(data);
	if (data_iter == NULL) return OPS_ALLOCATION_FAIL;
	output_iter = create_tensor_iterator(output);
	if (output_iter == NULL) return OPS_ALLOCATION_FAIL;

	if (strcmp(mode, "constant") == 0) {
		pad_array = calloc(data->dimension_size, sizeof(int64_t));
		if (pad_array == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}

		for (i = 0; i < axes->data_size; i++) {
			pad_array[((int64_t*)axes->data)[i]] = ((int64_t*)pads->data)[i];

		}
		while (true)
		{
			if (data->type == DATATYPE_FLOAT32 || data->type == DATATYPE_INT32) {
				if (constant_value != NULL)*(float*)get_data_tensor_iter(output_iter) = *(float*)constant_value->data;
				else *(float*)get_data_tensor_iter(output_iter) = 0;
			}
			else if (data->type == DATATYPE_INT64) {
				if (constant_value != NULL)*(int64_t*)get_data_tensor_iter(output_iter) = *(int64_t*)constant_value->data;
				else *(int64_t*)get_data_tensor_iter(output_iter) = 0;
			}
			else {
				return OPS_TYPE_NOT_SUPPORTED;
			}
			if (!is_not_done_tensor_iter(output_iter)) {
				break;
			}
			next_tensor_iter(output_iter);
		}
		reset_tensor_iter(output_iter);
		while (true)
		{
			is_pad = false;
			for (i = 0; i < data_iter->dimension_size; i++) {
				if (output_iter->coordinate[i] < pad_array[i] || output_iter->coordinate[i] >= data_iter->dimension[i] + pad_array[i]) {
					is_pad = true;

					break;
				}
			}
			if (!is_pad) {
				if (data->type == DATATYPE_FLOAT32 || data->type == DATATYPE_INT32) {
					*(float*)get_data_tensor_iter(output_iter) = *(float*)get_data_tensor_iter(data_iter);
				}
				else if (data->type == DATATYPE_INT64) {
					*(int64_t*)get_data_tensor_iter(output_iter) = *(int64_t*)get_data_tensor_iter(data_iter);
				}
				else {
					return OPS_TYPE_NOT_SUPPORTED;
				}
				next_tensor_iter(data_iter);
			}
			if (!is_not_done_tensor_iter(output_iter)) {
				break;
			}
			next_tensor_iter(output_iter);
		}
	}
	else if (strcmp(mode, "reflect") == 0) {
		return OPS_TYPE_UNIMPLEMENTED;
	}
	else if (strcmp(mode, "edge") == 0) {
		return OPS_TYPE_UNIMPLEMENTED;
	}
	else if (strcmp(mode, "wrap") == 0) {
		return OPS_TYPE_UNIMPLEMENTED;
	}
	else {
		return OPS_UNDEFINED;
	}
cleanup:
	release_tensor_iterator(&data_iter);
	release_tensor_iterator(&output_iter);
	safe_free(&pad_array);
	return OPS_SUCCESS;

}



int pad_function_simple(struct tensor* data, struct tensor* output, int64_t* pads, char* mode, void* constant_value) {
	struct tensor_iterator* data_iter = NULL, * output_iter = NULL;
	int64_t i = 0;
	int16_t is_pad = false;
	if (data == NULL || output == NULL)  return OPS_INPUT_IS_NULL;
	data_iter = create_tensor_iterator(data);
	if (data_iter == NULL) return OPS_ALLOCATION_FAIL;
	output_iter = create_tensor_iterator(output);
	if (output_iter == NULL) return OPS_ALLOCATION_FAIL;
	if (strcmp(mode, "constant") == 0) {
		while (true)
		{
			is_pad = false;
			for (i = 0; i < data_iter->dimension_size; i++) {

				if (output_iter->coordinate[i] < pads[i] || output_iter->coordinate[i] >= data_iter->dimension[i] + pads[i]) {
					is_pad = true;
					break;
				}
			}
			if (is_pad) {
				if (data->type == DATATYPE_FLOAT32 || data->type == DATATYPE_INT32) {
					*(float*)get_data_tensor_iter(output_iter) = *(float*)constant_value;
				}
				else if (data->type == DATATYPE_INT64) {
					*(int64_t*)get_data_tensor_iter(output_iter) = *(int64_t*)constant_value;
				}
				else {
					return OPS_TYPE_NOT_SUPPORTED;
				}
			}
			else {

				if (data->type == DATATYPE_FLOAT32 || data->type == DATATYPE_INT32) {
					*(float*)get_data_tensor_iter(output_iter) = *(float*)get_data_tensor_iter(data_iter);
				}
				else if (data->type == DATATYPE_INT64) {
					*(int64_t*)get_data_tensor_iter(output_iter) = *(int64_t*)get_data_tensor_iter(data_iter);
				}
				else {
					return OPS_TYPE_NOT_SUPPORTED;
				}
				next_tensor_iter(data_iter);
			}
			if (!is_not_done_tensor_iter(output_iter)) {
				break;
			}
			next_tensor_iter(output_iter);
		}

	}
	else if (strcmp(mode, "reflect") == 0) {
		return OPS_TYPE_UNIMPLEMENTED;
	}
	else if (strcmp(mode, "edge") == 0) {
		return OPS_TYPE_UNIMPLEMENTED;
	}
	else if (strcmp(mode, "wrap") == 0) {
		return OPS_TYPE_UNIMPLEMENTED;
	}
	else {
		return OPS_UNDEFINED;
	}

	release_tensor_iterator(&data_iter);
	release_tensor_iterator(&output_iter);
	return OPS_SUCCESS;
}

int gemm_function(struct tensor* a, struct tensor* b, struct tensor* c, struct tensor* output, float alpha, float beta, int64_t transA, int64_t transB) {
	int error = 0;
	int64_t l = 0, m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0;
	int transA_num = CblasNoTrans, transB_num = CblasNoTrans;
	if (a == NULL || b == NULL || output == NULL) return OPS_INPUT_IS_NULL;
	lda = a->dimension[1];
	ldb = b->dimension[1];
	ldc = output->dimension[1];

	if (transA == 0) {
		m = a->dimension[0];
		k = a->dimension[1];
	}
	else {
		transA_num = CblasTrans;
		m = a->dimension[1];
		k = a->dimension[0];
	}
	if (transB == 0) {
		n = b->dimension[1];
	}
	else {
		transB_num = CblasTrans;
		n = b->dimension[0];
	}
	memset(output->data, 0, output->data_size * output->item_size);
	if (c != NULL)memcpy(output->data, c->data, c->data_size * c->item_size);
	cblas_sgemm(CblasRowMajor, transA_num, transB_num, m, n, k, alpha, a->data, lda, b->data, ldb, beta, output->data, ldc);
	return OPS_SUCCESS;
}

int conv_function(struct tensor* x, struct tensor* w, struct tensor* b, struct tensor* y, int64_t* dilations, int64_t groups, int64_t* kernel_shapes, int64_t* pads, int64_t* stride) {
	struct tensor* x_copy = NULL;
	int64_t* padded_dimension = NULL, i = 0, j = 0, features = 1, *x_coordinate = NULL, *y_coordinate = NULL, group_size = 0, batch_size = 0, *new_pad = NULL;
	struct tensor_iterator *x_iter = NULL,  *y_iter = NULL;
	float pad_value = 0;
	int error = 0;
	if (x == NULL || w == NULL || y == NULL)return OPS_INPUT_IS_NULL;
	x_copy = create_tensor_copy(x);
	if (x_copy == NULL) return OPS_ALLOCATION_FAIL;

	if (pads != NULL) {
		padded_dimension = malloc(x->dimension_size * sizeof(int64_t));
		if (padded_dimension == NULL)return OPS_ALLOCATION_FAIL;
		j = 1; // for item size
		j *= x->dimension[0]; padded_dimension[0] = x->dimension[0];
		j *= x->dimension[1]; padded_dimension[1] = x->dimension[1];
		for (i = 2; i < x->dimension_size; i++) {
			padded_dimension[i] = x->dimension[i] + pads[i - 2] + pads[i - 2 + x->dimension_size - 2];
			j *= padded_dimension[i];
		}

		new_pad = malloc(2 * x->dimension_size * sizeof(int64_t));
		if (new_pad == NULL) return OPS_ALLOCATION_FAIL;
		new_pad[0] = 0; new_pad[1] = 0;
		new_pad[x->dimension_size] = 0; new_pad[x->dimension_size + 1] = 0;
		for (i = 2; i < x->dimension_size; i++) {
			new_pad[i] = pads[i - 2];
			new_pad[i + x->dimension_size] = pads[i - 2 + x->dimension_size - 2];
		}
		x_copy = create_tensor(NULL, j, padded_dimension, x->dimension_size, x->type, false);
		if (x_copy == NULL) return OPS_ALLOCATION_FAIL; 
		error = pad_function_simple(x, x_copy, new_pad, "constant", &pad_value);
		if (error != OPS_SUCCESS) {
			goto cleanup;
		}
	}
	else {
		x_copy = create_tensor_copy(x);
	}
	x_iter = create_tensor_iterator(x_copy);
	y_iter = create_tensor_iterator(y);
	if (x_iter == NULL||y_iter== NULL) {
		error =  OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	batch_size = x_iter->dimension[0];
	group_size = x_iter->dimension[1] / groups;

	if (x_copy->dimension_size == 3) {
		x_coordinate = calloc(x_copy->dimension_size, sizeof(int64_t));
		y_coordinate = calloc(y->dimension_size, sizeof(int64_t));
		if (x_coordinate == NULL || y_coordinate == NULL) {
			error =  OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		for (i = 0; i < batch_size; i++) {
			y_coordinate[1] = 0;
			for (j = 0; j < x_copy->dimension[1]; j += group_size) {
				x_coordinate[0] = i;
				x_coordinate[1] = j;
				y_coordinate[0] = i;
				y_coordinate[1] += w->dimension[0];
				goto_tensor_iter(x_iter, x_coordinate);
				goto_tensor_iter(y_iter, y_coordinate);// Same channel and batch position
				if (b == NULL) error = conv1df(get_data_tensor_iter(x_iter), w->data, NULL, get_data_tensor_iter(y_iter),
					group_size, x_copy->dimension[2], w->dimension[0], w->dimension[2], y->dimension[2], stride[0], dilations[0]);
				else error = conv1df(get_data_tensor_iter(x_iter), w->data, b->data, get_data_tensor_iter(y_iter),
					group_size, x_copy->dimension[2], w->dimension[0], w->dimension[2], y->dimension[2], stride[0], dilations[0]);
			}
		}
	}
	else if (x_copy->dimension_size == 4) {
		x_coordinate = calloc(x_copy->dimension_size, sizeof(int64_t));
		y_coordinate = calloc(y->dimension_size, sizeof(int64_t));
		if (x_coordinate == NULL || y_coordinate == NULL) {
			error  = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		for (i = 0; i < batch_size; i++) {
			y_coordinate[1] = 0;
			for (j = 0; j < x_copy->dimension[1]; j += group_size) {
				x_coordinate[0] = i;
				x_coordinate[1] = j;
				y_coordinate[0] = i;
				y_coordinate[1] += w->dimension[0];
				goto_tensor_iter(x_iter, x_coordinate);
				goto_tensor_iter(y_iter, y_coordinate);// Same channel and batch position
				if (b == NULL) error = conv2df(get_data_tensor_iter(x_iter), w->data, NULL, get_data_tensor_iter(y_iter),
					group_size, x_copy->dimension[2], x_copy->dimension[3], w->dimension[0], w->dimension[2], w->dimension[3], y->dimension[2], y->dimension[3], stride, dilations);
				else error = conv2df(get_data_tensor_iter(x_iter), w->data, b->data, get_data_tensor_iter(y_iter),
					group_size, x_copy->dimension[2], x_copy->dimension[3], w->dimension[0], w->dimension[2], w->dimension[3], y->dimension[2], y->dimension[3], stride, dilations);
			}
		}
	}
	else {
		printf("Currently only supports Conv1D and Conv2D");
		system("pause");
		error = OPS_UNDEFINED;
	}

cleanup:
	safe_free(&x_coordinate);
	safe_free(&y_coordinate);
	safe_free(&new_pad);
	release_tensor_iterator(&x_iter);
	release_tensor_iterator(&y_iter);
	release_tensor(&x_copy);
	return error;
}

int transposef_2d(float* x, int64_t  n, int64_t m) {
	int i = 0, j = 0; float temp = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			temp = x[i * m + j];
			x[i * m + j] = x[j * n + i];
			x[j * n + i] = temp;
		}
	}
	return 1;
}



int conv1df(float* x, float* w, float* b, float* y, int64_t C, int64_t F, int64_t M, int64_t K, int64_t NF, int64_t stride, int64_t dilation) {
	int64_t c = 0, f = 0, m = 0, k = 0, nf = 0, dilate = 0; float sum = 0;
	if (x == NULL || w == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (dilation == 1)dilate = K - 1;
	else {
		dilate = K + (K - 1) * (dilation - 1) - 1;
	}
	for (f = 0; f + dilate < F; f += stride) {
		for (m = 0; m < M; m++) {
			sum = 0.f;
			if (b != NULL) {
				sum += b[m];
			}
			for (k = 0; k < K; k++) {
				for (c = 0; c < C; c++) {
					sum += x[c * F + f + dilation * k]
						* w[m * C * K + c * K + k];
				}
			}
			y[m * NF + f / stride] = sum;

		}
	}
	return OPS_SUCCESS;
}


int conv2df(float* input, float* kernels, float* bias, float* output, int64_t C, int64_t H, int64_t W, int64_t M, int64_t kH, int64_t kW, int64_t nH, int64_t nW, int64_t* stride, int64_t* dilation) {
	int64_t c = 0, h = 0, w = 0, m = 0, kh = 0, kw = 0, dilateH = 0, dilateW = 0; float sum = 0;
	if (input == NULL || kernels == NULL || output == NULL) return OPS_INPUT_IS_NULL;
	// Calculate last index position
	if (dilation[0] == 1)dilateH = kH - 1;
	else {
		dilateH = kH + (kH - 1) * (dilation[0] - 1) - 1;
	}
	if (dilation[1] == 1)dilateW = kW - 1;
	else {
		dilateW = kW + (kW - 1) * (dilation[1] - 1) - 1;
	}
	for (h = 0; h + dilateH < H; h += stride[0]) {
		for (w = 0; w + dilateW < W; w += stride[1]) {
			for (m = 0; m < M; m++) {
				sum = 0.f;	
				if (bias != NULL) {
					sum += bias[m];

				}
				for (kh = 0; kh < kH; kh++) {
					for (kw = 0; kw < kW; kw++) {
						for (c = 0; c < C; c++) {
							sum += input[c * H * W + (h + kh * dilation[0]) * W + w + kw *dilation[1]]
								* kernels[m * C* kH * kW + c* kH*kW + kh*kW+kw];
						}
					}
				}
				output[m * nH * nW + h / stride[0] * nW + w / stride[1]] = sum;
			}
		}
	}
	return OPS_SUCCESS;
}
int addf_array(float* a, float* b, float* c, int64_t a_size, int64_t b_size, int64_t c_size) {
	int64_t i = 0;
	for (int i = 0; i < c_size; i++) {
		c[i] = a[i % a_size] + b[i % b_size];
	}
	return OPS_SUCCESS;
}

int mulf_array(float* a, float* b, float* c, int64_t a_size, int64_t b_size, int64_t c_size) {
	int64_t i = 0;
	for (int i = 0; i < c_size; i++) {
		c[i] = a[i % a_size] * b[i % b_size];
	}
	return OPS_SUCCESS;
}

int tanhf_array(float* input, float* output, int64_t size) {
	int64_t i = 0;
	if (input == NULL || output == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	for (i = 0; i < size; i++) {

		output[i] = tanhf(input[i]);
	}
	return OPS_SUCCESS;
}

int sigmoidf_array(float* input, float* output, int64_t size) {
	int64_t i = 0;
	if (input == NULL || output == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	for (i = 0; i < size; i++) {

		output[i] = 1.f / (1.0f + expf(-input[i]));
	}
	return OPS_SUCCESS;
}

int reluf_array(float* input, float* output, int64_t size) {
	int64_t i = 0;
	if (input == NULL || output == NULL) {
		return OPS_INPUT_IS_NULL;
	}
	for (i = 0; i < size; i++) {

		output[i] = max(0.0f, input[i]);
	}
	return OPS_SUCCESS;
}

int activationf_array(float* input, float* output, char* type, int64_t size, float alpha, float beta) {
	if (input == NULL || output == NULL || type == NULL) return OPS_INPUT_IS_NULL;
	if (strcmp(type, "Relu") == 0) return reluf_array(input, output, size);
	else if (strcmp(type, "Sigmoid") == 0) return sigmoidf_array(input, output, size);
	else if (strcmp(type, "Tanh") == 0)return tanhf_array(input, output, size);
	else {
		return OPS_UNDEFINED;
	}
}


int lstm_function(float* activation_alpha, float* activation_beta, struct list* activations, float clip, char* direction, int64_t hidden_size, int64_t input_forget, int64_t layout,
	struct tensor* x, struct tensor* w, struct tensor* r, struct tensor* b, int64_t* seq_length, struct tensor* initial_h, struct tensor* initial_c, struct tensor* P, struct tensor* Y, struct tensor* Y_h, struct tensor* Y_c) {
	int error = 0;
	// Creating iterators
	struct tensor_iterator* x_iter = create_tensor_iterator(x);
	if (x_iter == NULL) return OPS_ALLOCATION_FAIL;
	struct tensor_iterator* w_iter = create_tensor_iterator(w);
	if (w_iter == NULL) return OPS_ALLOCATION_FAIL;
	struct tensor_iterator* r_iter = create_tensor_iterator(r);
	if (r_iter == NULL) return OPS_ALLOCATION_FAIL;
	struct tensor_iterator* b_iter = create_tensor_iterator(b);
	if (b_iter == NULL) return OPS_ALLOCATION_FAIL;
	struct tensor_iterator* yh_iter = create_tensor_iterator(Y_h);
	if (yh_iter == NULL) return OPS_ALLOCATION_FAIL;
	struct tensor_iterator* yc_iter = create_tensor_iterator(Y_c);
	if (yc_iter == NULL) return OPS_ALLOCATION_FAIL;
	struct tensor_iterator* p_iter = create_tensor_iterator(P);
	if (p_iter == NULL) return OPS_ALLOCATION_FAIL;
	struct tensor_iterator* y_iter = create_tensor_iterator(Y);
	if (y_iter == NULL) return OPS_ALLOCATION_FAIL;

	float* it = NULL, * ft = NULL, * ct = NULL;	// temporary gate results
	float* XW = NULL, * HR = NULL, * PC = NULL;
	int64_t i = 0, j = 0, k = 0, l = 0, m = 0,
		batch_size = 0, num_direction = 0, sequence_length = 0, input_size = 0,
		* x_coordinate = NULL, * wr_coordinate = NULL, * b_coordinate = NULL, * value_coordinate = NULL,
		* p_coordinate = NULL, * output_coordinate = NULL;

	if (hidden_size == 0) hidden_size = w->dimension[1] / 4;
	batch_size = x->dimension[1];
	input_size = x->dimension[2];


	if (seq_length != NULL) {
		for (i = 0; i < batch_size; i++) {
			if (sequence_length < seq_length[i])sequence_length = seq_length[i];
		}
	}

	else sequence_length = x->dimension[0];
	//puting initial data into outputs, this way every iteration can read from y_h to calculate it's state
	memcpy(Y_h->data, initial_h->data, Y_h->data_size);
	memcpy(Y_c->data, initial_c->data, Y_c->data_size);
	//


	it = malloc(hidden_size * sizeof(float));
	ft = malloc(hidden_size * sizeof(float));
	ct = malloc(hidden_size * sizeof(float));

	// Allcoating memory to hold coordinates(index in nd space) and temporary storage for matmuls
	XW = malloc(hidden_size * sizeof(float));	// temp storage for input and transpose of weight
	HR = malloc(hidden_size * sizeof(float));	// temp storage for weight and transpose of recurrence
	PC = malloc(hidden_size * sizeof(float));	// temp storage for pipe and cell
	x_coordinate = calloc(3, sizeof(int64_t));	// Input coordinate
	wr_coordinate = calloc(3, sizeof(int64_t));		// weight and recurrence_weight share
	b_coordinate = calloc(2, sizeof(int64_t));		// bias
	value_coordinate = calloc(3, sizeof(int64_t));	// initial_h, initial_c, y_h, y_c share
	p_coordinate = calloc(2, sizeof(int64_t));		// pipe
	output_coordinate = calloc(4, sizeof(int64_t));	// output y
	if (XW == NULL || HR == NULL || PC == NULL || x_coordinate == NULL || wr_coordinate == NULL || b_coordinate == NULL ||
		value_coordinate == NULL || p_coordinate == NULL || output_coordinate == NULL) {
		return OPS_ALLOCATION_FAIL;
	}
	/*
	it = f(Xt * (Wi ^ T) + Ht - 1 * (Ri ^ T) + Pi(.) Ct - 1 + Wbi + Rbi)
	ft = f(Xt * (Wf ^ T) + Ht - 1 * (Rf ^ T) + Pf(.) Ct - 1 + Wbf + Rbf)
	ct = g(Xt * (Wc ^ T) + Ht - 1 * (Rc ^ T) + Wbc + Rbc)
	Ct = ft(.) Ct - 1 + it(.) ct
	ot = f(Xt * (Wo ^ T) + Ht - 1 * (Ro ^ T) + Po(.) Ct + Wbo + Rbo)
	Ht = ot(.) h(Ct)
	*/

	for (i = 0; i < batch_size; i++) {
		j = 0;
		if (strcmp(direction, "forward") == 0 || strcmp(direction, "bidirectional") == 0) {
			if (seq_length != NULL) {
				sequence_length = seq_length[i];
			}
			for (k = 0; k < sequence_length; k++) {
				// it = f(Xt * (Wi ^ T) + Ht - 1 * (Ri ^ T) + Pi(.) Ct - 1 + Wbi + Rbi)
				x_coordinate[0] = k; x_coordinate[1] = i;	// x coordinate
				wr_coordinate[0] = j;	wr_coordinate[1] = 0;
				b_coordinate[0] = j; b_coordinate[1] = 0;
				value_coordinate[0] = j; value_coordinate[1] = i;
				p_coordinate[0] = 0; p_coordinate[1] = 0;
				output_coordinate[0] = k; output_coordinate[1] = j; output_coordinate[2] = i;
				goto_tensor_iter(x_iter, x_coordinate);
				goto_tensor_iter(w_iter, wr_coordinate);
				goto_tensor_iter(r_iter, wr_coordinate);
				goto_tensor_iter(b_iter, b_coordinate);
				goto_tensor_iter(p_iter, p_coordinate);
				goto_tensor_iter(yh_iter, value_coordinate);
				goto_tensor_iter(yc_iter, value_coordinate);
				goto_tensor_iter(y_iter, output_coordinate);

				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(x_iter), 1, (float*)get_data_tensor_iter(w_iter), input_size, 1.0, XW, 1);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(yh_iter), 1, (float*)get_data_tensor_iter(r_iter), input_size, 1.0, HR, 1);
				mulf_array(get_data_tensor_iter(p_iter), get_data_tensor_iter(yc_iter), PC, hidden_size, hidden_size, hidden_size);
				memcpy_s(it, hidden_size * sizeof(int64_t), XW, hidden_size * sizeof(int64_t));
				addf_array(XW, HR, it, hidden_size, hidden_size, hidden_size);
				addf_array(it, PC, it, hidden_size, hidden_size, hidden_size);
				addf_array(it, get_data_tensor_iter(b_iter), it, hidden_size, hidden_size, hidden_size);
				b_coordinate[1] = 4 * hidden_size;
				goto_tensor_iter(b_iter, b_coordinate);
				addf_array(it, get_data_tensor_iter(b_iter), it, hidden_size, hidden_size, hidden_size);
				if (activation_alpha != NULL) {
					error = activationf_array(it, it, get_list(activations, j * 3), hidden_size, activation_alpha[j * 3], activation_beta[j * 3]);
				}
				else {
					error = activationf_array(it, it, get_list(activations, j * 3), hidden_size, 1.0f, 1.0f);
				}
				if (error != OPS_SUCCESS) return error;

				//ft = f(Xt * (Wf ^ T) + Ht - 1 * (Rf ^ T) + Pf(.) Ct - 1 + Wbf + Rbf)
				wr_coordinate[1] = 2 * hidden_size;;
				b_coordinate[1] = 2 * hidden_size;
				p_coordinate[1] = 2 * hidden_size;
				goto_tensor_iter(w_iter, wr_coordinate);
				goto_tensor_iter(r_iter, wr_coordinate);
				goto_tensor_iter(b_iter, b_coordinate);
				goto_tensor_iter(p_iter, p_coordinate);

				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(x_iter), 1, (float*)get_data_tensor_iter(w_iter), input_size, 1.0, XW, 1);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(yh_iter), 1, (float*)get_data_tensor_iter(r_iter), input_size, 1.0, HR, 1);
				mulf_array(get_data_tensor_iter(p_iter), get_data_tensor_iter(yc_iter), PC, hidden_size, hidden_size, hidden_size);
				memcpy_s(ft, hidden_size * sizeof(int64_t), XW, hidden_size * sizeof(int64_t));
				addf_array(XW, HR, ft, hidden_size, hidden_size, hidden_size);
				addf_array(ft, PC, ft, hidden_size, hidden_size, hidden_size);
				addf_array(ft, get_data_tensor_iter(b_iter), ft, hidden_size, hidden_size, hidden_size);
				b_coordinate[1] = (4 + 2) * hidden_size;
				goto_tensor_iter(b_iter, b_coordinate);
				addf_array(ft, get_data_tensor_iter(b_iter), ft, hidden_size, hidden_size, hidden_size);
				if (activation_alpha != NULL) {
					error = activationf_array(it, it, get_list(activations, j * 3), hidden_size, activation_alpha[j * 3], activation_beta[j * 3]);
				}
				else {
					error = activationf_array(it, it, get_list(activations, j * 3), hidden_size, 1.0f, 1.0f);
				}
				if (error != OPS_SUCCESS) return error;

				//ct = g(Xt * (Wc ^ T) + Ht - 1 * (Rc ^ T) + Wbc + Rbc)
				wr_coordinate[1] = 3 * hidden_size;
				b_coordinate[1] = 3 * hidden_size;
				goto_tensor_iter(w_iter, wr_coordinate);
				goto_tensor_iter(r_iter, wr_coordinate);
				goto_tensor_iter(b_iter, b_coordinate);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(x_iter), 1, (float*)get_data_tensor_iter(w_iter), input_size, 1.0, XW, 1);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(yh_iter), 1, (float*)get_data_tensor_iter(r_iter), input_size, 1.0, HR, 1);
				memcpy_s(ct, hidden_size * sizeof(int64_t), XW, hidden_size * sizeof(int64_t));
				addf_array(XW, HR, ct, hidden_size, hidden_size, hidden_size);
				addf_array(ct, get_data_tensor_iter(b_iter), ct, hidden_size, hidden_size, hidden_size);
				b_coordinate[1] = (4 + 3) * hidden_size;
				goto_tensor_iter(b_iter, b_coordinate);
				addf_array(ct, get_data_tensor_iter(b_iter), ct, hidden_size, hidden_size, hidden_size);
				if (activation_alpha != NULL) {
					error = activationf_array(it, it, get_list(activations, j * 3 + 1), hidden_size, activation_alpha[j * 3 + 1], activation_beta[j * 3 + 1]);
				}
				else {
					error = activationf_array(it, it, get_list(activations, j * 3 + 1), hidden_size, 1.0f, 1.0f);
				}
				if (error != OPS_SUCCESS) return error;
				//Ct = ft(.) Ct - 1 + it(.) ct
				mulf_array(ft, get_data_tensor_iter(yc_iter), ft, hidden_size, hidden_size, hidden_size);
				mulf_array(it, ct, it, hidden_size, hidden_size, hidden_size);
				addf_array(it, ft, get_data_tensor_iter(yc_iter), hidden_size, hidden_size, hidden_size);
				// 	ot = f(Xt * (Wo ^ T) + Ht - 1 * (Ro ^ T) + Po(.) Ct + Wbo + Rbo)
				wr_coordinate[1] = 1 * hidden_size;;
				b_coordinate[1] = 1 * hidden_size;
				p_coordinate[1] = 1 * hidden_size;
				goto_tensor_iter(w_iter, wr_coordinate);
				goto_tensor_iter(r_iter, wr_coordinate);
				goto_tensor_iter(b_iter, b_coordinate);
				goto_tensor_iter(p_iter, p_coordinate);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(x_iter), 1, (float*)get_data_tensor_iter(w_iter), input_size, 1.0, XW, 1);
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden_size, input_size, 1.0f, (float*)get_data_tensor_iter(yh_iter), 1, (float*)get_data_tensor_iter(r_iter), input_size, 1.0, HR, 1);
				mulf_array(get_data_tensor_iter(p_iter), get_data_tensor_iter(yc_iter), PC, hidden_size, hidden_size, hidden_size);
				memcpy_s(get_data_tensor_iter(y_iter), hidden_size * sizeof(int64_t), XW, hidden_size * sizeof(int64_t));
				addf_array(XW, HR, get_data_tensor_iter(y_iter), hidden_size, hidden_size, hidden_size);
				addf_array(get_data_tensor_iter(y_iter), PC, get_data_tensor_iter(y_iter), hidden_size, hidden_size, hidden_size);
				addf_array(get_data_tensor_iter(y_iter), get_data_tensor_iter(b_iter), get_data_tensor_iter(y_iter), hidden_size, hidden_size, hidden_size);
				b_coordinate[1] = (4 + 1) * hidden_size;
				goto_tensor_iter(b_iter, b_coordinate);
				addf_array(get_data_tensor_iter(y_iter), get_data_tensor_iter(b_iter), get_data_tensor_iter(y_iter), hidden_size, hidden_size, hidden_size);
				if (activation_alpha != NULL) {
					error = activationf_array(it, it, get_list(activations, j * 3), hidden_size, activation_alpha[j * 3], activation_beta[j * 3]);
				}
				else {
					error = activationf_array(it, it, get_list(activations, j * 3), hidden_size, 1.0f, 1.0f);
				}
				if (error != OPS_SUCCESS) return error;
				//Ht = ot(.) h(Ct)
				if (activation_alpha != NULL) {
					error = activationf_array(it, it, get_list(activations, j * 3 + 2), hidden_size, activation_alpha[j * 3 + 2], activation_beta[j * 3 + 2]);
				}
				else {
					error = activationf_array(it, it, get_list(activations, j * 3 + 2), hidden_size, 1.0f, 1.0f);
				}
				if (error != OPS_SUCCESS) return error;
				mulf_array(get_data_tensor_iter(yh_iter), get_data_tensor_iter(y_iter), get_data_tensor_iter(yh_iter), hidden_size, hidden_size, hidden_size);
			}
			j++;

		}
		if (strcmp(direction, "bidirectional") == 0 || strcmp(direction, "reverse")) {

			printf("note yet implemented reverse abd vidirectional\n");
			return OPS_TYPE_UNIMPLEMENTED;
		}

	}
	free(it); free(ct); free(ft);
	free(XW); free(HR); free(PC);
	free(x_coordinate); free(wr_coordinate); free(b_coordinate); free(value_coordinate); free(p_coordinate); free(output_coordinate);
	release_tensor_iterator(&x_iter);
	release_tensor_iterator(&w_iter);
	release_tensor_iterator(&r_iter);
	release_tensor_iterator(&b_iter);
	release_tensor_iterator(&p_iter);
	release_tensor_iterator(&yh_iter);
	release_tensor_iterator(&yc_iter);
	release_tensor_iterator(&y_iter);
	return OPS_SUCCESS;
}

int concat_function(int64_t* axis, struct list* inputs, struct tensor* concat_result) {
	int error = 0;
	int64_t i = 0, j = 0, axisoffset = 0, *target = NULL;
	struct tensor* a = NULL, * b = NULL;
	struct tensor_iterator *input_iter = NULL;
	struct tensor_iterator* result_iter = NULL;
	if (axis == NULL || inputs == NULL || concat_result == NULL) return OPS_INPUT_IS_NULL;
	a = (struct tensor*)get_list(inputs, 0);
	// if negative axis
	if (*axis < 0) *axis += a->dimension_size;
	// Check dimension match
	for (i = 1; i < inputs->size; i++) {
		b = (struct tensor*)get_list(inputs, i);
		if (a->dimension_size != b->dimension_size) {
			return OPS_DIMENSION_MISMATCH;
		}
		for (j = 0; j < a->dimension_size; j++) {
			if (j == *axis) {
				continue;
			}
			if (a->dimension[j] != b->dimension[j]) {
				return OPS_DIMENSION_MISMATCH;
			}
		}
	}
	target = calloc(concat_result->dimension_size, sizeof(int64_t));
	if (target == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	result_iter = create_tensor_iterator(concat_result);
	if (result_iter == NULL) {
		error = OPS_ALLOCATION_FAIL;
		goto cleanup;
	}
	for (i = 0; i < inputs->size; i++) {
		input_iter = create_tensor_iterator(get_list(inputs,i));
		if (input_iter == NULL) {
			error = OPS_ALLOCATION_FAIL;
			goto cleanup;
		}
		memcpy(target, input_iter->coordinate, input_iter->dimension_size * sizeof(int64_t));
		target[*axis] += axisoffset;
		goto_tensor_iter(result_iter, target);
		while (1) {
			memcpy(get_data_tensor_iter(result_iter), get_data_tensor_iter(input_iter), concat_result->item_size);
			if (!is_not_done_tensor_iter(input_iter)) break;
			next_tensor_iter(input_iter);
			memcpy(target, input_iter->coordinate, input_iter->dimension_size * sizeof(int64_t));
			target[*axis] += axisoffset;
			goto_tensor_iter(result_iter, target);
		}
		axisoffset += a->dimension[*axis];
		release_tensor_iterator(&input_iter);
	}
	error = OPS_SUCCESS;
cleanup:
	safe_free(&target);
	release_tensor_iterator(&result_iter);
	release_tensor_iterator(&input_iter);
	return error;

}

int split_function(int64_t axis, int64_t num_outputs, struct tensor* input, int64_t* split, struct list* outputs) {
	int64_t i = 0, j = 0, axisoffset = 0, interval = 1;
	struct tensor* temp = NULL;
	struct tensor_iterator* input_iter = NULL;
	if (input == NULL || outputs == NULL) return OPS_INPUT_IS_NULL;
	if (axis < 0) axis += input->dimension_size;

	if (split == NULL) {
		for (j = 0; j < input->dimension_size; j++) {
			if (j == axis)interval *= (input->dimension[axis] / num_outputs);
			else interval *= input->dimension[j];
		}
		for (i = 0; i < num_outputs; i++) {
			temp = (struct tensor*)get_list(outputs, i);
			if (input->data_size - axisoffset < interval) {
				memcpy((char*)temp->data, (char*)input->data + axisoffset * input->item_size, (input->data_size - axisoffset) * input->item_size);
			}
			else {
				memcpy((char*)temp->data, (char*)input->data + axisoffset * input->item_size, interval * input->item_size);
				axisoffset += interval;
			}
		}
	}
	else {
		input_iter = create_tensor_iterator(input);
		interval = 1;
		for (i = 0; i < input->dimension_size; i++) {
			if (i != axis) {
				interval *= input->dimension[i];
			}
		}
		for (i = 0; i < num_outputs; i++) {
			temp = (struct tensor*)get_list(outputs, i);
			memcpy((char*)temp->data, (char*)input->data + axisoffset * input->item_size, split[i] * interval * input->item_size);
			axisoffset += split[i] * interval;
		}
	}
	release_tensor_iterator(&input_iter);
	return OPS_SUCCESS;
}


int reducemean_function(int64_t keepdims, int64_t noop_with_empty_axes, struct tensor* data, struct list* axes, struct tensor* reduced) {
	int64_t reduce_offset = 1, i = 0, j = 0, k = 0;
	struct tensor_iterator* data_iter = NULL, * reduced_iter = NULL;
	if (data == NULL || reduced == NULL) return OPS_INPUT_IS_NULL;

	if (axes == NULL) {
		if (!noop_with_empty_axes) {
			for (i = 0; i < data->dimension_size; i++) {
				reduce_offset *= data->dimension[i];
			}
		}
		else {
			memcpy(reduced->data, data->data, data->data_size * data->item_size);
			return OPS_SUCCESS;
		}
	}
	else {
		for (i = 0; i < axes->size; i++) {
			if (*(int64_t*)get_list(axes, i) < 0) {
				*(int64_t*)get_list(axes, i) += data->dimension_size;
			}
			reduce_offset *= data->dimension[*(int64_t*)get_list(axes, i)];
		}
	}
	data_iter = create_tensor_iterator(data);
	reduced_iter = create_tensor_iterator(reduced);
	if (data_iter == NULL || reduced_iter == NULL) return OPS_ALLOCATION_FAIL;
	i = 0; j = 0;
	while (1) {
		mean_array(get_data_tensor_iter(data_iter), get_data_tensor_iter(reduced_iter), reduce_offset, data->type);
		i += reduce_offset;
		goto_1d_tensor_iter(data_iter, i);
		next_tensor_iter(reduced_iter);
		if (!is_not_done_tensor_iter(reduced_iter)) break;

	}
	release_tensor_iterator(&data_iter);
	release_tensor_iterator(&reduced_iter);
	return OPS_SUCCESS;
}