#include "ops.h"


int add_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type) {
	if (a == NULL || b == NULL || c == NULL)return OPS_INPUT_IS_NULL;
	int64_t i = 0;
	if (type == DATATYPE_INT32)
		for (i = 0; i < c_size; i++) {
			((int32_t*)c)[i] = ((int32_t*)a)[i % a_size] + ((int32_t*)b)[i % b_size];
		}
	if (type == DATATYPE_INT64)
		for (i = 0; i < c_size; i++) {
			((int64_t*)c)[i] = ((int64_t*)a)[i % a_size] + ((int64_t*)b)[i % b_size];
		}
	if (type == DATATYPE_FLOAT)
		for (i = 0; i < c_size; i++) {
			((float*)c)[i] = ((float*)a)[i % a_size] + ((float*)b)[i % b_size];
		}
	if (type == DATATYPE_DOUBLE)
		for (i = 0; i < c_size; i++) {
			((double*)c)[i] = ((double*)a)[i % a_size] + ((double*)b)[i % b_size];
		}
	else return OPS_TYPE_NOT_SUPPORTED;

	return OPS_SUCCESS;
}
int sub_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type) {
	if (a == NULL || b == NULL || c == NULL)return OPS_INPUT_IS_NULL;
	int64_t i = 0;
	if (type == DATATYPE_INT32)
		for (i = 0; i < c_size; i++) {
			((int32_t*)c)[i] = ((int32_t*)a)[i % a_size] - ((int32_t*)b)[i % b_size];
		}
	if (type == DATATYPE_INT64)
		for (i = 0; i < c_size; i++) {
			((int64_t*)c)[i] = ((int64_t*)a)[i % a_size] - ((int64_t*)b)[i % b_size];
		}
	if (type == DATATYPE_FLOAT)
		for (i = 0; i < c_size; i++) {
			((float*)c)[i] = ((float*)a)[i % a_size] - ((float*)b)[i % b_size];
		}
	else return OPS_TYPE_NOT_SUPPORTED;
	return OPS_SUCCESS;
}
int mul_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type) {
	if (a == NULL || b == NULL || c == NULL)return OPS_INPUT_IS_NULL;
	int64_t i = 0;
	if (type == DATATYPE_INT32)
		for (i = 0; i < c_size; i++) {
			((int32_t*)c)[i] = ((int32_t*)a)[i % a_size] * ((int32_t*)b)[i % b_size];
		}
	if (type == DATATYPE_INT64)
		for (i = 0; i < c_size; i++) {
			((int64_t*)c)[i] = ((int64_t*)a)[i % a_size] * ((int64_t*)b)[i % b_size];
		}
	if (type == DATATYPE_FLOAT)
		for (i = 0; i < c_size; i++) {
			((float*)c)[i] = ((float*)a)[i % a_size] * ((float*)b)[i % b_size];
		}
	else return OPS_TYPE_NOT_SUPPORTED;

	return OPS_SUCCESS;
}
int div_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type) {
	if (a == NULL || b == NULL || c == NULL)return OPS_INPUT_IS_NULL;
	int64_t i = 0;
	if (type == DATATYPE_INT32)
		for (i = 0; i < c_size; i++) {
			((int32_t*)c)[i] = ((int32_t*)a)[i % a_size] / ((int32_t*)b)[i % b_size];
		}
	if (type == DATATYPE_INT64)
		for (i = 0; i < c_size; i++) {
			((int64_t*)c)[i] = ((int64_t*)a)[i % a_size] / ((int64_t*)b)[i % b_size];
		}
	if (type == DATATYPE_FLOAT)
		for (i = 0; i < c_size; i++) {
			((float*)c)[i] = ((float*)a)[i % a_size] / ((float*)b)[i % b_size];
		}
	else return OPS_TYPE_NOT_SUPPORTED;

	return OPS_SUCCESS;
}


int matmul_array(const void* A, const void* B, void* C, int64_t n, int64_t m, int64_t p, const DATATYPE type) {
	int64_t i = 0, j = 0, k = 0;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < p; j++) {
				((float*)C)[i * p + j] = 0;
				for (k = 0; k < m; k++) {
					((float*)C)[i * p + j] += ((float*)A)[i * m + k] * ((float*)B)[k * p + j];
				}
			}
		}
	}
	else if (type == DATATYPE_FLOAT) {
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
int mean_array(void* array, void* result, int64_t num_elements, const DATATYPE type) {
	int64_t i = 0;
	if (array == NULL || result == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
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

int transpose_2d(const void* input, void* output, int64_t n, int64_t m, const DATATYPE type) {
	int64_t i = 0, j = 0;
	if (input == NULL || output == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_INT32) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < m; j++) {
				((int32_t*)output)[i * m + j] = ((int32_t*)input)[j * n + i];
			}
		}
	}
	if (type == DATATYPE_INT64) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < m; j++) {
				((int64_t*)output)[i * m + j] = ((int64_t*)input)[j * n + i];
			}
		}
	}
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < m; j++) {
				((float*)output)[i * m + j] = ((float*)input)[j * n + i];
			}
		}
	}
	else {
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;

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
							sum += input[c * H * W + (h + kh * dilation[0]) * W + w + kw * dilation[1]]
								* kernels[m * C * kH * kW + c * kH * kW + kh * kW + kw];
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


int tanh_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = tanhf(((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}

int atan_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = atanf(((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}

int atanh_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = atanhf(((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}

int sigmoid_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = 1.f / (1.0f + expf(((float*)x)[i]));
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}
int relu_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = max(0, ((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}

int tanhf_array(float* input, float* output, int64_t size) {
	//printf("tanh\n");
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
	//printf("sigmoid\n");
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
	//printf("Relu\n");
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
	if (input == NULL || output == NULL || type == NULL) {
		printf("activation input is NULL!\n");
		return OPS_INPUT_IS_NULL;
	}
	if (strcmp(type, "Relu") == 0) return reluf_array(input, output, size);
	else if (strcmp(type, "Sigmoid") == 0) {
		//printf("sigmoid\n");
		return sigmoidf_array(input, output, size);
	}
	else if (strcmp(type, "Tanh") == 0) {
		//printf("tanh\n");
		return tanhf_array(input, output, size);
	}
	else {
		printf("Undefined activalion\n");
		//system("puase\n");
		return OPS_UNDEFINED;
	}
	return OPS_SUCCESS;
}

int activation_array(const void* input, void* output, char* activation, int64_t size, float* alpha, float* beta, const DATATYPE type) {
	if (input == NULL || output == NULL || type == NULL) {
		printf("activation input is NULL!\n");
		return OPS_INPUT_IS_NULL;
	}
	if (strcmp(type, "Relu") == 0) return relu_array(input, output, size, type);
	else if (strcmp(type, "Sigmoid") == 0) return sigmoid_array(input, output, size,type);
	else if (strcmp(type, "Tanh") == 0) return tanh_array(input, output, size,type);
	else {
		printf("Undefined activalion\n");
		//system("puase\n");
		return OPS_UNDEFINED;
	}
}

int clip_array(const void* input, const void* min, const void* max, void* output, const int64_t size, const DATATYPE type) {
	double double_max = DBL_MAX, double_min = DBL_MIN;
	float float_max = FLT_MAX, float_min = FLT_MIN;
	int32_t int32_max = INT32_MAX, int32_min = INT32_MIN;
	int64_t int64_max = INT64_MAX, int64_min = INT64_MIN;
	int i = 0;
	if (input == NULL || output == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_INT32) {
		if (min != NULL) if (*((int32_t*)min) != 0) int32_min = *((int32_t*)min);
		if (max != NULL) if (*((int32_t*)max) != 0) int32_max = *((int32_t*)max);
		for (i = 0; i < size; i++) {
			if (((int32_t*)input)[i] > int32_max) ((int32_t*)output)[i] = int32_max;
			if (((int32_t*)input)[i] < int32_min) ((int32_t*)output)[i] = int32_min;
			else {
				((int32_t*)output)[i] = ((int32_t*)input)[i];
			}
		}
	}
	else if (type == DATATYPE_INT64) {
		if (min != NULL) if (*((int64_t*)min) != 0) int64_min = *((int64_t*)min);
		if (max != NULL) if (*((int64_t*)max) != 0) int64_max = *((int64_t*)max);
		for (i = 0; i < size; i++) {
			if (((int64_t*)input)[i] > int64_max) ((int64_t*)output)[i] = int64_max;
			if (((int64_t*)input)[i] < int64_min) ((int64_t*)output)[i] = int64_min;
			else {
				((int64_t*)output)[i] = ((int64_t*)input)[i];
			}
		}
	}
	else if (type == DATATYPE_FLOAT) {
		if (min != NULL) if (*((float*)min) != 0) float_min = *((float*)min);
		if (max != NULL) if (*((float*)max) != 0) float_max = *((float*)max);
		for (i = 0; i < size; i++) {
			if (((float*)input)[i] > float_max) ((float*)output)[i] = float_max;
			else if (((float*)input)[i] < float_min) ((float*)output)[i] = float_min;
			else {
				((float*)output)[i] = ((float*)input)[i];
			}
		}
	}
	else if (type == DATATYPE_DOUBLE) {
		if (min != NULL) if (*((double*)min) != 0) double_min = *((double*)min);
		if (max != NULL) if (*((double*)max) != 0) double_max = *((double*)max);
		for (i = 0; i < size; i++) {
			if (((double*)input)[i] > double_max) ((double*)output)[i] = double_max;
			else if (((double*)input)[i] < double_min) ((double*)output)[i] = double_min;
			else {
				((double*)output)[i] = ((double*)input)[i];
			}
		}
	}
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}

int abs_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_INT32) {
		for (i = 0; i < size; i++) {
			((int32_t*)y)[i] = labs(((int32_t*)x)[i]);
		}
	}
	else if (type == DATATYPE_INT64) {
		for (i = 0; i < size; i++) {
			((int64_t*)y)[i] = llabs(((int64_t*)x)[i]);
		}
	}
	else if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = fabsf(((float*)x)[i]);
		}
	}
	else if (type == DATATYPE_DOUBLE) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = fabsf(((float*)x)[i]);
		}
	}
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}


int acos_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = acosf(((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}

int acosh_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = acoshf(((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}


int asin_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = asinf(((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}

int asinh_array(const void* x, void* y, const int64_t size, const DATATYPE type) {
	int error = 0;
	int64_t i = 0;
	if (x == NULL || y == NULL) return OPS_INPUT_IS_NULL;
	if (type == DATATYPE_FLOAT) {
		for (i = 0; i < size; i++) {
			((float*)y)[i] = asinhf(((float*)x)[i]);
		}
	}		// add in double type maaybe
	else {	// Unknown type
		return OPS_TYPE_NOT_SUPPORTED;
	}
	return OPS_SUCCESS;
}


void xgemm(const int transA, const int transB, const int64_t m, const int64_t n, const int64_t k, const float alpha, void* a, const int64_t lda, void* b, const int64_t ldb, const float beta, void* c, const int64_t ldc, const DATATYPE type) {
	char* a_copy = NULL, * b_copy = NULL, * c_temp = NULL;
	int error = 0;
	if (a == NULL || b == NULL || c == NULL) {
		error = OPS_INPUT_IS_NULL;
		goto cleanup;
	}
	a_copy = malloc(k * m * datatype_size(type));
	b_copy = malloc(k * n * datatype_size(type));
	c_temp = malloc(ldc * n * datatype_size(type));
	if (transA) {

		transpose_2d(a, a_copy, k, m, type);
	}

	if (transB) {
		transpose_2d(b, b_copy, n, k, type);
	}






cleanup:
	if (error == OPS_ALLOCATION_FAIL) {
		printf("GEMM allocate memory fail\n");
	}
	safe_free(&a_copy);



}