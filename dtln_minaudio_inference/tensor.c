#include "tensor.h"


struct tensor* create_tensor( void* data, int64_t num_elements, int64_t* dimension, int64_t num_dimension, short DataType, short is_static) {
	struct tensor* newTensor = (struct tensor*)calloc(1, sizeof(struct tensor));
	int64_t* Stride = NULL, i = 0;
	if (newTensor != NULL) {
		newTensor->is_size_unknown = false;
		// Set datatype
		if (DataType == DATATYPE_FLOAT || DataType == DATATYPE_INT32) {
			newTensor->item_size = 4;
		}
		else if (DataType == DATATYPE_INT64) {
			newTensor->item_size = 8;
		}
		else {
			return NULL;
		}
		// Assign input
		if (data != 0) {
			newTensor->data = data;
		}
		else {

#ifdef ONE_MKL
			newTensor->data = mkl_malloc(num_elements * newTensor->item_size, 64);
			memset(newTensor->data, 0 , num_elements * newTensor->item_size);
#else
			newTensor->data = calloc(num_elements, newTensor->item_size);
#endif

		}
		newTensor->data_size = num_elements;
		if (dimension == NULL) return NULL;
		newTensor->dimension = dimension;
		newTensor->dimension_size = num_dimension;
		newTensor->type = DataType;
		newTensor->is_static = is_static;

		// Calculate stride
		Stride = (int64_t*)malloc(num_dimension * sizeof(int64_t));
		if (Stride == NULL) return 0;
		Stride[num_dimension - 1] = newTensor->item_size;
		for (i = num_dimension - 2; i >= 0; i--) {
			Stride[i] = Stride[i + 1] * dimension[i + 1];
		}
		newTensor->stride = Stride;

	}
	else {
		perror("Create tensor fail: ");
		system("pause");
		return NULL;
	}
	return newTensor;
}

struct tensor* create_empty_tensor() {
	struct tensor* newTensor = (struct tensor*)calloc(1, sizeof(struct tensor));
	if (newTensor == NULL) return NULL;
	newTensor->is_size_unknown = 1;
	newTensor->is_static = false;
	return newTensor;
}

struct tensor* create_tensor_copy(struct tensor* t) {
	if (t == NULL) return NULL;
	struct tensor* newTensor = (struct tensor*)calloc(1, 
		sizeof(struct tensor));
	if (newTensor == NULL) {
		perror("Create tensor copy fail: ");
		return NULL;
	}
	// create according to define
#ifdef ONE_MKL
	newTensor->data = MKL_malloc(t->data_size * t->item_size, 64);
#else
	newTensor->data = malloc(t->data_size * t->item_size);
#endif
	if (newTensor->data == NULL) {
		perror("Create tensor data fail: ");
		return NULL;
	}

	memcpy(newTensor->data, t->data, t->data_size * t->item_size);

	newTensor->data_size = t->data_size;
	newTensor->type = t->type;
	newTensor->item_size = t->item_size;

	newTensor->dimension = (int64_t*)malloc(t->dimension_size * sizeof(int64_t));
	memcpy_s(newTensor->dimension, t->dimension_size * sizeof(int64_t), t->dimension, t->dimension_size * sizeof(int64_t));
	newTensor->dimension_size = t->dimension_size;

	newTensor->stride = (int64_t*)malloc(t->dimension_size * sizeof(int64_t));
	memcpy_s(newTensor->stride, t->dimension_size * sizeof(int64_t), t->stride, t->dimension_size * sizeof(int64_t));
	newTensor->is_static = 0;
	if (t->is_size_unknown == true)newTensor->is_size_unknown = true;
	if (t->is_size_unknown == false)newTensor->is_size_unknown = false;
	return newTensor;
}

int resize_tensor(struct tensor* t, int64_t* new_dimension, int64_t new_dimension_size, int item_type) {
	char* temp_ptr = NULL;
	int64_t i = 0, new_total = 1;
	if (t == NULL || new_dimension == NULL) return NULL;

	if (t->is_static) return 0;
	t->type = item_type;
	if (item_type == DATATYPE_FLOAT || item_type == DATATYPE_INT32) {
		t->item_size = 4;
	}
	else if (item_type == DATATYPE_INT64) {
		t->item_size = 8;
	}
	else {
		return 0;
	}
	for (i = 0; i < new_dimension_size; i++) {
		new_total *= new_dimension[i];
	}
	// Set data
	if (new_total != t->data_size) {
		if (t->data == NULL) t->data = calloc(new_total ,  t->item_size);
		else {
			temp_ptr = realloc(t->data, new_total * t->item_size);
			if (temp_ptr != NULL) t->data = temp_ptr;
			else return NULL;
			memset(temp_ptr , 0 , new_total * t->item_size);
		}
		if (t->data == NULL) return NULL;
		t->data_size = new_total;

	}
	//Set dimension
	if (t->dimension == NULL) {
		t->dimension = malloc(new_dimension_size * sizeof(int64_t));
	}
	else {
		temp_ptr = realloc(t->dimension, new_dimension_size * sizeof(int64_t));
		if (temp_ptr != NULL) t->dimension = (int64_t*)temp_ptr;
		else return NULL;
	}
	if (t->dimension == NULL) return NULL;

	memcpy_s(t->dimension, new_dimension_size * sizeof(int64_t), new_dimension, new_dimension_size * sizeof(int64_t));
	t->dimension_size = new_dimension_size;
	// Set stride
	if (t->stride == NULL) {
		t->stride = malloc(t->dimension_size * sizeof(int64_t));
	}
	else {
		temp_ptr = realloc(t->stride, t->dimension_size * sizeof(int64_t));
		if(temp_ptr != NULL) t->stride = (int64_t*)temp_ptr;
	}
	if (t->stride == NULL) return 0;
	t->stride[t->dimension_size - 1] = t->item_size;
	for (i = t->dimension_size - 2; i >= 0; i--) {	// Calcualte stride
		t->stride[i] = t->stride[i + 1] * t->dimension[i + 1];
	}
	t->is_size_unknown = false;	// Size must be known
	return 1;
}
int overwrite_tensor(struct tensor* to, struct tensor* from) {
	if (!is_shape_compatible_tensor(from, to)) return 0;
	memcpy(to->data, from->data, from->data_size * from->item_size);
}

void release_tensor(struct tensor** t) {
	if (*t) {
		if (!(*t)->is_static) {
#ifdef ONE_MKL
			mkl_free((*t)->data);
#else 
			free((*t)->data);
#endif // ONE_MKL
			free((*t)->dimension);
		}
		free((*t)->stride);
		free((*t));
		*t = NULL;
	}
}

int is_shape_compatible_tensor(struct tensor* A, struct tensor* B) {
	int n = 0;
	if (A == NULL || B == NULL) return NULL;
	if (A->dimension_size != B->dimension_size || A->type != B->type) {
		return NULL;
	}
	n = memcmp(A->dimension, B->dimension, 1);
	if (n != 0) return 0;
	else return 1;
}

int64_t is_shape_broadcastable_tensor(struct tensor* A, struct tensor* B, int64_t** dimension) {
	__int64 i = 0, j = 0;
	if (A == NULL || B == NULL) return 0;
	i = A->dimension_size - 1;
	j = B->dimension_size - 1;
	//https://numpy.org/doc/stable/user/basics.broadcasting.html
	// from dimension right to left 
	// If both dimension is equal
	// IF one of them is one
	// they are broad castable
	while (i > -1 && j > -1) {
		if (A->dimension[i] == 1 || B->dimension[j] == 1 || A->dimension[i] == B->dimension[j]) {
			i--;
			j--;
		}
		else {
			return 0;
		}
	}
	if (A->data_size >= B->data_size) {
		*dimension = A->dimension;
		return A->dimension_size;
	}
	else {
		*dimension = B->dimension;
		return B->dimension_size;
	}
}

int is_dimension_tensor(struct tensor* t, int64_t* dimension, int64_t length) {
	int64_t i = 0;
	if (t == NULL) return 0;
	if (t->dimension_size != length) return 0;
	for (i = 0; i < length; i++) {
		if (t->dimension[i] != dimension[i]) return 0;
	}
	return 1;
}
void print_tensor(struct tensor* t) {
	
	if (t == NULL) {
		printf("tensor is null\n");
		return;
	}
	printf("\nPRINT start\n");
	if (t->is_size_unknown) {
		printf("Unknown size tensor\n");
	}
	else {
		printf("dimension: ");
		for (int i = 0; i < t->dimension_size; i++) {
			printf("%"PRId64", ", t->dimension[i]);
		}
		printf("\ndata:\n ");
		if (t->type == DATATYPE_FLOAT) {
			float* d = t->data, total = 0;
			for (int64_t i = 0; i < t->data_size; i++) {
				printf("%f, ", d[i]);
				total += d[i];
			}
			total /= t->data_size;
			printf("\n Average: %f\n", total);
		}
		else if (t->type == DATATYPE_INT32) {
			int32_t* d = t->data, total = 0;
			for (int64_t i = 0; i < t->data_size; i++) {
				printf("%d, ", d[i]);
				total += d[i];
			}
			total /= t->data_size;
			printf("\n Average: %ld\n", total);
		}
		else if (t->type == DATATYPE_INT64) {
			int64_t* d = t->data, total = 0;
			for (int64_t i = 0; i < t->data_size; i++) {
				printf("%lld, ", d[i]);
				total += d[i];
			}
			printf("\n Average: %lld\n", total);
		}
		else {
			printf("Unsupported type\n");
			return;
		}

		printf("\n\nStride: ");
		for (int i = 0; i < t->dimension_size; i++) {
			printf("%"PRId64" ", t->stride[i]);
		}
	}

	printf("\nPRINT END\n");


}

void print_tensor_dim(struct tensor* t) {
	if (t == NULL) {
		printf("t is null\n");
		return;
	}
	if (t->is_size_unknown) {
		printf("Unknown size tensor\n");
	}
	else {
		printf("dimension: ");
		for (int i = 0; i < t->dimension_size; i++) {
			printf("%"PRId64" ", t->dimension[i]);
		}
		printf("\nStride: ");
		for (int i = 0; i < t->dimension_size; i++) {
			printf("%"PRId64" ", t->stride[i]);
		}
	}
	printf("\nPRINT END\n");
}


struct tensor_iterator* create_tensor_iterator(struct tensor* t) {
	int64_t i = 0;
	if (t == NULL) return NULL;
	struct tensor_iterator* it = (struct tensor_iterator*)calloc(1, sizeof(struct tensor_iterator));
	if (it == NULL) {
		perror("create Create tensor iterator fail: ");
		return NULL;
	}
	it->data = t->data;
	it->size = t->data_size;
	it->coordinate = calloc(t->dimension_size, sizeof(int64_t));
	if (it->coordinate == NULL) {
		perror("create Create tensor iterator fail: ");
		return NULL;
	}
	it->dimension = t->dimension;
	it->dimension_size = t->dimension_size;
	it->stride = t->stride;
	it->backstride = calloc(t->dimension_size, sizeof(int64_t));
	if (it->backstride == NULL) {
		perror("create Create tensor iterator backstrides fail: ");
		return NULL;
	}
	for (i = 0; i < t->dimension_size; i++) {
		it->backstride[i] = it->stride[i] * (it->dimension[i] - 1);
	}
	it->factor = calloc(t->dimension_size, sizeof(int64_t));
	if (it->factor == NULL) {
		perror("create Create tensor iterator factor fail: ");
		return NULL;
	}
	for (int i = 0; i < it->dimension_size; i++) {
		it->factor[i] = it->stride[i] / t->item_size;
	}
	return it;
}
void reset_tensor_iter(struct tensor_iterator* it) {
	if (it == NULL)return;
	if (it->helper == NULL) {
		int64_t i = 0;
		// remove on each dimension
		for (i = 0; i < it->dimension_size; i++) {
			it->data -= it->coordinate[i] * it->stride[i];
			it->coordinate[i] = 0;
		}
		it->index = 0;
	}
	else {
		// other types of iterators?

	}
}
int16_t next_tensor_iter(struct tensor_iterator* it) {
	int64_t i = 0;
	if (it == NULL) return 0;
	if (it->helper == NULL) {
		if (it->index + 1 == it->size) return -1;	// last element
		it->coordinate[it->dimension_size - 1]++;
		i = it->dimension_size - 1;
		//from right to left, if coordinate reaches dimension max, set to 0 and carry 1 to next dimension.
		while (it->coordinate[i] == it->dimension[i] && i >= 0) {
			it->coordinate[i] = 0;
			i--;
			it->coordinate[i]++;
		}
		it->index++;
		it->data += it->stride[it->dimension_size - 1];
		return 1;
	}
	else return NULL;
}
int16_t is_not_done_tensor_iter(struct tensor_iterator* it) {
	if (it == NULL) return NULL;
	if (it->helper == NULL) {
		if (it->index < it->size - 1) return 1;
		else return 0;
	}
	else {
		return 0;
	}

}
int16_t goto_tensor_iter(struct tensor_iterator* it, int64_t* target_coordinate) {
	int64_t i = 0;
	if (it == NULL || target_coordinate == NULL) return NULL;
	if (it->helper == NULL) {
		for (i = 0; i < it->dimension_size; i++) {
			// Out of bounds
			if (target_coordinate[i] >= it->dimension[i])return NULL;
		}
		it->index = 0;
		for (i = 0; i < it->dimension_size; i++) {
			it->data += (target_coordinate[i] - it->coordinate[i]) * it->stride[i];
			it->index += target_coordinate[i] * it->factor[i];
			it->coordinate[i] = target_coordinate[i];
		}
	}
	return 1;
}
int16_t goto_1d_tensor_iter(struct tensor_iterator* it, int64_t index) {
	int64_t i = 0, j = 0;
	if (it == NULL) return NULL;
	if (index < 0) index = it->size + index;
	if (index >= it->size)return -1;
	i = 0;
	index -= it->index;
	it->index += index;
	while (index != 0 && i < it->dimension_size) {
		j = index / it->factor[i];
		if (j != 0) {
			it->data += j * it->stride[i];
			it->coordinate[i] += j;
			index -= j * it->factor[i];
		}

		i++;
	}
	if (index != 0) return -1;
	return 1;
}

void* get_data_tensor_iter(struct tensor_iterator* it) {
	if (it == NULL) return NULL;
	else return  it->data;
}



//debug function
void print_tensor_iter(struct tensor_iterator* it) {
	if (it == NULL) {
		printf("Iterator is null\n");
		return;
	}
	printf("1d: index %"PRId64" of total %"PRId64"\n", it->index, it->size);
	printf("3d coordinate: ");
	for (int i = 0; i < it->dimension_size; i++) {
		printf("%"PRId64" ", it->coordinate[i]);
	}
	printf(", of dimension: ");
	for (int i = 0; i < it->dimension_size; i++) {
		printf("%"PRId64" ", it->dimension[i]);
	}
	printf("\n");
	if (it->data == NULL) {
		printf("Iterator no data\n");
	}
	else {
		printf("Data is %f or %d or %" PRId64 "\n", *(float*)it->data, *(int*)it->data, *(int64_t*)it->data);
	}
	printf("stride: ");
	for (int i = 0; i < it->dimension_size; i++) {
		printf("%"PRId64" ", it->stride[i]);
	}
	printf("\nbackstride:  ");
	for (int i = 0; i < it->dimension_size; i++) {
		printf("%"PRId64" ", it->backstride[i]);
	}
	printf("\nfactor ");
	for (int i = 0; i < it->dimension_size; i++) {
		printf("%"PRId64" ", it->factor[i]);
	}
	printf("\n");
}

void release_tensor_iterator(struct tensor_iterator** it) {
	if (*it) {
		free((*it)->backstride);
		free((*it)->coordinate);
		free((*it)->factor);
		free(*it);
		*it = NULL;
	}
}