
#include "utility.h"
void swap(void* a, void* b)
{
	 void* c = a;
	 a = b;
	 b = c;
}

void swapf(float* a, float* b)
{
	float c = *a;
	*a = *b;
	*b = c;
}
void safe_free(void** p) {
	if (*p != 0) {
		free(*p);
		*p = 0;
	}
}
void print_int64_t(int64_t* arr, int len) {

	for (int i = 0; i < len; i++) {
		printf("%lld, " ,arr[i]);
	}
	printf("\b\b");

}

size_t datatype_size(const DATATYPE type) {
	if (type == DATATYPE_FLOAT) {
		return 4;
	}
	else if (type == DATATYPE_INT32) {
		return 4;
	}
	else if (type == DATATYPE_INT64) {
		return 8;
	}
	return 0;
}
