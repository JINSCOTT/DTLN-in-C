#include "linkedlist.h"

void* get_list(struct list* list, int64_t index) {
	int64_t i = 0;
	struct list_node* current = list->first;
	if (list->size <= index || index < 0) return NULL;
	else {
		for (i = 0; i < index; i++) current = current->next;
		return current->data;
	}
}

void* replace_list(struct list* list,void* data,  int64_t index) {
	int64_t i = 0;
	struct list_node* current = list->first;
	char* temp = NULL;
	if (list->size <= index || index < 0) return NULL;
	else {
		for (i = 0; i < index; i++) current = current->next;
		temp = current->data;
		current->data = data;
		return temp;
	}
}

int pushback_list(struct list* list, void* new_data) {
	int i = 0;
	struct list_node** current = &list->first;
	for (i = 0; i < list->size; i++)current = &(*current)->next;

	struct list_node* new_node = (struct list_node*)malloc(sizeof(struct list_node));
	if (new_node != NULL) {
		new_node->data = new_data;
		new_node->next = NULL;
		*current = new_node;
		list->size += 1;
		return 1;
	}
	else {
		printf("is null\n");
		return 0;  // Can't create new node
	}
}



struct list_node* popback_list(struct list* list) {
	int64_t i = 0;
	struct list_node* current = list->first;
	if (list->size != 0) {
		for (i = 0; i < list->size - 1; i++)current = current->next;
		list->size--;
		return current;
	}
	return NULL;
}


struct  dynamic_array* create_darray(int64_t item_size) {
	struct dynamic_array* arr = malloc(sizeof(struct dynamic_array));
	if (arr == NULL)return NULL;
	arr->size = 0;
	arr->item_size = item_size;
	arr->data = calloc(DYNAMIC_ARRAY_EXPAND , item_size);
	if (arr->data == NULL) {
		free(arr);
		return NULL;
	}
	arr->capacity = 5;
	return arr;
}

struct  dynamic_array* create_darray_array(void* data, int64_t num_items, int64_t item_size) {
	struct dynamic_array* arr = malloc(sizeof(struct dynamic_array));
	if (arr == NULL)return NULL;
	arr->size = num_items;
	arr->item_size = item_size;
	arr->data = calloc(num_items, item_size);
	if (arr->data == NULL) {
		free(arr);
		return NULL;
	}
	arr->capacity = num_items;
	memcpy_s(arr->data, num_items * item_size, data, num_items * item_size);
	return arr;
}
int popfront_darray(struct dynamic_array* arr) {
	if (arr == NULL) return 0;
	if (arr->size == 0)return  0;
	memcpy_s(arr->data, arr->capacity * arr->size, arr->data + arr->item_size, arr->item_size * (arr->size - 1));
	arr->size--;
	return 1;
}
int popback_darray(struct dynamic_array* arr) {
	if (arr == NULL) return 0;
	if (arr->size == 0) return 0;
	arr->size--;
	return 1;
}
int pushfront_darray(struct dynamic_array* arr, void* item) {
	char* new_arr = NULL;
	if (arr == NULL || item == NULL) return 0;
	if (arr->size != 0) {
		if (arr->capacity == arr->size) {
			new_arr = realloc(arr->data, (arr->capacity + DYNAMIC_ARRAY_EXPAND) * arr->item_size);
			if (new_arr == NULL) { return 0; };
			arr->data = new_arr;
			arr->capacity += 5;
		}
		memcpy_s(arr->data + arr->item_size, arr->capacity * arr->item_size, arr->data, arr->item_size * arr->size);
	}
	memcpy_s(arr->data, arr->capacity * arr->item_size, item, arr->item_size);
	arr->size++;
	return 1;

}
int pushback_darray(struct dynamic_array* arr, void* item) {
	char* new_arr = NULL;
	if (arr == NULL || item == NULL) return 0;
	if (arr->capacity == arr->size) {
		new_arr = realloc(arr->data, (arr->capacity + DYNAMIC_ARRAY_EXPAND) * arr->item_size);
		if (new_arr == NULL) { return 0; };
		arr->data = new_arr;
		arr->capacity += 5;
	}
	memcpy_s(arr->data + arr->size * arr->item_size, (arr->capacity - arr->size) * arr->item_size, item, arr->item_size);
	arr->size++;
	return 1;
}

void delete_darray(struct dynamic_array* arr, int64_t pos) {
	if (pos < 0 || arr == NULL) return;
	if (pos >= arr->size) return;
	memcpy_s(arr->data + pos * arr->item_size, (arr->capacity - pos) * arr->item_size, arr->data+(pos+1) * arr->item_size, (arr->size - pos-1) * arr->item_size);
	arr->size--;
}
char* get_darray(struct dynamic_array* arr, int64_t pos) {
	if (pos < 0 || arr == 0) return NULL;
	if (pos >= arr->size) return NULL;
	return arr->data + pos * arr->item_size;

}
char* front_darray(struct dynamic_array* arr) {
	if (arr == 0) return 0;
	if (arr->size == 0) return 0;
	else return arr->data;
}
char* back_darray(struct dynamic_array* arr) {
	if (arr == 0) return 0;
	if (arr->size == 0) return 0;
	else return arr->data + (arr->size - 1) * arr->item_size;
}

void release_darray(struct dynamic_array** arr) {
	if (arr != NULL) {
		free((*arr)->data);
		free(*arr);
		arr = NULL;
	}
}
void shrink_to_fit_darray(struct dynamic_array* arr) {
	char *temp_ptr = NULL;
	if (arr!= NULL) {
		if (arr->size >= 1) {
			if (arr->capacity != arr->size) {
				if (arr->data != NULL) {
					temp_ptr = (char*)realloc(arr->data, arr->size * arr->item_size);
					if (temp_ptr != NULL) {
						arr->data = temp_ptr;
					}// Use the original block if shrink fails
					arr->capacity = arr->size;
				}
			}
		}
	}
}
void release_darray_keep_data(struct dynamic_array** arr) {
	if (arr != NULL) {
		shrink_to_fit_darray(*arr);
		free(*arr);
		arr = NULL;
	}
}
void print_darray_int64(struct dynamic_array* arr) {
	if (arr == NULL) return;
	printf("int64_t array %lld elements: ", arr->size);
	for (int i = 0; i < arr->size; i++) {
		printf("%lld, ", *(int64_t*)get_darray(arr, i));
	}
	printf("\b\b array print end\n");
}

void clear_darray(struct dynamic_array* arr){
	if (arr != NULL)arr->size = 0;
}

