#ifndef LINKEDLIST_H
#define LINKEDLIST_H

# define DYNAMIC_ARRAY_EXPAND 5
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
/// <summary>
/// node of list
/// </summary>
struct list_node
{
    void* data;
    struct list_node* next;
};

/// <summary>
/// linked list
/// </summary>
struct list
{
    int64_t size;
    struct list_node* first;
};

struct list* create_list_from_array(void* data,int64_t num_items, int64_t item_size);
/// <summary>
/// Get node from index
/// </summary>
void* get_data_list(struct list* list, int64_t index);

void* replace_data_list(struct list* list, void* new_data, int64_t index);
/// <summary>
/// Get node from index
/// </summary>
int push_back_list(struct list* list, void* new_data);

/// <summary>
/// Retreives node from the back and pop from list. Do remember to free popped object
/// </summary>
struct list_node* pop_back_list(struct list* list);


struct dynamic_array {
    // element size
    int64_t size;
    // byte size of item
    int64_t item_size;
    // number of elements available
    int64_t capacity;
    //data pointer
    char *data;
};
struct  dynamic_array* create_array(int64_t item_size);
struct  dynamic_array* create_array_from_array(void* data, int64_t num_items, int64_t item_size);
int popfront_array(struct dynamic_array* arr);
int popback_array(struct dynamic_array* arr);
int pushfront_array(struct dynamic_array* arr, void* item);
int pushback_array(struct dynamic_array* arr, void *item);
void delete_item_array(struct dynamic_array* arr, int64_t pos);
char* get_item_array(struct dynamic_array* arr, int64_t pos);
char* front_array(struct dynamic_array* arr);
char* back_array(struct dynamic_array* arr);
void release_array(struct dynamic_array* arr);
void print_array_int64(struct dynamic_array* arr);
void clear_array(struct dynamic_array* arr);
#endif // LINKEDLIST_H







