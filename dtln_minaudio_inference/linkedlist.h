#ifndef LINKEDLIST_H
#define LINKEDLIST_H

#define DYNAMIC_ARRAY_EXPAND 5  // Expand increment of dynamic_array

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Mainly used for node.h nodes
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

/// <summary>
/// Get node from index
/// </summary>
void* get_list(struct list* list, int64_t index);
/// <summary>
///  change data in node
/// </summary>
void* replace_list(struct list* list, void* new_data, int64_t index);
/// <summary>
/// Get node from index
/// </summary>
int pushback_list(struct list* list, void* new_data);
/// <summary>
/// Retreives node from the back and pop from list. Do remember to free popped object
/// </summary>
struct list_node* popback_list(struct list* list);



/// <summary>
/// contiguous data array
/// </summary>
struct dynamic_array {
    // number of elements
    int64_t size;
    // byte size of item
    int64_t item_size;
    // number of elements available
    int64_t capacity;
    //data pointer
    char *data;
};

struct  dynamic_array* create_darray(int64_t item_size);
/// <summary>
/// Create array from 
/// </summary>
/// <param name="data"></param>
/// <param name="num_items"></param>
/// <param name="item_size"></param>
/// <returns>Address or NULL if fails</returns>
struct  dynamic_array* create_darray_array(void* data, int64_t num_items, int64_t item_size);
/// <summary>
/// Pop front
/// </summary>
/// <param name="arr"></param>
/// <returns>1, if something is popped.0, if nothing popped.</returns>
int popfront_darray(struct dynamic_array* arr);
/// <summary>
/// Pop back
/// </summary>
/// <param name="arr"></param>
/// <returns>1, if something is popped.0, if nothing popped.</returns>
int popback_darray(struct dynamic_array* arr);
/// <summary>
/// Push front
/// </summary>
/// <param name="arr"></param>
/// <returns>1, if succes.0, if fail.</returns>
int pushfront_darray(struct dynamic_array* arr, void* item);
/// <summary>
/// Push back
/// </summary>
/// <param name="arr"></param>
/// <returns>1, if succes.0, if fail.</returns>
int pushback_darray(struct dynamic_array* arr, void *item);

/// <summary>
/// Delete array data at pos
/// </summary>
/// <param name="arr"></param>
/// <param name="pos"></param>
void delete_darray(struct dynamic_array* arr, int64_t pos);
/// <summary>
/// get array data at pos
/// </summary>
char* get_darray(struct dynamic_array* arr, int64_t pos);
/// <summary>
/// get data at front
/// </summary>
char* front_darray(struct dynamic_array* arr);
/// <summary>
/// get data at back
/// </summary>
char* back_darray(struct dynamic_array* arr);
/// <summary>
///  Clear array
/// </summary>
void clear_darray(struct dynamic_array* arr);
/// <summary>
/// Release evething in array
/// </summary>
void release_darray(struct dynamic_array** arr);
/// <summary>
/// Shring data size to current number of data
/// </summary>
void shrink_to_fit_darray(struct dynamic_array* arr);
/// <summary>
/// release dynamic array, but does not free data. Useful for calulating output dimension
/// </summary>
void release_darray_keep_data(struct dynamic_array** arr);

// Print with data type int64
void print_darray_int64(struct dynamic_array* arr);

#endif // LINKEDLIST_H







