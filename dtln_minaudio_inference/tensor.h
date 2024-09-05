// Tensor holds the data to be computed by inferencing the nodes
// 

#ifndef TENSOR_H
#define TENSOR_H

#include "define.h"

#ifdef ONE_MKL
#include <mkl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
// If attribute can be multitype use tensor to substitute it!!!

/// <summary>
/// Tensor is the data type used to hold data between layers
/// </summary>
struct tensor {
	int64_t* dimension;
	int64_t dimension_size;	// number of elements
	void* data;
	int64_t data_size;		// number of elements
	short type;				// Data type
	short item_size;		// size of the data type
	int64_t* stride;		// number of bytes that must be skipped to get to the next element in that dimension
	short is_static;		// don't change shape and free static
	short is_size_unknown;  // Tensor is created without informationj
};

//Tensor functions

/// <summary>
/// Creates tensor. If tensor is created this way, do "not" free data and dimension after tensor creation;
/// </summary>
/// <param name="data"></param>
/// <param name="num_elements"></param>
/// <param name="dimension"></param>
/// <param name="num_dimension"></param>
/// <param name="DataType"></param>
/// <param name="is_static"></param>
/// <returns></returns>
struct tensor* create_tensor( void* data, int64_t num_elements, int64_t* dimension, int64_t num_dimension, short DataType, short is_static);

struct tensor* create_empty_tensor();
// Create a direct copy, will not be static
struct tensor* create_tensor_copy(struct tensor* t);
/// <summary>
/// Resize tensor with new dimension. new_dimension has to be freed.
/// </summary>
/// <param name="t"></param>
/// <param name="new_dimension"></param>
/// <param name="new_dimension_size"></param>
/// <param name="item_type"></param>
/// <returns></returns>
int resize_tensor(struct tensor* t, int64_t* new_dimension, int64_t new_dimension_size, int item_type);
// change content if their dimension and datasize match 
int overwrite_tensor(struct tensor* to, struct tensor* from);
//release data
void release_tensor(struct tensor** t);
//check if both tensor has the same shape and type
int is_shape_compatible_tensor(struct tensor* A, struct tensor* B);
//debug function
void print_tensor(struct tensor* t);

/// <summary>
/// check if shape is braodcastables
/// </summary>
/// <param name="dimension">Pointer to output dimension. Do not free.</param>
/// <returns>if 0, not broadcastable. If positive, it is dimension length. </returns>
int64_t is_shape_broadcastable_tensor(struct tensor* A, struct tensor* B, int64_t** dimension);

/// <summary>
/// Tensor has the dimension
/// </summary>
/// <returns>1 or 0</returns>
int is_dimension_tensor(struct tensor* t, int64_t* dimension, int64_t length);

/// <summary>
/// very similar to numpy PyArrayIterObject
/// https://numpy.org/doc/stable/license.html
/// </summary>
struct tensor_iterator {
	char* data;				// pointer to tensor data
	int64_t  size;			// total size of data. Number of elements	
	int64_t  index;			// 1d position
	int64_t* coordinate;	// multi dimension position (e.g. (x,y,z))
	int64_t* dimension;		// dimension ref to tensor
	int64_t  dimension_size;// number of dimension
	int64_t* stride;		// How many bytes needed to jump to the next element in each dimension. point to tensor stride
	int64_t* backstride;   // How many bytes needed to jump from the end of a dimension back to its beginning
	int64_t* factor;       // Factor used to convert 1d index into multi-dimension coordinate;
	void* helper;			// reserved for other helpers to iterate for padding or dilate iteration
};

// Tensor iterator functions

/// <summary>
/// Create plain tensor iterator for multi dimension access
/// </summary>
struct tensor_iterator* create_tensor_iterator(struct tensor* t);

/// <summary>
/// Move to the front of tensor data
/// </summary>
void reset_tensor_iter(struct tensor_iterator* iti);
/// <summary>
/// Move to point to the next element
/// </summary>
int16_t next_tensor_iter(struct tensor_iterator* it);
// Go to with multi-dimension coordinate
int16_t goto_tensor_iter(struct tensor_iterator* it, int64_t* target_coordinate);
// Go to with id index
int16_t goto_1d_tensor_iter(struct tensor_iterator* it, int64_t index);
// check if this is the last element
int16_t is_not_done_tensor_iter(struct tensor_iterator* it);
// Get current data from iterator
void* get_data_tensor_iter(struct tensor_iterator* it);

void print_tensor_dim(struct tensor* t);
//debug function
void print_tensor_iter(struct tensor_iterator* it);
//release data
void release_tensor_iterator(struct tensor_iterator** it);
#endif	 // TENSOR_H