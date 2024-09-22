#pragma once
// Function level calculation
// 1. It's not necassay to create OPS function on array level, many opeation requires traverseing the tensor on multi dimension
// 2. Transpose has to be done with knowing the tensor dimension, nut it is difficult to use a full transpose function in an integral part of antor tensor fun ction
#ifndef FUNCTION_OPS_H
#define FUNCTION_OPS_H
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "linkedlist.h"
#include "utility.h"
#include "define.h"
#include "tensor.h"
#include "ops.h"
#ifdef ONE_MKL	// One mkl header
#include "mkl.h"
#endif

/// <summary>
/// Broadcast type function 
/// </summary>
/// <param name="A"></param>
/// <param name="B"></param>
/// <param name="C"></param>
/// <param name="op_type"></param>
/// <returns></returns>
int broadcast_function(struct tensor* A, struct tensor* B, struct tensor* C, NODE_TYPE op_type);

// one to one TENSORS
int tanh_function(struct tensor* A, struct tensor* B);
int sigmoid_function(struct tensor* A, struct tensor* B);
int relu_function(struct tensor* A, struct tensor* B);
int sqrt_function(struct tensor* A, struct tensor* B);
int copy_function(struct tensor* A, struct tensor* B);


int matmul_function(struct tensor* a, struct tensor* b, struct tensor* c);

int clip_function(struct tensor* input, struct tensor* min, struct tensor* max, struct tensor* output);

int argmax_function(int64_t* axis, int64_t* keepdims, int64_t* select_last_index, struct tensor* input, struct tensor* reduced);
int argmin_function(int64_t* axis, int64_t* keepdims, int64_t* select_last_index, struct tensor* input, struct tensor* reduced);
int slice_function(struct tensor* data, struct tensor* output, int64_t* starts, int64_t* ends, int64_t* axis, int64_t* steps);
//// transpose multi_dim
int transpose_function(struct tensor* data, struct tensor* transposed, int64_t* perm);



// gemm for tensor
int gemm_function(struct tensor* a, struct tensor* b, struct tensor* c, struct tensor* output, float alpha, float beta, int64_t transA, int64_t transB);
int lstm_function(float* activation_alpha, float* activation_beta, struct list* activations, float* clip, char* direction, int64_t hidden_size, int64_t *input_forget, int64_t *layout,
	struct tensor* x, struct tensor* w, struct tensor* r, struct tensor* b, int64_t* seq_length, struct tensor* initial_h, struct tensor* initial_c, struct tensor* P, struct tensor* Y, struct tensor* Y_h, struct tensor* Y_c);
int conv_function(struct tensor* x, struct tensor* w, struct tensor* b, struct tensor* y, int64_t* dilations, int64_t groups, int64_t* kernel_shapes, int64_t* pads, int64_t* stride);
int concat_function(int64_t* axis, struct list* inputs, struct tensor* concat_result);
int split_function(int64_t axis, int64_t num_outputs, struct tensor* input, int64_t* split, struct list* outputs);
int reducemean_function(int64_t keepdims, int64_t noop_with_empty_axes, struct tensor* data, struct list* axes, struct tensor* reduced);

int pad_function(char* mode, struct tensor* data, struct tensor* pads, struct tensor* constant_value, struct tensor* axes, struct tensor* output);
// pad for tensor
int pad_function_simple(struct tensor* data, struct tensor* output, int64_t* pads, char* mode, void* constant_value);
/// <summary>
/// Zero pad for use by averagepool and conv
/// functionality, output will automatically be created if it is NULL;
/// </summary>
/// <param name="ceil_mode"></param>
/// <param name="input"></param>
/// <param name="kernel_shape"></param>
/// <param name="pads"></param>
/// <param name="stides"></param>
/// <param name="output"></param>
/// <returns></returns>
//int zero_pad_function(int64_t* ceil_mode, const struct tensor* input, int64_t* kernel_shape, int64_t* pads, int64_t* stides, struct tensor** output);
//
//int averagepool_function();
#endif