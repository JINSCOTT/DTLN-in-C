// Node base

#ifndef NODE_H
#define NODE_H

#include "define.h"
#include "linkedlist.h"
#include "ops.h"
#include "tensor.h"
#include "shape_calc.h"
#include "utility.h"
#include "stdarg.h"
/// <summary>
/// Node Class
/// </summary>
struct node {
	int16_t type;
	struct list attribute;
	struct list input;
	struct list output;
};

// base
struct node* create_node();

// Specific node creationa
void print_ops_error(int i);
struct node* create_add_node(struct tensor* A, struct tensor* B, struct tensor* C);
struct node* create_sub_node(struct tensor* A, struct tensor* B, struct tensor* C);
struct node* create_mul_node(struct tensor* A, struct tensor* B, struct tensor* C);
struct node* create_div_node(struct tensor* A, struct tensor* B, struct tensor* C);
struct node* create_tanh_node(struct tensor* X, struct tensor* Y);
struct node* create_sigmoid_node(struct tensor* X, struct tensor* Y);
struct node* create_sqrt_node(struct tensor* X, struct tensor* Y);
struct node* create_squeeze_node(struct tensor* data, struct tensor* axes, struct tensor* squeezed);
struct node* create_unsqueeze_node(struct tensor* data, struct tensor* axes, struct tensor* expanded);
struct node* create_transpose_node(int64_t* perm, struct tensor* data, struct tensor* transposed);
struct node* create_matmul_node(struct tensor* a, struct tensor* b, struct tensor* c);
// starts, ends, axes, steps length has to conform to data dimension length
struct node* create_slice_node(struct tensor* data, struct tensor* starts, struct tensor* ends, struct tensor* axes, struct tensor* steps, struct tensor* output);
struct node* create_gemm_node(float* alpha, float* beta, int64_t* transA, int64_t* transB, struct tensor* A, struct tensor* B, struct tensor* C, struct tensor* Y);
struct node* create_concat_node(int64_t* axis, struct tensor* concat_result, int64_t num_input, ...);
struct node* create_split_node(int64_t* axis, int64_t* num_outputs, struct tensor* input, int64_t* split, ...);
struct node* create_reshape_node(int64_t* allowzero, struct tensor* data, struct tensor* shape, struct tensor* reshaped);
struct node* create_pad_node(char* mode, struct tensor* data, struct tensor* pads, struct tensor* constant_value, struct tensor* axes, struct tensor* output);
struct node* create_conv_node(char* auto_pad, int64_t* dilations, int64_t* group, int64_t* kernel_shape, int64_t* pads, int64_t* strides, struct tensor* x, struct tensor* w, struct tensor* b, struct tensor* y);
struct node* create_lstm_node(float* activation_alpha, float* activation_beta, struct list* activations, float* clip, char* direction, int64_t* hidden_size, int64_t* input_forget, int64_t* layout,
	struct tensor* x, struct tensor* w, struct tensor* r, struct tensor* b, int64_t* sequence_lens, struct tensor* initial_h, struct tensor* initial_c, struct tensor* p,
	struct tensor* y, struct tensor* y_h, struct tensor* y_c);
struct node* create_reducemean_node(int64_t* keepdim, int64_t* noop_with_empty_axes, struct tensor* data, struct tensor* axes, struct tensor* reduced);
struct node* create_constant_node(struct tensor* value, float* value_float, float* value_floats, int64_t* value_int, int64_t* value_ints, struct tensor* output);
//// nodes inferemce
int inference_node(struct node* node);

int16_t inference_add_node(struct node* n);
int16_t inference_sub_node(struct node* n);
int16_t inference_mul_node(struct node* n);
int16_t inference_div_node(struct node* n);
int16_t inference_tanh_node(struct node* n);
int16_t inference_sigmoid_node(struct node* n);
int16_t inference_sqrt_node(struct node* n);
int16_t inference_squeeze_node(struct node* n);
int16_t inference_unsqueeze_node(struct node* n);
int16_t inference_transpose_node(struct node* n);
int16_t inference_slice_node(struct node* n);
int16_t inference_matmul_node(struct node* n);
int16_t inference_gemm_node(struct node* n);
int16_t inference_concat_node(struct node* n);
int64_t inference_split_node(struct node* n);
int16_t inference_reshape_node(struct node* n);
int16_t inference_pad_node(struct node* n);
int16_t inference_conv_node(struct node* n);
int16_t inference_lstm_node(struct node* n);
int16_t inference_reducemean_node(struct node* n);
int16_t inference_constant_node(struct node* n);
//


#endif // !NODE_H