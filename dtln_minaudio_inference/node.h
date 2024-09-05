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
	int type;
	struct list attribute;
	struct list input;
	struct list output;
};

// Create empty tensor
struct node* create_node();

// Specific node creationa
void print_ops_error(int i);
/// <summary>
/// Create add node.
/// </summary>
/// <param name="A">Input 0</param>
/// <param name="B">Input 1</param>
/// <param name="C">Output 0</param>
struct node* create_add_node(struct tensor* A, struct tensor* B, struct tensor* C);
/// <summary>
/// Create sub node.
/// </summary>
/// <param name="A">Input 0</param>
/// <param name="B">Input 1</param>
/// <param name="C">Output 0</param>
struct node* create_sub_node(struct tensor* A, struct tensor* B, struct tensor* C);
/// <summary>
/// Create mul node.
/// </summary>
/// <param name="A">Input 0</param>
/// <param name="B">Input 1</param>
/// <param name="C">Output 0</param>
struct node* create_mul_node(struct tensor* A, struct tensor* B, struct tensor* C);
/// <summary>
/// Create div node.
/// </summary>
/// <param name="A">Input 0</param>
/// <param name="B">Input 1</param>
/// <param name="C">Output 0</param>
struct node* create_div_node(struct tensor* A, struct tensor* B, struct tensor* C);
/// <summary>
/// Create Tanh node.
/// </summary>
/// <param name="input">Input 0</param>
/// <param name="output">Output 0</param>
struct node* create_tanh_node(struct tensor* input, struct tensor* output);
/// <summary>
/// Create Sigmoid node.
/// </summary>
/// <param name="X">Input 0</param>
/// <param name="Y">Output 0</param>
struct node* create_sigmoid_node(struct tensor* X, struct tensor* Y);
/// <summary>
/// Create Sqrt node.
/// </summary>
/// <param name="X">Input 0</param>
/// <param name="Y">Output 0</param>
struct node* create_sqrt_node(struct tensor* X, struct tensor* Y);
/// <summary>
/// Create relu node.
/// </summary>
/// <param name="X">Input 0</param>
/// <param name="Y">Output 0</param>
struct node* create_sqrt_node(struct tensor* X, struct tensor* Y);
/// <summary>
/// Create squeeze node
/// </summary>
/// <param name="data">input 0</param>
/// <param name="axes">input 1. Optional.</param>
/// <param name="squeezed">output 0</param>
struct node* create_squeeze_node(struct tensor* data, struct tensor* axes, struct tensor* squeezed);
/// <summary>
/// Create unsqueeze ndoe
/// </summary>
/// <param name="data">input 0</param>
/// <param name="axes">input 1</param>
/// <param name="expanded">output 0</param>
struct node* create_unsqueeze_node(struct tensor* data, struct tensor* axes, struct tensor* expanded);
/// <summary>
/// create transpose node
/// </summary>
/// <param name="perm">attribute 0. Optional</param>
/// <param name="data">input 0</param>
/// <param name="transposed">output 0</param>
struct node* create_transpose_node(int64_t* perm, struct tensor* data, struct tensor* transposed);
/// <summary>
/// create matmul node
/// </summary>
/// <param name="A">input 0</param>
/// <param name="B">input 1</param>
/// <param name="Y">output 0</param>
struct node* create_matmul_node(struct tensor* A, struct tensor* B, struct tensor* Y);
/// <summary>
/// create slice node
/// </summary>
/// <param name="data">input 0</param>
/// <param name="starts">input 1</param>
/// <param name="ends">input 2</param>
/// <param name="axes">input 3. Optional</param>
/// <param name="steps">input 4. Optional</param>
/// <param name="output">output 0</param>
struct node* create_slice_node(struct tensor* data, struct tensor* starts, struct tensor* ends, struct tensor* axes, struct tensor* steps, struct tensor* output);
/// <summary>
/// Create gemm node
/// </summary>
/// <param name="alpha">attribute 0</param>
/// <param name="beta">attribute 1</param>
/// <param name="transA">attribute 2</param>
/// <param name="transB">attribute 3</param>
/// <param name="A">input 0</param>
/// <param name="B">input 1</param>
/// <param name="C">input 2. Optional</param>
/// <param name="Y">output 0</param>
struct node* create_gemm_node(float* alpha, float* beta, int64_t* transA, int64_t* transB, struct tensor* A, struct tensor* B, struct tensor* C, struct tensor* Y);
/// <summary>
/// create concat node
/// </summary>
/// <param name="axis">attribute 0</param>
/// <param name="concat_result">output 0</param>
/// <param name="num_input">size of inputs</param>
/// <param name="...">inputs variadic</param>
struct node* create_concat_node(int64_t* axis, struct tensor* concat_result, int64_t num_input, ...);
/// <summary>
/// create split node
/// </summary>
/// <param name="axis">attribute 0. optional</param>
/// <param name="num_outputs">attribute 1</param>
/// <param name="input">input 0</param>
/// <param name="split">input 1. optional</param>
/// <param name="...">outputs variadic</param>
struct node* create_split_node(int64_t* axis, int64_t* num_outputs, struct tensor* input, int64_t* split, ...);
/// <summary>
/// create reshape node
/// </summary>
/// <param name="allowzero">attribute 0. optional</param>
/// <param name="data">input 0</param>
/// <param name="shape">input 1</param>
/// <param name="reshaped">output 0</param>
struct node* create_reshape_node(int64_t* allowzero, struct tensor* data, struct tensor* shape, struct tensor* reshaped);
/// <summary>
/// create pad node
/// </summary>
/// <param name="mode">attribute 0. optiona;</param>
/// <param name="data">input 0</param>
/// <param name="pads">input 1</param>
/// <param name="constant_value">input 2. optional</param>
/// <param name="axes">input 3. optional</param>
/// <param name="output">output 0</param>
struct node* create_pad_node(char* mode, struct tensor* data, struct tensor* pads, struct tensor* constant_value, struct tensor* axes, struct tensor* output);
/// <summary>
/// create conv node
/// </summary>
/// <param name="auto_pad">attribute 0. Optional</param>
/// <param name="dilations">attibute 1. Optional</param>
/// <param name="group">attribute 2. Optional</param>
/// <param name="kernel_shape">Attribute 3. Optional</param>
/// <param name="pads">Attribute 4. Optional</param>
/// <param name="strides">Attribute 5. Optional</param>
/// <param name="X">Input 0</param>
/// <param name="W">Input 1</param>
/// <param name="B">Input 2. Optional</param>
/// <param name="Y">Output 0</param>
struct node* create_conv_node(char* auto_pad, int64_t* dilations, int64_t* group, int64_t* kernel_shape, int64_t* pads, int64_t* strides, struct tensor* X, struct tensor* W, struct tensor* B, struct tensor* Y);
/// <summary>
/// Create lstm node
/// </summary>
/// <param name="activation_alpha">Attrbute 0. Optional</param>
/// <param name="activation_beta">Attribute 1. Optional</param>
/// <param name="activations">Attribute 2. Optional</param>
/// <param name="clip">Attribute 3. Optional</param>
/// <param name="direction">Attribute 4. Optional</param>
/// <param name="hidden_size">Attribute 5</param>
/// <param name="input_forget">Attribute 6. Optional</param>
/// <param name="layout">Attribute 7. Optional</param>
/// <param name="x">input 0</param>
/// <param name="w">input 1</param>
/// <param name="r">input 2</param>
/// <param name="b">input 3. Optional</param>
/// <param name="sequence_lens">input 4. Optional</param>
/// <param name="initial_h">Input 5. Optional</param>
/// <param name="initial_c">Input 6. Optional</param>
/// <param name="p">Input 7. Optional</param>
/// <param name="y">Output 0. Optional</param>
/// <param name="y_h">Output 1. Optional</param>
/// <param name="y_c">Output 2. Optional</param>
struct node* create_lstm_node(float* activation_alpha, float* activation_beta, struct list* activations, float* clip, char* direction, int64_t* hidden_size, int64_t* input_forget, int64_t* layout,
	struct tensor* x, struct tensor* w, struct tensor* r, struct tensor* b, int64_t* sequence_lens, struct tensor* initial_h, struct tensor* initial_c, struct tensor* p,
	struct tensor* y, struct tensor* y_h, struct tensor* y_c);
/// <summary>
/// create reduce mean node 
/// </summary>
/// <param name="keepdim">attribute 0. Optional</param>
/// <param name="noop_with_empty_axes">attribute 1. Optional</param>
/// <param name="data">Input 0</param>
/// <param name="axes">Input 1. Optional</param>
/// <param name="reduced">Output 0</param>
struct node* create_reducemean_node(int64_t* keepdim, int64_t* noop_with_empty_axes, struct tensor* data, struct tensor* axes, struct tensor* reduced);
/// <summary>
/// Create constant node
/// note: saprse value, not supported. Only one of the attribute should not be NULL
/// </summary>
/// <param name="value">attribute 0. Optional</param>
/// <param name="value_float">attribute 1. Optional</param>
/// <param name="value_floats">attribute 2. Optional</param>
/// <param name="value_int">attribute 3. Optional</param>
/// <param name="value_ints">attribute 4. Optional</param>
/// <param name="output">Output 0</param>
struct node* create_constant_node(struct tensor* value, float* value_float, float* value_floats, int64_t* value_int, int64_t* value_ints, struct tensor* output);

//// nodes inferemce
int inference_node(struct node* node);

int inference_add_node(struct node* n);
int inference_sub_node(struct node* n);
int inference_mul_node(struct node* n);
int inference_div_node(struct node* n);
int inference_tanh_node(struct node* n);
int inference_sigmoid_node(struct node* n);
int inference_sqrt_node(struct node* n);
int inference_relu_node(struct node* n);
int inference_squeeze_node(struct node* n);
int inference_unsqueeze_node(struct node* n);
int inference_transpose_node(struct node* n);
int inference_slice_node(struct node* n);
int inference_matmul_node(struct node* n);
int inference_gemm_node(struct node* n);
int inference_concat_node(struct node* n);
int inference_split_node(struct node* n);
int inference_reshape_node(struct node* n);
int inference_pad_node(struct node* n);
int inference_conv_node(struct node* n);
int inference_lstm_node(struct node* n);
int inference_reducemean_node(struct node* n);
int inference_constant_node(struct node* n);
//


#endif // !NODE_H