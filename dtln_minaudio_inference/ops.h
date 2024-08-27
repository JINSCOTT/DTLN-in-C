// Actual calculation
#ifndef OPS_H
#define OPS_H
#include "linkedlist.h"
#include <float.h>
#include <string.h>
#include <math.h>
#include "utility.h"
#include "define.h"
#include "tensor.h"
#include "mkl.h"


/// <summary>
/// Broadcast type function 
/// </summary>
/// <param name="A"></param>
/// <param name="B"></param>
/// <param name="C"></param>
/// <param name="op_type"></param>
/// <returns></returns>
int16_t broadcast_function(struct tensor* A, struct tensor* B, struct tensor* C, NODE_TYPE op_type);

// one to one TENSORS
int tanh_function(struct tensor* A, struct tensor* B);
int sigmoid_function(struct tensor* A, struct tensor* B);
int relu_function(struct tensor* A, struct tensor* B);
int sqrt_function(struct tensor* A, struct tensor* B);
int copy_function(struct tensor* A, struct tensor* B);

int slice_function(struct tensor* data, struct tensor* output, int64_t* starts, int64_t* ends, int64_t* axis,int64_t* steps);
//// transpose multi_dim
int transpose_function(struct tensor* data, struct tensor* transposed , int64_t* perm);

// Matmul
/// <param name="A">n*m</param>
/// <param name="B">m*p</param>
/// <param name="C">n*p</param>
int matmul_array(void* A, void* B, void* C, int64_t n, int64_t m, int64_t p, int type);
int matmul_function(struct tensor* a, struct tensor* b, struct tensor* c);

int mean_array(void* array, void* result, int64_t num_elements, int type);
int pad_function(char* mode, struct tensor* data, struct tensor* pads, struct tensor* constant_value, struct tensor* axes, struct tensor* output);
// pad for tensor
int pad_function_simple(struct tensor* data, struct tensor* output, int64_t* pads, char* mode,void* constant_value );
// gemm for tensor
int gemm_function(struct tensor* a, struct tensor* b, struct tensor* c, struct tensor* output,float alpha, float beta, int64_t transA, int64_t transB);
 
int lstm_function(float* activation_alpha, float* activation_beta, struct list* activations, float clip, char* direction, int64_t hidden_size, int64_t input_forget, int64_t layout,
	struct tensor* x, struct tensor* w, struct tensor* r, struct tensor* b, int64_t* seq_length, struct tensor* initial_h, struct tensor* initial_c, struct tensor* P, struct tensor* Y, struct tensor* Y_h, struct tensor* Y_c);


int conv_function(struct tensor* x, struct tensor* w, struct tensor* b, struct tensor* y, int64_t* dilations, int64_t groups, int64_t* kernel_shapes, int64_t* pads, int64_t* stride);
int concat_function(int64_t* axis, struct list* inputs, struct tensor* concat_result);
int split_function(int64_t axis, int64_t num_outputs, struct tensor* input, int64_t* split, struct list* outputs);
int reducemean_function(int64_t keepdims, int64_t noop_with_empty_axes, struct tensor* data, struct list* axes, struct tensor* reduced);
/*----------------------------FLOAT FUNCTIONS----------------------------*/


int transposef_2d(float* x, int64_t n, int64_t m ); 

int addf_array(float* a, float* b, float* c, int64_t a_size, int64_t b_size, int64_t c_size);
int mulf_array(float* a, float* b, float* c, int64_t a_size, int64_t b_size, int64_t c_size);

int activationf_array(float* input, float* output, char *type ,int64_t size, float alpha, float beta);
int tanhf_array(float* input, float* output, int64_t size);
int sigmoidf_array(float* input, float* output, int64_t size);
int reluf_array(float* input, float* output, int64_t size);

///<summary>
/// 
/// </summary>
/// <param name="x">Input[C][F], C: Original_channels, F:features </param>
/// <param name="w">Kernel[M][C][K], M: New_channels,C: Original_channels, K: KERNEL DIMENSION</param>
/// <param name="b">Bias[M], M: New_channels </param>
/// <param name="y">Output[M][NF] , M: New_channels , NF: new number of features</param>
/// <param name=""></param>
/// <returns></returns>
int conv1df(float* x, float* w, float* b, float* y, int64_t C, int64_t F, int64_t M, int64_t k, int64_t NF, int64_t stride, int64_t dilation);




/// <summary>
/// 
/// </summary>
/// <param name="x">Input[C][H][W]</param>
/// <param name="w">kernel[M][C][kH][kW]</param>
/// <param name="b">bias[M]</param>
/// <param name="y">[M][nH][nW]</param>
/// <param name="C"></param>
/// <param name="H"></param>
/// <param name="W"></param>
/// <param name="M"></param>
/// <param name="kH"></param>
/// <param name="kW"></param>
/// <param name="nH"></param>
/// <param name="nW"></param>
/// <param name="stride"></param>
/// <param name="dilation"></param>
int conv2df(float* input, float* kernels, float* bias, float* output, int64_t C, int64_t H, int64_t W, int64_t M, int64_t kH, int64_t kW, int64_t nH, int64_t nW,int64_t* stride, int64_t* dilation);


#endif // !OPS_H
