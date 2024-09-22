#ifndef SHAPE_CALC_H
#define SHAPE_CALC_H
#include "tensor.h"
#include "linkedlist.h"
#include "utility.h"
#include "define.h"

int set_broadcast_shape(struct tensor* A, struct tensor* B, struct tensor* C);
int set_squeeze_shape(struct tensor* data, struct tensor* axes, struct tensor* squeezed);
int set_unsqueeze_shape(struct tensor* data, struct tensor* axes, struct tensor* expanded);
int set_transpose_shape(struct tensor* data,  int64_t* perm, struct tensor* transposed);
int set_matmul_shape(struct tensor* a, struct tensor* b, struct tensor* c);
int set_slice_shape(struct tensor* data,  int64_t* starts,  int64_t* ends,  int64_t* axes, int64_t* steps, struct tensor* output);
int set_gemm_shape(struct tensor* a, struct tensor* B, int64_t* transA, int64_t* transB, struct tensor* y);
int set_concat_shape(int64_t* axis, struct list* inputs, struct tensor* concat_result);
int set_split_shape(int64_t axis, int64_t num_outputs, struct tensor* input, int64_t* split, struct list* outputs);
int set_reshaped_shape(struct tensor* input, struct tensor* shape, struct tensor* reshaped);
int set_pad_shape(struct tensor* data, struct tensor* pads, struct tensor* axes,  struct tensor* output);
int set_conv_shape(struct tensor* x, struct tensor* w, struct tensor* y, int64_t* pads, int64_t* strides);
//int set_zeropad_shape(int64_t* ceilmode ,struct tensor* x, struct tensor* y, int64_t* kernel_shape,int64_t* dilation, int64_t* pads, int64_t* strides);
int set_lstm_shape(struct tensor* x, int64_t num_direction, int64_t hidden_size, struct tensor* y, struct tensor* y_h, struct tensor* y_c);
int set_reducemean_shape(struct tensor* axes, int64_t* noop_with_empty_axes, int64_t keepdims, struct tensor* data, struct tensor* reduced);
#endif // SHAPE_CALC_H
