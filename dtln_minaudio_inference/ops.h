// array level calculation
#ifndef OPS_H
#define OPS_H
#include "linkedlist.h"
#include <float.h>
#include <string.h>
#include <math.h>
#include "utility.h"
#include "define.h"
#include "tensor.h"
#ifdef ONE_MKL	// One mkl header
#include "mkl.h"
#endif

/*----------------------------Array Function-----------------------------*/

// Broadcast function
int add_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type);
int sub_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type);
int mul_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type);
int div_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type);
//int and_array(const void* a, const void* b, void* c, const int64_t a_size, const int64_t b_size, const int64_t c_size, const DATATYPE type);
// One to One
int sigmoid_array(const void* x, void* y,  const int64_t size, const DATATYPE type);
int relu_array(const void* x, void* y,const int64_t size, const DATATYPE type);
int abs_array(const void* x, void* y, const int64_t size, const DATATYPE type);
int acos_array(const void* x, void* y, const int64_t size, const DATATYPE type);
int acosh_array(const void* x, void* y,  const int64_t size, const DATATYPE type);
int tanh_array(const void* x, void* y, const int64_t size, const DATATYPE type);
int atan_array(const void* x, void* y, const int64_t size, const DATATYPE type);
int atanh_array(const void* x, void* y, const int64_t size, const DATATYPE type);
int asin_array(const void* x, void* y, const int64_t size, const DATATYPE type);
int asinh_array(const void* x, void* y, const int64_t size, const DATATYPE type);


int clip_array(const void* input, const void* min, const void* max, void* output, const int64_t size, const DATATYPE type);
// Matmul
/// <param name="A">n*m</param>
/// <param name="B">m*p</param>
/// <param name="C">n*p</param>
int matmul_array(const void* A, const void* B, void* C, int64_t n, int64_t m, int64_t p, const DATATYPE type);
int mean_array(const void* array, void* result, int64_t num_elements, const DATATYPE type);



int transpose_2d(const void* input, void* output, int64_t n, int64_t m , const DATATYPE type);
/*----------------------------Phase out----------------------------------*/

int addf_array(float* a, float* b, float* c, int64_t a_size, int64_t b_size, int64_t c_size);
int mulf_array(float* a, float* b, float* c, int64_t a_size, int64_t b_size, int64_t c_size);
int tanhf_array(float* input, float* output, int64_t size);
int sigmoidf_array(float* input, float* output, int64_t size);
int reluf_array(float* input, float* output, int64_t size);
int transposef_2d(float* x, int64_t n, int64_t m);
/*----------------------------FLOAT FUNCTIONS----------------------------*/



/// <summary>
/// row major gemm
/// </summary>
/// <param name="transA"></param>
/// <param name="transB"></param>
/// <param name="m"></param>
/// <param name="n"></param>
/// <param name="k"></param>
/// <param name="alpha"></param>
/// <param name="a"></param>
/// <param name="lda"></param>
/// <param name="b"></param>
/// <param name="ldb"></param>
/// <param name="beta"></param>
/// <param name="c"></param>
/// <param name="ldc"></param>
void xgemm(const int transA,const int transB,  const int64_t m, const int64_t n, const int64_t k, const float alpha, void* a, const int64_t lda, void* b, const int64_t ldb, const float beta, void* c, const int64_t ldc, const DATATYPE type);

int activation_array(const void* input, void* output, char* activation, int64_t size, float* alpha, float* beta, const DATATYPE type);

int activationf_array(float* input, float* output, char* type, int64_t size, float alpha, float beta);


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
int conv2df(float* input, float* kernels, float* bias, float* output, int64_t C, int64_t H, int64_t W, int64_t M, int64_t kH, int64_t kW, int64_t nH, int64_t nW, int64_t* stride, int64_t* dilation);


#endif // !OPS_H
