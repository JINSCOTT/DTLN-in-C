#ifndef DEFINE_H
#define DEFINE_H

#define DEBUG
//#define OMP
//#define ONE_MKL
#define NULL 0
#define true 1
#define false 0

enum OPS_FUNCTION_RETURN {
	OPS_UNDEFINED,
	OPS_SUCCESS,
	OPS_INPUT_IS_NULL,
	OPS_NOT_BROADCASTABLE,
	OPS_DIMENSION_MISMATCH,
	OPS_ALLOCATION_FAIL,
	OPS_TYPE_UNIMPLEMENTED,
	OPS_TYPE_NOT_SUPPORTED,
	OPS_INVALID_ARGUMENT,
	OPS_NO_OUTPUT_SHAPE
};


enum NODE_TYPE {
	UNDEFINED,
	Transpose,		// 1. tested
	Slice,			// 1. tested
	Squeeze,		// 1. tested
	LSTM,			// 1
	Concat,			// 1. tested
	MatMul,			// 1, pass
	Unsqueeze,		// 1, pass
	Conv,			// 1. tested
	ReduceMean,		// 1, tested
	Sub,			// 1, tested
	Mul,			// 1, pass
	Add,			// 1, pass
	Sqrt,			// 1, pass
	Div,			// 1, pass
	Split,			// 1, pass
	Tanh,			// 1. tested
	Sigmoid,		// 1. tested
	Pad,			// 1
	Gemm,			// 1
	Reshape,		// 1
	Constant		// 1. seems to work


}typedef NODE_TYPE;


enum datatype {
	DATATYPE_UKNOWN,
	DATATYPE_INT32,
	DATATYPE_INT64,
	DATATYPE_FLOAT32
};

#endif // !DEFINE_H
