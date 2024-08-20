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
	OPS_NOT_BROADCAST,
	OPS_NO_OUTPUT_SHAPE
};


enum NODE_TYPE {
	UNDEFINED,
	Transpose,		// 1
	Slice,			// 1
	Squeeze,		// 1
	LSTM,			// 1
	Concat,			// 1
	MatMul,			// 1
	Unsqueeze,		// 1
	Conv,			// 1
	ReduceMean,		// 1
	Sub,			// 1
	Mul,			// 1
	Add,			// 1
	Sqrt,			// 1
	Div,			// 1
	Split,			// 1
	Tanh,			// 1
	Sigmoid,		// 1
	Pad,			// 1
	Gemm,			// 1
	Reshape,		// 1
	Constant		// 1


}typedef NODE_TYPE;


//enum datatype {
//	UKNOWN,
//	INT32,
//	INT64,
//	FLOAT32
//};

#endif // !DEFINE_H
