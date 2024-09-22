#ifndef DEFINE_H
#define DEFINE_H

//#define DEBUG
//#define OMP
#define ONE_MKL

// Assume shape will be stable during inference
// In other words the dimension will always be the same
#define ASSUME_SHAPE_STABLE

#ifndef NULL
#define NULL 0
#endif // !NULL
#ifndef true
#define true 1
#endif // !true

#ifndef false
#define false 0
#endif // !false


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
	// dtln operators
	UNDEFINED,
	TRANSPOSE,
	SLICE,
	SQUEEZE,
	LSTM,
	CONCAT,
	MATMUL,
	UNSQUEEZE,
	CONV,
	REDUCEMEAN,
	SUB,
	MUL,
	ADD,
	SQRT,
	DIV,
	SPLIT,
	TANH,
	SIGMOID,
	PAD,
	GEMM,
	RESHAPE,
	CONSTANT,
	RELU,
	//tested
	CLIP,
	ARGMAX,
	// untested
	ARGMIN,// Very close to argmax
	ABS,
	ACOS,
	ACOSH,
	ATAN,
	ATANH,
	ASIN,
	ASINH,
	// Implemented
	AVERAGEPOOL,
	// not panned to implement
	AND // bool type not supported
}typedef NODE_TYPE;

/// <summary>
///  Data type of operation.
/// </summary>
enum DATATYPE {
	DATATYPE_UKNOWN,
	DATATYPE_INT32,
	DATATYPE_INT64,
	DATATYPE_FLOAT,
	DATATYPE_DOUBLE	// New added, not integrated yet
}typedef DATATYPE;

#endif // !DEFINE_H
