#ifndef DEFINE_H
#define DEFINE_H

//#define DEBUG
//#define OMP
//#define ONE_MKL



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
	UNDEFINED,
	Transpose,		
	Slice,		
	Squeeze,	
	Concat,		
	MatMul,
	Unsqueeze,		
	Conv,
	ReduceMean,		
	Sub,
	Mul,
	Add,			
	Sqrt,		
	Div,		
	Split,		
	Tanh,			
	Sigmoid,		
	Pad,			
	Gemm,			
	Reshape,		
	Constant,		
	Relu,
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
	RELU


}typedef NODE_TYPE;

/// <summary>
///  Data type of operation.
/// </summary>
enum DATATYPE {
	DATATYPE_UKNOWN,
	DATATYPE_INT32,
	DATATYPE_INT64,
	DATATYPE_FLOAT32
};

#endif // !DEFINE_H
