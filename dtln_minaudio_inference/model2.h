#ifndef MODEL__H
#define MODEL__H
#include "weight2.h"
#include "model.h"
#include "node.h"
// Creates tensors for the model
// If return is 0, function fail. Success if return is 1
int create_model2_onnx_tensor() ;
int create_model2_onnx_attributes(); 
int setup_model2_onnx( struct model* m);
#endif //MODEL__H
