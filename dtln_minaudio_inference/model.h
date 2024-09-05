/*
* Definition for model
*/

#ifndef MODEL_H
#define MODEL_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "define.h"
#include "node.h"

struct model {
	struct list node;
	struct list tensor;

};

struct model* create_model();


// Please directly manipulate data from array
/*
//  copied into, free to edlete
int16_t set_model_tensor_data(struct model* m, int64_t index, void* data, int64_t datasize);
// Do not free data got
int16_t get_model_tensor_data(struct model* m, int64_t index, void** data);
*/

/// <summary>
/// Inference model
/// </summary>
/// <param name="m"></param>
/// <returns>0, For fail.1, for success.</returns>
int inference_model(struct model* m);

#ifdef DEBUG
void print_model(struct model* model);
#endif // DEBUG

#endif // MODEL_H









