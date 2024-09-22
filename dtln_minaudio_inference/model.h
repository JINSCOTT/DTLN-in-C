/*
* Definition for model
*/
// Please directly manipulate data from array
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

//Create empty model 
struct model* create_model();

/// <summary>
/// Inference model
/// </summary>
/// <param name="m">Pointer to model</param>
/// <returns>0, For fail.1, for success.</returns>
int inference_model(struct model* m);

#ifdef DEBUG
void print_model(struct model* model);
#endif // DEBUG

#endif // MODEL_H









