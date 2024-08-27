#ifndef UTILITY_H
#define UTILITY_H
#include "stdio.h"
#include "stdint.h"
void swap(void* a, void* b);
void safe_free(void** p);
void swapf(float* a, float* b);
void print_int64_t(int64_t* arr, int len);

#endif UTILITY_H
