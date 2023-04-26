#pragma once

#include <iostream>

/* Поэлементная сумма двух векторов */
__global__ 
void KernelAdd(int n, float* x, float* y, float* result);
void TestKernelAdd();