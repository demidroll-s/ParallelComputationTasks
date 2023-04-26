#pragma once

#include <iostream>

/* Поэлементное произведение двух векторов */
__global__ 
void KernelMul(int n, float* x, float* y, float* res);
void TestKernelMul();