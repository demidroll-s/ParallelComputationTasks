#pragma once

#include <iostream>

#define WARP_SIZE 32

/* Скалярное произведение двух векторов (с использованием разделяемой памяти) */
__global__ 
void KernelScalarMul(int n, float* x, float* y, float* res);
float SumOfSquares(float x);
void TestKernelScalarMul();