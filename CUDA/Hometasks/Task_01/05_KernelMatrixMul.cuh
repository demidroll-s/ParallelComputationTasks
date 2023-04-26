#pragma once

#include <iostream>
#include <random>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int WARM_UP = 100;

/* Произведение матриц (наивный подход) */
__global__ 
void KernelMatrixMul(int m, int n, int p, const float* matrix_a, const float* matrix_b, float* result);

/* Произведение матриц (с использованием stride) */
__global__ 
void KernelMatrixMulStride(int m, int n, int p, const float* matrix_a, const float* matrix_b, float* result);

/* Произведение матриц, выполненное на хосте */
void MatrixMulHost(int m, int n, int p, const float* matrix_a, const float* matrix_b, float* result);

/* Вывод матрицы на экран */
void PrintMatrix(int height, int width, const float* matrix);

/* Проверка результата работы функции */
void CheckResults(int n, float* ref, float* result, float time_elapsed, const std::string& function_name);

void TestKernelMatrixMul();