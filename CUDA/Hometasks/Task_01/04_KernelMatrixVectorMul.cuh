#pragma once

#include <iostream>
#include <random>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int WARM_UP = 100;

/* Произведение матрицы на вектор */
__global__ 
void KernelMatrixVectorMul(int m, int n, const float* matrix, const float* vector, float* result);

/* Произведение матрицы на вектор, выполненное на хосте */
void MatrixMulVectorHost(int m, int n, const float* matrix, const float* vector, float* result);

/* Вывод вектора на экран */
void PrintVector(int length, float* vector);

/* Проверка результата работы функции */
void CheckResults(int n, float* ref, float* result, float time_elapsed, const std::string& function_name);

void TestKernelMatrixVectorMul();