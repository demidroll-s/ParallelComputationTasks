#pragma once

#include <iostream>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int WARM_UP = 100;

/* Транспонирование матрицы (наивный подход) */
__global__ 
void KernelNaiveTranspose(const float* matrix, float* result);

/* Транспонирование матрицы (с использованием разделяемой памяти) */
__global__ 
void KernelCoalesced(const float* matrix, float* result);

/* Транспонирование матрицы (с использованием разделяемой памяти, разрешение банк-конфликтов) */
__global__ 
void KernelNoBankConficts(float* matrix, float* result);

/* Проверка результата работы функции */
void CheckResults(int n, float* ref, float* result, float time_elapsed, const std::string& function_name);

void TestKernelMatrixTranspose();