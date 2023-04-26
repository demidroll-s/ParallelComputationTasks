#include "04_KernelMatrixVectorMul.cuh"

/* Произведение матрицы на вектор */
__global__ 
void KernelMatrixVectorMul(int m, int n, const float* matrix, const float* vector, float* result) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_x + gridDim.x * TILE_DIM * tid_y;

    if (tid >= m)
        return;

    result[tid] = 0.0;

    for (int j = 0; j < n; ++j)
        result[tid] += matrix[tid * n + j] * vector[j];
}

/* Произведение матрицы на вектор, выполненное на хосте */
void MatrixMulVectorHost(int m, int n, const float* matrix, const float* vector, float* result) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            result[i] += matrix[i * n + j] * vector[j];
}

/* Вывод вектора на экран */
void PrintVector(int length, float* vector) {
    for (int i = 0; i < length; ++i)
        std::cout << vector[i] << " ";
    std::cout << std::endl;
}

/* Проверка результата работы функции */
void CheckResults(int n, float* ref, float* result, float time_elapsed, const std::string& function_name) {
    for (int i = 0; i < n; ++i) {
        if (result[i] != ref[i]) {
            std::cout << "Failed test " << function_name << "!" << std::endl;
            return;
        }
    }
        
    std::cout << "Test " << function_name << " OK!" << std::endl;
    std::cout << "Time elapsed : " << time_elapsed / WARM_UP << " ms" << std::endl;
}

void TestKernelMatrixVectorMul() {
    int n_x = 1 << 10;
    int n_y = 1 << 10;

    int n_matrix = n_x * n_y;
    int n_vector = n_y;
    int n_result = n_x;

    float* h_matrix = new float[n_matrix];
    float* h_vector = new float[n_vector];
    float* h_result = new float[n_result];
    float *check_result = new float[n_result];

    float *d_matrix, *d_vector, *d_result;

    cudaMalloc(&d_matrix, n_matrix * sizeof(float));
    cudaMalloc(&d_vector, n_vector * sizeof(float));
    cudaMalloc(&d_result, n_result * sizeof(float));

    dim3 threads_per_block(TILE_DIM, TILE_DIM);
    dim3 blocks_per_grid(1, 1);

    blocks_per_grid.x = 1;
    blocks_per_grid.y = std::ceil(static_cast<double>(n_x) / static_cast<double>(threads_per_block.y));

    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(-256, 256);

    /* Инициализация матриц */
    for (int i = 0; i < n_x; ++i)
        for (int j = 0; j < n_y; ++j)
            h_matrix[i * n_x + j] = static_cast<float>(uniform_dist(e));

    for (int i = 0; i < n_y; ++i)
        h_vector[i] = static_cast<float>(uniform_dist(e));

    MatrixMulVectorHost(n_x, n_y, h_matrix, h_vector, check_result);

    cudaMemcpy(d_matrix, h_matrix, n_matrix * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, n_vector * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time_elapsed;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    {
        cudaMemset(d_result, 0.0, n_result * sizeof(float));
        
        KernelMatrixVectorMul<<<blocks_per_grid, threads_per_block>>>(n_x, n_y, d_matrix, d_vector, d_result);

        cudaEventRecord(start, 0);
        
        for (int i = 0; i < WARM_UP; ++i)
            KernelMatrixVectorMul<<<blocks_per_grid, threads_per_block>>>(n_x, n_y, d_matrix, d_vector, d_result);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 
        cudaEventElapsedTime(&time_elapsed, start, stop);
        cudaMemcpy(h_result, d_result, n_result * sizeof(float), cudaMemcpyDeviceToHost);

        CheckResults(n_result, check_result, h_result, time_elapsed, "KernelMatrixVectorMul");
    }
    
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    delete[] h_matrix;
    delete[] h_vector;
    delete[] h_result;
    delete[] check_result;
}