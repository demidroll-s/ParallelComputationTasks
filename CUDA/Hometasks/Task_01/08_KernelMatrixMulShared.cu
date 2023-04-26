#include "08_KernelMatrixMulShared.cuh"

/* Произведение матриц (с использованием разделяемой памяти) */
__global__ 
void KernelMatrixMulShared(int m, int n, int p, const float* matrix_a, const float* matrix_b, float* result) {
    __shared__ float tile_a[TILE_DIM][TILE_DIM];
    __shared__ float tile_b[TILE_DIM][TILE_DIM];

    float sum = 0.0;

    for (int tile_idx = 0; tile_idx < ceilf(static_cast<float>(n) / TILE_DIM); ++tile_idx) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = tile_idx * blockDim.x + threadIdx.x;

        if (row < m && col < n)
            tile_a[threadIdx.y][threadIdx.x] = matrix_a[row * n + col];
        else
            tile_a[threadIdx.y][threadIdx.x] = 0.0;

        row = tile_idx * blockDim.y + threadIdx.y;
        col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < n && col < p)
            tile_b[threadIdx.y][threadIdx.x] = matrix_b[row * p + col];
        else
            tile_b[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];

        __syncthreads();
    }

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p)
        result[row * p + col] = sum;    
}

/* Произведение матриц, выполненное на хосте */
void MatrixMulHost(int m, int n, int p, const float* matrix_a, const float* matrix_b, float* result) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            for (int k = 0; k < n; ++k)
                result[i * p + j] += matrix_a[i * n + k] * matrix_b[k * p + j];
}

/* Вывод матрицы на экран */
void PrintMatrix(int height, int width, const float* matrix) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j)
			std::cout << matrix[i * width + j] << " ";
        std::cout << std::endl;
	}
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

void TestKernelMatrixMulShared() {
    int rows_a = 1 << 10;
    int cols_a = 1 << 10;
    int rows_b = 1 << 10;
    int cols_b = 1 << 10;

    if (cols_a != rows_b) {
        std::cout << "The number of rows in matrix A should be equal to number of rows in matrix B" << std::endl;
        return;
    }

    int n_total_a = rows_a * cols_a;
    int n_total_b = rows_b * cols_b;
    int n_total_result = rows_a * cols_b;

    float* h_matrix_a = new float[n_total_a];
    float* h_matrix_b = new float[n_total_b];
    float* h_result = new float[n_total_result];
    float* check_result = new float[n_total_result];

    float *d_matrix_a, *d_matrix_b, *d_result;

    cudaMalloc(&d_matrix_a, n_total_a * sizeof(float));
    cudaMalloc(&d_matrix_b, n_total_b * sizeof(float));
    cudaMalloc(&d_result, n_total_result * sizeof(float));

    dim3 threads_per_block(TILE_DIM, TILE_DIM);
    dim3 blocks_per_grid(1, 1);

    blocks_per_grid.x = std::ceil(static_cast<double>(cols_b) / static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(rows_a) / static_cast<double>(threads_per_block.y));

    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(-256, 256);

    /* Инициализация матриц */
    for (int i = 0; i < rows_a; ++i)
        for (int j = 0; j < cols_a; ++j)
            h_matrix_a[i * cols_a + j] = static_cast<float>(uniform_dist(e));

    for (int i = 0; i < rows_b; ++i)
        for (int j = 0; j < cols_b; ++j)
            h_matrix_b[i * cols_b + j] = static_cast<float>(uniform_dist(e));

    MatrixMulHost(rows_a, cols_a, cols_b, h_matrix_a, h_matrix_b, check_result);

    cudaMemcpy(d_matrix_a, h_matrix_a, n_total_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, h_matrix_b, n_total_b * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time_elapsed;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* с использованием разделяемой памяти */
    {
        cudaMemset(d_result, 0.0, n_total_result * sizeof(float));
        
        KernelMatrixMulShared<<<blocks_per_grid, threads_per_block>>>(rows_a, cols_a, cols_b, 
                d_matrix_a, d_matrix_b, d_result);

        cudaEventRecord(start, 0);
        
        for (int i = 0; i < WARM_UP; ++i)
            KernelMatrixMulShared<<<blocks_per_grid, threads_per_block>>>(rows_a, cols_a, cols_b, 
                d_matrix_a, d_matrix_b, d_result);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 
        cudaEventElapsedTime(&time_elapsed, start, stop);
        cudaMemcpy(h_result, d_result, n_total_result * sizeof(float), cudaMemcpyDeviceToHost);

        CheckResults(n_total_result, check_result, h_result, time_elapsed, "KernelMatrixMulShared");
    }

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_result);

    delete[] h_matrix_a;
    delete[] h_matrix_b;
    delete[] h_result;
    delete[] check_result;
}