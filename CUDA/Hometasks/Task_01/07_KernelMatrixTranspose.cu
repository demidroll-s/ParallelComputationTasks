#include "07_KernelMatrixTranspose.cuh"

/* Транспонирование матрицы (наивный подход) */
__global__ 
void KernelNaiveTranspose(const float* matrix, float* result) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        result[x * width + (y + j)] = matrix[(y + j) * width + x];
}

/* Транспонирование матрицы (с использованием разделяемой памяти) */
__global__ 
void KernelCoalesced(const float* matrix, float* result) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        result[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

/* Транспонирование матрицы (с использованием разделяемой памяти, разрешение банк-конфликтов) */
__global__ 
void KernelNoBankConficts(float* matrix, float* result) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        result[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
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

void TestKernelMatrixTranspose() {
    int n_x = 1 << 10;
    int n_y = 1 << 10;
    int n_total = n_x * n_y;

    int nbytes = n_total * sizeof(float);

    float *h_matrix = (float*) malloc(nbytes);
    float *h_result = (float*) malloc(nbytes);
    float *check_result = (float*) malloc(nbytes);

    float *d_matrix, *d_result;
    cudaMalloc(&d_matrix, nbytes);
    cudaMalloc(&d_result, nbytes);

    dim3 grid_dim(n_x / TILE_DIM, n_y / TILE_DIM, 1);
    dim3 block_dim(TILE_DIM, BLOCK_ROWS, 1);

    if (n_x % TILE_DIM || n_y % TILE_DIM) {
        std::cout << "n_x and n_y must be a multiple of TILE_DIM" << std::endl;
        return;
    }

    if (TILE_DIM % BLOCK_ROWS) {
        std::cout << "TILE_DIM must be a multiple of BLOCK_ROWS" << std::endl;\
        return;
    }

    /* Инициализация входной матрицы */
    for (int j = 0; j < n_y; ++j)
        for (int i = 0; i < n_x; ++i)
            h_matrix[j * n_x + i] = j * n_x + i;

    /* Инициализация проверочной матрицы */
    for (int j = 0; j < n_y; ++j)
        for (int i = 0; i < n_x; ++i)
            check_result[j * n_x + i] = h_matrix[i * n_x + j];

    cudaMemcpy(d_matrix, h_matrix, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time_elapsed;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Наивный подход */
    {
        cudaMemset(d_result, 0.0, nbytes);
        
        KernelNaiveTranspose<<<grid_dim, block_dim>>>(d_matrix, d_result);
        cudaEventRecord(start, 0);
        for (int i = 0; i < WARM_UP; ++i)
            KernelNaiveTranspose<<<grid_dim, block_dim>>>(d_matrix, d_result);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 
        cudaEventElapsedTime(&time_elapsed, start, stop);
        cudaMemcpy(h_result, d_result, nbytes, cudaMemcpyDeviceToHost);

        CheckResults(n_total, check_result, h_result, time_elapsed, "KernelNaiveTranspose");
    }
    
    /* с использованием разделяемой памяти */
    {
        cudaMemset(d_result, 0.0, nbytes);
        
        KernelCoalesced<<<grid_dim, block_dim>>>(d_matrix, d_result);
        cudaEventRecord(start, 0);
        for (int i = 0; i < WARM_UP; ++i)
            KernelCoalesced<<<grid_dim, block_dim>>>(d_matrix, d_result);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 
        cudaEventElapsedTime(&time_elapsed, start, stop);
        cudaMemcpy(h_result, d_result, nbytes, cudaMemcpyDeviceToHost);

        CheckResults(n_total, check_result, h_result, time_elapsed, "KernelCoalesced");
    }

    /* с использованием разделяемой памяти, разрешение банк-конфликтов) */
    {
        cudaMemset(d_result, 0.0, nbytes);
        
        KernelNoBankConficts<<<grid_dim, block_dim>>>(d_matrix, d_result);
        cudaEventRecord(start, 0);
        for (int i = 0; i < WARM_UP; ++i)
            KernelNoBankConficts<<<grid_dim, block_dim>>>(d_matrix, d_result);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 
        cudaEventElapsedTime(&time_elapsed, start, stop);
        cudaMemcpy(h_result, d_result, nbytes, cudaMemcpyDeviceToHost);

        CheckResults(n_total, check_result, h_result, time_elapsed, "KernelNoBankConflicts");
    }
    
    cudaFree(d_matrix);
    cudaFree(d_result);

    free(h_matrix);
    free(h_result);
    free(check_result);
    
    return;
}