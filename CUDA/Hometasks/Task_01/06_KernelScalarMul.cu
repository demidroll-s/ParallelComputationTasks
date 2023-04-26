#include "05_KernelScalarMul.cuh"

/* Скалярное произведение двух векторов (с использованием разделяемой памяти) */
__global__ 
void KernelScalarMul(int n, float* x, float* y, float* res) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = x[index] * y[index];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_data[tid] += shared_data[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        res[blockIdx.x] = shared_data[0];
    }
}

float SumOfSquares(float x) {
    return x * (x + 1) * (2 * x + 1) / 6;
}

void TestKernelScalarMul() {
    int n = 1 << 18;
    const int block_size = 1024;
    const int num_blocks = n / block_size;
    
    float* h_x = new float[n];
    float* h_y = new float[n];
    float* h_result = new float[num_blocks];

    float* d_x;
    float* d_y;
    float* d_temp;
    float* d_result;

    int nbytes_array = n * sizeof(float);
    int nbytes_result = num_blocks * sizeof(float);

    cudaMalloc(&d_x, nbytes_array);
    cudaMalloc(&d_y, nbytes_array);
    cudaMalloc(&d_temp, nbytes_array);
    cudaMalloc(&d_result, nbytes_result);

    for (int i = 0; i < n; ++i) {
        h_x[i] = i;
        h_y[i] = 2 * i;
    }

    cudaMemcpy(d_x, h_x, nbytes_array, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, nbytes_array, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    KernelScalarMul<<<num_blocks, block_size, sizeof(float) * block_size>>>(n, d_x, d_y, d_result);

    cudaEventRecord(stop);

    cudaMemcpy(h_result, d_result, nbytes_result, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float sum = 0;
    for (int i = 0; i < num_blocks; ++i) {
		sum += h_result[i];
	}

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    std::cout << "Experiment computation: " << sum << std::endl;
    std::cout << "Real result: " << 2 * SumOfSquares(n - 1) << std::endl; 
    
    std::cout << "Time elapsed: " << time_elapsed << " ms" << std::endl;
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    
    delete[] h_x;
    delete[] h_y;
    delete[] h_result;
}