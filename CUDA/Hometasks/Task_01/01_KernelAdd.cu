#include "01_KernelAdd.cuh"

/* Поэлементная сумма двух векторов */
__global__ 
void KernelAdd(int n, float* x, float* y, float* result) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < n; index += stride) {
        result[index] = x[index] + y[index];
    }
}

void TestKernelAdd() {
    int n = 1 << 18;
    
    float* h_x = new float[n];
    float* h_y = new float[n];
    float* h_result = new float[n];

    float* d_x;
    float* d_y;
    float* d_result;

    int nbytes = n * sizeof(float);
    cudaMalloc(&d_x, nbytes);
    cudaMalloc(&d_y, nbytes);
    cudaMalloc(&d_result, nbytes);

    for (int i = 0; i < n; ++i) {
        h_x[i] = i;
        h_y[i] = 3.0;
    }

    cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    KernelAdd<<<4, 256>>>(n, d_x, d_y, d_result);

    cudaEventRecord(stop);

    cudaMemcpy(h_result, d_result, nbytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    std::cout << h_result[n - 1] << std::endl; 

    std::cout << "Time elapsed: " << time_elapsed << " ms" << std::endl;
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    
    delete[] h_x;
    delete[] h_y;
    delete[] h_result;
}