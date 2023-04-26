//#include "01_KernelAdd.cuh"
//#include "02_KernelMul.cuh"
//#include "04_KernelMatrixVectorMul.cuh"
//#include "05_KernelMatrixMul.cuh"
//#include "06_KernelScalarMul.cuh"
//#include "07_KernelMatrixTranspose.cuh"
#include "08_KernelMatrixMulShared.cuh"

int main() {
    TestKernelMatrixMulShared();
    return 0;
}