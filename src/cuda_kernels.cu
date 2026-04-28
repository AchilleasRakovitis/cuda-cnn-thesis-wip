#include "cuda_kernels.h"

__global__ void add_bias_kernel(float* Y, const float* b, int N, int O){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = O * N;
    if(idx < total){
        int col = idx % O;
        Y[idx] += b[col];
    }
}