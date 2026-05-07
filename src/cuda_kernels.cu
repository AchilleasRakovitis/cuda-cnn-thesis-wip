#include "cuda_kernels.h"
#include <stdint.h>

__global__ void add_bias_kernel(float* Y, const float* b, int N, int O){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = O * N;
    if(idx < total){
        int col = idx % O;
        Y[idx] += b[col];
    }
}

__global__ void nll_kernel(const float* d_logprobs ,const uint8_t* d_labels, 
                            float* d_losses, const int batch_size, const int num_classes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size){
        int label = d_labels[idx];
        float logp = d_logprobs[idx * num_classes + label];
        float loss = -logp;
        d_losses[idx] = loss;
    }
}