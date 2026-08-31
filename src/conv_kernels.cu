#include "conv_kernels.h"

__global__ void conv_forward_naive_kernel(const float* __restrict__ input, const float* __restrict__ filter,
                                          float* __restrict__ output, ConvDims d){

        int q = blockIdx.x * blockDim.x + threadIdx.x;
        int p = blockIdx.y * blockDim.y + threadIdx.y;
        int n = blockIdx.z / d.K;
        int k = blockIdx.z % d.K;

        if(p >= d.H ||q>= d.W ) return;

        float acc = 0;
        for(int c = 0; c <= d.C - 1; c++){
            for(int r = 0; r <= d.R - 1; r++){
                for(int s = 0; s <= d.S - 1; s++){
                    int ih = p + r - d.pad;
                    int iw = q + s - d.pad;
                    if(ih >= 0 && ih < d.H && iw >= 0 && iw < d.W){
                        int in_idx = ((n*d.C + c)*d.H + ih)*d.W + iw;
                        int flt_idx = ((k*d.C + c)*d.R + r)*d.S + s;
                        acc += input[in_idx] * filter[flt_idx];        
                    }
                }
            }
        }

        int out_idx = ((n*d.K + k)*d.H + p)*d.W + q;
        output[out_idx] = acc;
}

void launch_conv_naive(const float* d_input, const float* d_filter, float* d_output, 
                        const ConvDims& d){
    dim3 block(16, 16);
    dim3 grid( (d.W + 15) / 16, (d.H + 15) / 16, d.N * d.K);

    conv_forward_naive_kernel<<<grid, block>>>(d_input, d_filter, d_output, d);
    
}

