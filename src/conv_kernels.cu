#include "conv_kernels.h"
#include <cmath>
#include <vector>
#include <cstdio>

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

void verify_conv_naive(cudnnHandle_t cudnn, convLayer& layer, float* d_input, void* d_workspace){
    int total_elements = layer.out_n * layer.out_c * layer.out_h * layer.out_w;
    size_t total_bytes = total_elements * sizeof(float);

    float* d_naive;
    float* d_cudnn;
    CHECK_CUDA(cudaMalloc(&d_naive, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_cudnn, total_bytes));

    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;
        
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        layer.input_desc,
        d_input,
        layer.filter_desc,
        layer.d_filter,
        layer.conv_desc,
        layer.algo,
        d_workspace,
        layer.workspace_bytes,
        &beta_overwrite,
        layer.output_desc,
        d_cudnn
    ));

    ConvDims d;
    d.N = layer.in_n;
    d.C = layer.in_c;
    d.H = layer.in_h;
    d.W = layer.in_w;
    d.K = layer.out_c;
    d.R = layer.kernel_size;
    d.S = layer.kernel_size;
    d.pad = layer.kernel_size / 2;

    launch_conv_naive(d_input, layer.d_filter, d_naive, d);
    
    CHECK_CUDA(cudaDeviceSynchronize());


    std::vector<float> h_cudnn(total_elements);
    std::vector<float> h_naive(total_elements);

    CHECK_CUDA(cudaMemcpy(h_cudnn.data(), d_cudnn, total_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_naive.data(), d_naive, total_bytes, cudaMemcpyDeviceToHost));

    const float threshold = 1e-3f;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int max_abs_idx = -1;
    int n_over = 0;

    for(int i = 0; i < total_elements; i++){
        const float a = h_cudnn[i];
        const float b = h_naive[i];

        const float abs_diff = fabsf(a-b);
        const float rel_diff = abs_diff / (fabsf(a) + fabsf(b) + 1e-8f);

        if(abs_diff > max_abs){
            max_abs = abs_diff;
            max_abs_idx = i;
        }
        if (rel_diff > max_rel) max_rel = rel_diff;
        if (abs_diff > threshold) n_over++;
    }

    printf("\n=== VERIFY conv naive vs cuDNN ===\n");
    printf("  shape        : [%d, %d, %d, %d]  (%d elements)\n",
           layer.out_n, layer.out_c, layer.out_h, layer.out_w, total_elements);
    printf("  max abs diff : %.3e  (at index %d)\n", max_abs, max_abs_idx);
    printf("  max rel diff : %.3e\n", max_rel);
    printf("  elems > %.0e : %d\n", threshold, n_over);
    printf("  verdict      : %s\n", (max_abs < threshold ? "PASS" : "FAIL"));

    CHECK_CUDA(cudaFree(d_cudnn));
    CHECK_CUDA(cudaFree(d_naive));
}

