#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

__global__ void add_bias_kernel(float* output, const float* bias, int batch_size, int out_features);

__global__ void nll_kernel(const float* d_logprobs, const uint8_t* d_labels,
                            float* d_losses, int batch_size, int num_classes);

__global__ void mean_reduce_kernel(const float* in, float* out_sum, int batch_size);


__global__ void loss_backward_kernel(const float* d_logprobs, const uint8_t* d_labels,
                                     float* d_grad_logits, int batch_size, int num_classes);

__global__ void bias_backward_kernel(const float* d_grad_output, float* d_grad_bias,
                                     int batch_size, int out_features);