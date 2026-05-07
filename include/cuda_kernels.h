#pragma once

#include <stdint.h>

__global__ void add_bias_kernel(float*, const float*, int, int);

__global__ void nll_kernel(const float* d_logprobs, const uint8_t* d_labels,
                            float* d_losses, int batch_size, int num_classes);