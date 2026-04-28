#pragma once
#include "common.h"
#include <cudnn.h>

#pragma once

#include "common.h"
#include <cudnn.h>

// =========================================================
// Struct that holds everything for one CNN layer
// =========================================================
struct convLayer {
    // Descriptors
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t pool_out_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t relu_desc;
    cudnnPoolingDescriptor_t pool_desc;

    // Algorithm
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_bytes;

    // GPU data
    float* d_filter;
    float* d_bias;
    float* d_conv_out;
    float* d_pool_out;

    // Dimensions
    int out_n, out_c, out_h, out_w;
    int pool_n, pool_c, pool_h, pool_w;
};

convLayer create_layer(cudnnHandle_t cudnn, int in_n, int in_c, int in_h,
                       int in_w, int num_filters, int kernel_size,
                       cudnnTensorDescriptor_t input_desc);

void forward_layer(cudnnHandle_t cudnn, convLayer& layer, float* d_input,
                   void* d_workspace);

void destroy_layer(convLayer& layer);