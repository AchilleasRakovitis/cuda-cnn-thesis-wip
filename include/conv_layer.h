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

    //Backward Algorithm selection same logic as forward
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    size_t bwd_filter_workspace_bytes;
    size_t bwd_data_workspace_bytes;

    // GPU data
    float* d_filter;
    float* d_bias;
    float* d_conv_out;
    float* d_pool_out;

    //Gradient buffers
    //One Buffer covers all intermediate gradients at the conv_output, pooling
    //backward writes it, ReLU backward masks it, conv backward reads it, its 
    //safe to use the same buffer unlike in the fc layer that i used a new preact buffer
    //because this buffers is owned by this layer and each stage consumes the previous one.
    float* d_grad_conv_out; // [N, K, H, W] gradient at the conv output
    float* d_grad_bias; // [K] same shape as d_bias
    float* d_grad_filter;   // [K, C, R, S] same shape as d_filter   
    float* d_grad_input;    // [N, C, H_in, W_in] - travels back at the previous layer

    //Dimentions needed for d_grad_input
    int in_n, in_c, in_h, in_w;

    // Dimensions
    int out_n, out_c, out_h, out_w;
    int pool_n, pool_c, pool_h, pool_w;
};

convLayer create_layer(cudnnHandle_t cudnn, int in_n, int in_c, int in_h,
                       int in_w, int num_filters, int kernel_size,
                       cudnnTensorDescriptor_t input_desc, unsigned seed);

void forward_layer(cudnnHandle_t cudnn, convLayer& layer, float* d_input,
                   void* d_workspace);

void backward_conv_layer(cudnnHandle_t cudnn, convLayer& layer, float* d_input, 
                         float* d_grad_pool_out, void* d_workspace);

void destroy_layer(convLayer& layer);