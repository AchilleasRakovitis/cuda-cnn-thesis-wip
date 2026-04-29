#pragma once

#include "common.h"
#include <cudnn.h>
#include <cublas_v2.h>

struct fcLayer{
    //Descriptors
    cudnnTensorDescriptor_t output_desc;
    cudnnActivationDescriptor_t relu_desc;

    //GPU Data
    float* d_bias;
    float* d_weights;
    float* d_output;

    //Dimensions
    int in_features, out_features, batch_size;
    
    //Configuration
    bool apply_relu;
};

fcLayer create_fc_layer(cudnnHandle_t cudnn, int in_features, int out_features,
                        int batch_size, bool apply_relu);

void forward_fc_layer(cudnnHandle_t cudnn, cublasHandle_t cublas, fcLayer& layer,
                      float* d_input);

void destroy_fc_layer(fcLayer& layer);