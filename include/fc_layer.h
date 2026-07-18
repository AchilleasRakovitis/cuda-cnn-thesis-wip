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

    //Gradients
    float* d_grad_weights; //dL/dW [O, I] -same shape as d_weights
    float* d_grad_bias; // dL/dB [O] -same shape as d_bias
    float* d_grad_input; // dL/dX[N, I] -same shape as fcLayer input
    float* d_grad_preact; //dL/d(pre-activation) [N, O] ReLU backward output

    //Dimensions
    int in_features, out_features, batch_size;
    
    //Configuration
    bool apply_relu;
};

fcLayer create_fc_layer(cudnnHandle_t cudnn, int in_features, int out_features,
                        int batch_size, bool apply_relu, unsigned seed);

void forward_fc_layer(cudnnHandle_t cudnn, cublasHandle_t cublas, fcLayer& layer,
                      float* d_input);

void backward_fc_layer(cudnnHandle_t cudnn, cublasHandle_t cublas, fcLayer& layer,
                       float* d_input, float* d_grad_output);

            
void update_fc_layer(fcLayer& layer, float lr);

void destroy_fc_layer(fcLayer& layer);