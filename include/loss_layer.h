#pragma once

#include "common.h"
#include <cudnn.h>

struct lossLayer{
    //Descriptors
    cudnnTensorDescriptor_t logits_desc;

    //Dimensions
    int batch_size;
    int num_classes;

    //GPU buffers
    float* d_logprobs;
    float* d_losses_per_sample;
    float* d_final_loss;
    float* d_grad_logits; // dL/dLogits [batch, classes]
};

lossLayer create_loss_layer(cudnnHandle_t cudnn, int batch_size, int num_classes);

void forward_loss_layer(cudnnHandle_t cudnn, lossLayer& layer, float* d_logits,
                        uint8_t* d_labels);

void backward_loss_layer(lossLayer& layer, uint8_t* d_labels);

void destroy_loss_layer(lossLayer& layer);

