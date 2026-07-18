#pragma once
#include "conv_layer.h"
#include "fc_layer.h"
#include "loss_layer.h"

// Finite-difference gradient check: perturbs one conv1 filter weight and compares
// the empirical loss change against the analytical gradient from backward.
void gradient_check(cudnnHandle_t cudnn, cublasHandle_t cublas, convLayer& layer1,
                    convLayer& layer2, convLayer& layer3, fcLayer& fc1, fcLayer& fc2,
                    fcLayer& fc3, lossLayer& loss, float* d_input, uint8_t* d_labels,
                    void* d_workspace, float loss_old, int weight_idx, float eps);
