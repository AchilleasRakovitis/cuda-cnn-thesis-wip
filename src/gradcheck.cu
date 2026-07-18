#include "gradcheck.h"
#include <iostream>

// The idea is the operational definition of a derivative: nudge one weight by a small
// epsilon, re-run the forward pass, and see how much the loss actually moved. That
// measured ratio (loss_new - loss_old) / eps is the true gradient, computed without
// using any of the backward-pass code. Comparing it against what backward computed
// therefore validates backward independently.
//
// We check a conv1 weight because its gradient travels the entire chain:
// loss -> FC3 -> FC2 -> FC1 -> conv3 -> conv2 -> conv1. Agreement validates almost
// every backward call in the network.
//

// The weight is restored afterwards, so the network state is unchanged.
void gradient_check(cudnnHandle_t cudnn, cublasHandle_t cublas, convLayer& layer1,
                    convLayer& layer2, convLayer& layer3, fcLayer& fc1, fcLayer& fc2,
                    fcLayer& fc3, lossLayer& loss, float* d_input, uint8_t* d_labels,
                    void* d_workspace, float loss_old, int weight_idx, float eps){
    
    // The analytical gradient that backward computed for this weight
    float analytic;
    CHECK_CUDA(cudaMemcpy(&analytic, layer1.d_grad_filter + weight_idx, sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // Save the original weight, then pertub it: w += eps
    float w_orig;
    CHECK_CUDA(cudaMemcpy(&w_orig, layer1.d_filter + weight_idx, sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    float w_pert = w_orig + eps;
    CHECK_CUDA(cudaMemcpy(layer1.d_filter + weight_idx, &w_pert, sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Re-run the forward pass only — exactly the same chain that produced loss_old
    forward_layer(cudnn, layer1, d_input, d_workspace);
    forward_layer(cudnn, layer2, layer1.d_pool_out, d_workspace);
    forward_layer(cudnn, layer3, layer2.d_pool_out, d_workspace);
    forward_fc_layer(cudnn, cublas, fc1, layer3.d_pool_out);
    forward_fc_layer(cudnn, cublas, fc2, fc1.d_output);
    forward_fc_layer(cudnn, cublas, fc3, fc2.d_output);
    forward_loss_layer(cudnn, loss, fc3.d_output, d_labels);

    float loss_new;
    CHECK_CUDA(cudaMemcpy(&loss_new, loss.d_final_loss, sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Restore the weight so the check leaves no side effects
    CHECK_CUDA(cudaMemcpy(layer1.d_filter + weight_idx, &w_orig, sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Compare
    const float empirical = (loss_new - loss_old) / eps;

    
    std::cout << "\n=== GRADIENT CHECK (conv1 filter[" << weight_idx
              << "], eps=" << eps << ") ===" << std::endl;
    std::cout << "  loss_old  = " << loss_old  << std::endl;
    std::cout << "  loss_new  = " << loss_new  << std::endl;
    std::cout << "  analytic  = " << analytic  << std::endl;
    std::cout << "  empirical = " << empirical << std::endl;
    std::cout << "  ratio     = " << (analytic / empirical) << "  (want ~1.0)" << std::endl;
}

