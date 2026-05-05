#include "loss_layer.h"
#include "cuda_kernels.h"

lossLayer create_loss_layer(cudnnHandle_t cudnn, int batch_size, int num_classes){
    // Get the parameters into the structs arguments
    lossLayer layer;
    layer.batch_size = batch_size;
    layer.num_classes = num_classes;

    //Logits tensor: shape [N, 10, 1, 1] for cuDNN's 4D expectation
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.logits_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        layer.logits_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        layer.batch_size,
        layer.num_classes,
        1, 1
    ));

    //GPU memory size buffers
    const int logits_size = layer.batch_size * layer.num_classes;
    const int losses_per_sample_size = layer.batch_size;
    const int final_loss_size = 1;

    //Memory Allocation
    CHECK_CUDA(cudaMalloc(&layer.d_logprobs, logits_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_losses_per_sample, losses_per_sample_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_final_loss, final_loss_size * sizeof(float)));

    //Debug Sanity 
    CHECK_CUDA(cudaMemset(layer.d_final_loss, 0, final_loss_size * sizeof(float)));

    return layer;
}

void destroy_loss_layer(lossLayer& layer){
    CHECK_CUDA(cudaFree(layer.d_logprobs));
    CHECK_CUDA(cudaFree(layer.d_losses_per_sample));
    CHECK_CUDA(cudaFree(layer.d_final_loss));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.logits_desc));

}

