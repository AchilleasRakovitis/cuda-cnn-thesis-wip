#include "loss_layer.h"
#include "cuda_kernels.h"

lossLayer create_loss_layer(cudnnHandle_t cudnn, int batch_size, int num_classes){
    // Get the parameters into the structs arguments
    lossLayer layer;
    layer.batch_size = batch_size;
    layer.num_classes = num_classes;

    //Logits tensor: shape [N, 10, 1, 1] for cuDNN's 4D expectation
    //Same descriptor used for input (logits) and output (log-probs) since softmax preserves shape
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

void forward_loss_layer(cudnnHandle_t cudnn, lossLayer& layer, float* d_logits,
                        uint8_t* d_labels){
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Stage 1: log-softmax via cuDNN
    // Use CUDNN_SOFTMAX_LOG (numerically stable, computes log(p) directly)
    // Use CUDNN_SOFTMAX_MODE_INSTANCE (one softmax per image, reducing across the C dim)
    CHECK_CUDNN(cudnnSoftmaxForward(
        cudnn,
        CUDNN_SOFTMAX_LOG,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        layer.logits_desc,
        d_logits,
        &beta,
        layer.logits_desc,
        layer.d_logprobs
    ));
    
    // Stage 2: NLL gather (custom kernel)
    // For each image, look up its label and grab -log_p[label]. One thread per sample.
    int threads = 256;
    int blocks = (layer.batch_size + threads -1 ) / threads;

    nll_kernel<<<blocks, threads>>>(layer.d_logprobs, d_labels, layer.d_losses_per_sample,
                                    layer.batch_size, layer.num_classes);
    CHECK_CUDA(cudaGetLastError());


    // Stage 3: mean reduction (custom kernel)
    // Single-block tree reduction in shared memory, divides by N at the end.
    // Constraint: batch_size <= 1024 (CUDA hardware limit on threads per block).
    int reduce_threads = layer.batch_size;
    int reduce_blocks = 1;
    size_t shared_bytes = reduce_threads * sizeof(float);

    mean_reduce_kernel<<<reduce_blocks, reduce_threads, shared_bytes>>>(layer.d_losses_per_sample,
                        layer.d_final_loss, layer.batch_size);
    CHECK_CUDA(cudaGetLastError());

}

void destroy_loss_layer(lossLayer& layer){
    CHECK_CUDA(cudaFree(layer.d_logprobs));
    CHECK_CUDA(cudaFree(layer.d_losses_per_sample));
    CHECK_CUDA(cudaFree(layer.d_final_loss));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.logits_desc));

}

