/*
 * Example 6: Mini-VGG — 3 Convolutional Layers Chained
 * 
 * This is the closest to a real CNN so far.
 * Three layers, each doing: Conv → Bias → ReLU → MaxPool
 * The output of layer 1 feeds into layer 2, etc.
 * 
 * Architecture:
 *   Layer 1: [1, 3, 32, 32]  → conv 3x3 → 16 channels → pool → [1, 16, 16, 16]
 *   Layer 2: [1, 16, 16, 16] → conv 3x3 → 32 channels → pool → [1, 32, 8, 8]
 *   Layer 3: [1, 32, 8, 8]   → conv 3x3 → 64 channels → pool → [1, 64, 4, 4]
 * 
 * Notice how channels INCREASE (3→16→32→64) while spatial dims
 * DECREASE (32→16→8→4). This is the fundamental pattern of all CNNs.
 */

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cudnn.h>
#include "data_loader.h"
#include <cublas_v2.h>

#define CHECK_CUDNN(call) \
    do{ \
        cudnnStatus_t status = (call); \
        if(status != CUDNN_STATUS_SUCCESS){ \
            std::cerr << "cuDNN error at " << __FILE__ << ":" \
                      << __LINE__ << ": " \
                      << cudnnGetErrorString(status) << std::endl; \
                      std::exit(EXIT_FAILURE); \
        } \
    }while(0)

#define CHECK_CUDA(call) \
    do{ \
        cudaError_t err = (call); \
        if(err != cudaSuccess){ \
            std::cerr << "CUDA error at " << __FILE__ << ":" \
            << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        }   \
    }while(0)

#define CHECK_CUBLAS(call) \
    do{ \
        cublasStatus_t status = (call); \
        if(status != CUBLAS_STATUS_SUCCESS){ \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" \
                      << __LINE__ << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    }while(0)

void print_shape(const std::string& label, int n, int c, int h, int w){
    std::cout << "  " << label << ": [" << n << ", " << c
              << ", " << h << ", " << w << "]" << std::endl;
}

void print_gpu_tensor(const std::string& name, float* d_data, int count){
    std::vector<float> h_data(count);
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, count * sizeof(float), 
                            cudaMemcpyDeviceToHost));
    std::cout << "  " << name << ": [";
    int show = (count < 6) ? count : 6;
    for(int i = 0; i < show; i++){
        std::cout << h_data[i];
        if(i < show - 1) std::cout << ", ";
    }
    if(count > 6) std::cout << ",...";
    std::cout << "] (" << count << " total)" << std::endl;
}

// =========================================================
// Struct that holds everything for one CNN layer
// =========================================================
struct convLayer{
    //Descriptor
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc; //after conv, before pool
    cudnnTensorDescriptor_t pool_out_desc; // after pool
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t relu_desc;
    cudnnPoolingDescriptor_t pool_desc;

    //Algorithm
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_bytes;

    //GPU DATA
    float* d_filter;
    float* d_bias;
    float* d_conv_out; // intermediate: after conv+bias+relu
    float* d_pool_out; //final output of this layer

    //Dimensions for printing
    int out_n, out_c, out_h, out_w; //after conv
    int pool_n, pool_c, pool_h, pool_w; //after pool

};

// =========================================================
// Create and setup one layer
// =========================================================

convLayer create_layer(cudnnHandle_t cudnn, int in_n, int in_c, int in_h,
                        int in_w, int num_filters, int kernel_size,
                        cudnnTensorDescriptor_t input_desc){

    convLayer layer;
    layer.input_desc = input_desc;

    //Filter
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&layer.filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        layer.filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        num_filters,
        in_c,
        kernel_size,
        kernel_size
    ));

    //Convolution pad=1 for same output size, stride=1
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&layer.conv_desc));
    int pad = kernel_size / 2;
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        layer.conv_desc,
        pad, pad, 
        1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT 
    ));

    //Get conv output dims
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        layer.conv_desc,
        input_desc,
        layer.filter_desc,
        &layer.out_n,
        &layer.out_c,
        &layer.out_h,
        &layer.out_w
    ));

    //Conv output descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        layer.output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        layer.out_n,
        layer.out_c,
        layer.out_h,
        layer.out_w
    ));

    //Bias: [1, K, 1, 1]
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.bias_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        layer.bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, num_filters, 1, 1
    ));

    //ReLU
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&layer.relu_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        layer.relu_desc,
        CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN,
        0.0
    ));

    //Pooling: 2x2 max pool, stride=2
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&layer.pool_desc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(
        layer.pool_desc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        2, 2, 0, 0, 2, 2
    ));

    //Pool out Dims
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(
        layer.pool_desc,
        layer.output_desc,
        &layer.pool_n,
        &layer.pool_c,
        &layer.pool_h,
        &layer.pool_w
    ));

    //Pool Output Descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.pool_out_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        layer.pool_out_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        layer.pool_n,
        layer.pool_c,
        layer.pool_h,
        layer.pool_w
    ));

    //Pick best algorithm
    int max_algos;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
        cudnn,
        &max_algos
    ));
    
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(max_algos);
    int returned;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_desc,
        layer.filter_desc,
        layer.conv_desc,
        layer.output_desc,
        max_algos,
        &returned,
        perf.data()
    ));

    layer.algo = perf[0].algo;

    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc, layer.filter_desc, layer.conv_desc,
        layer.output_desc, layer.algo, &layer.workspace_bytes
    ));

    //Allocate GPU memory for this layer's weights and outputs
    int filter_size = num_filters * in_c * kernel_size * kernel_size;
    int conv_out_size = layer.out_n * layer.out_c * layer.out_h * layer.out_w;
    int pool_out_size = layer.pool_n * layer.pool_c * layer.pool_h * layer.pool_w;

    CHECK_CUDA(cudaMalloc(&layer.d_filter, filter_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_bias, num_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_conv_out, conv_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_pool_out, pool_out_size * sizeof(float)));

    //Initialize weights with small values, bias with zeros
    std::vector<float> h_filter(filter_size, 0.01f);
    std::vector<float> h_bias(num_filters, 0.0f);

    CHECK_CUDA(cudaMemcpy(layer.d_filter, h_filter.data(), filter_size * sizeof(float),
                            cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(layer.d_bias, h_bias.data(), num_filters * sizeof(float),
                            cudaMemcpyHostToDevice));

    return layer;
}

// =========================================================
// Run one layer's forward pass
// =========================================================
void forward_layer(cudnnHandle_t cudnn, convLayer& layer, float* d_input, void* d_workspace){
    float alpha = 1.0f;
    float beta_overwrite = 0.0f;
    float beta_accumulate = 1.0f;

    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        layer.input_desc,
        d_input,
        layer.filter_desc,
        layer.d_filter,
        layer.conv_desc,
        layer.algo,
        d_workspace,
        layer.workspace_bytes,
        &beta_overwrite,
        layer.output_desc,
        layer.d_conv_out
    ));

    //Bias (accumulate onto conv results)
    CHECK_CUDNN(cudnnAddTensor(
        cudnn,
        &alpha,
        layer.bias_desc,
        layer.d_bias,
        &beta_accumulate,
        layer.output_desc,
        layer.d_conv_out
    ));

    //ReLU (in place)
    CHECK_CUDNN(cudnnActivationForward(
        cudnn,
        layer.relu_desc,
        &alpha,
        layer.output_desc,
        layer.d_conv_out,
        &beta_overwrite,
        layer.output_desc,
        layer.d_conv_out
    ));

    //Max Pool
    CHECK_CUDNN(cudnnPoolingForward(
        cudnn,
        layer.pool_desc,
        &alpha,
        layer.output_desc,
        layer.d_conv_out,
        &beta_overwrite,
        layer.pool_out_desc,
        layer.d_pool_out
    ));

}

// =========================================================
// Cleanup one layer
// =========================================================
void destroy_layer(convLayer& layer){
    CHECK_CUDA(cudaFree(layer.d_filter));
    CHECK_CUDA(cudaFree(layer.d_bias));
    CHECK_CUDA(cudaFree(layer.d_conv_out));
    CHECK_CUDA(cudaFree(layer.d_pool_out));
    // Don't destroy input_desc — it belongs to the previous layer or main
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.pool_out_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(layer.filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(layer.conv_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(layer.relu_desc));
     CHECK_CUDNN(cudnnDestroyPoolingDescriptor(layer.pool_desc));
}

int main(){
    
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    std::cout << "=== Mini-VGG: 3-Layer CNN ===" << std::endl;

    // =========================================================
    // Input: 64 images, 3 channels (RGB), 32x32 (like CIFAR-10)
    // =========================================================
    const int in_n = 64, in_c = 3, in_h = 32, in_w = 32;
    int input_size = in_n * in_c * in_h * in_w;

    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w
    ));

    //Allocate and Initialize input
    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));

    std::vector<float> pixels;
    std::vector<uint8_t> labels;

    load_cifar10_batch("cifar-10-batches-bin/data_batch_1.bin", pixels, labels);

    CHECK_CUDA(cudaMemcpy(d_input, pixels.data(), input_size * sizeof(float),
                        cudaMemcpyHostToDevice));

    std::cout << "First 5 labels in batch: ";
    for (int i = 0; i < 5; i++) std::cout << static_cast<int>(labels[i]) << " ";
    std::cout << std::endl;
    
// =========================================================
    // Create 3 layers
    // =========================================================
    // Layer 1: 3 → 16 channels, 32x32 → 16x16 after pool
    // Layer 2: 16 → 32 channels, 16x16 → 8x8 after pool
    // Layer 3: 32 → 64 channels, 8x8 → 4x4 after pool

    std::cout << "\n--- Creating layers ---" << std::endl;
    print_shape("Input", in_n, in_c, in_h, in_w);

    convLayer layer1 = create_layer(cudnn, in_n, in_c, in_h, in_w, 16, 3, input_desc);

    std::cout << "Layer 1: " << in_c << " → " << 16 << " channels" << std::endl;

    print_shape(" After conv", layer1.out_n, layer1.out_c, layer1.out_h, layer1.out_w);

    print_shape(" After pool", layer1.pool_n, layer1.pool_c, layer1.pool_h, layer1.pool_w);

    convLayer layer2 = create_layer(cudnn, layer1.pool_n, layer1.pool_c, layer1.pool_h, layer1.pool_w,
                                    32, 3, layer1.pool_out_desc);
    
    std::cout << "Layer 2: " << 16 << " → " << 32 << " channels" << std::endl;
    print_shape("  After conv", layer2.out_n, layer2.out_c, layer2.out_h, layer2.out_w);
    print_shape("  After pool", layer2.pool_n, layer2.pool_c, layer2.pool_h, layer2.pool_w);

    convLayer layer3 = create_layer(cudnn, layer2.pool_n, layer2.pool_c, layer2.pool_h, layer2.pool_w,
                                    64, 3, layer2.pool_out_desc);

    std::cout << "Layer 3: " << 32 << " → " << 64 << " channels" << std::endl;
    print_shape("  After conv", layer3.out_n, layer3.out_c, layer3.out_h, layer3.out_w);
    print_shape("  After pool", layer3.pool_n, layer3.pool_c, layer3.pool_h, layer3.pool_w);

    // =========================================================
    // Allocate shared workspace (max of all layers)
    // =========================================================
    size_t max_ws = layer1.workspace_bytes;
    if(layer2.workspace_bytes > max_ws) max_ws = layer2.workspace_bytes;
    if(layer3.workspace_bytes > max_ws) max_ws = layer3.workspace_bytes;

    void* d_workspace = nullptr;
    if(max_ws > 0) CHECK_CUDA(cudaMalloc(&d_workspace, max_ws));
    std::cout << "\nShared workspace: " << max_ws << " bytes" << std::endl;

    // =========================================================
    // Forward pass through all 3 layers
    // =========================================================
    std::cout << "\n--- Forward Pass ---" << std::endl;

    //Layer 1: input -> layer1.d_pool_out
    forward_layer(cudnn, layer1, d_input, d_workspace);
    print_gpu_tensor("Layer 1 output", layer1.d_pool_out,
                     layer1.pool_n * layer1.pool_c * layer1.pool_h * layer1.pool_w);
    
    //Layer 2: layer1 output -> layer2.d_pool_out
    forward_layer(cudnn, layer2, layer1.d_pool_out, d_workspace);
    print_gpu_tensor("Layer 2 output", layer2.d_pool_out,
                     layer2.pool_n * layer2.pool_c * layer2.pool_h * layer2.pool_w);

    //Layer 3: layer2 output -> layer3.d_pool_out
    forward_layer(cudnn, layer3, layer2.d_pool_out, d_workspace);
    print_gpu_tensor("Layer 3 output", layer3.d_pool_out,
                     layer3.pool_n * layer3.pool_c * layer3.pool_h * layer3.pool_w);

    // =========================================================
    // Timing the full forward pass
    // =========================================================
    std::cout << "\n--- Timing full forward pass (100 iterations) ---" << std::endl;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    //WarmUp
    for(int i = 0; i < 10; i++){
        forward_layer(cudnn, layer1, d_input, d_workspace);
        forward_layer(cudnn, layer2, layer1.d_pool_out, d_workspace);
        forward_layer(cudnn, layer3, layer2.d_pool_out, d_workspace);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    //Timed
    const int iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < iters; i++){
        forward_layer(cudnn, layer1, d_input, d_workspace);
        forward_layer(cudnn, layer2, layer1.d_pool_out, d_workspace);
        forward_layer(cudnn, layer3, layer2.d_pool_out, d_workspace);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms;

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Total: " << ms << " ms for " << iters << " iterations" << std::endl;

    std::cout << "Average: " << ms / iters << " ms per forward pass" << std::endl;

    // =========================================================
    // Summary
    // =========================================================    
    std::cout << "\n=== Network Summary ===" << std::endl;
    std::cout << "Input:         [64, 3, 32, 32]   = " << input_size << " values" << std::endl;
    std::cout << "After Layer 1: [64, 16, 16, 16]  = " << 64 *16*16*16 << " values" << std::endl;
    std::cout << "After Layer 2: [64, 32, 8, 8]    = " << 64 * 32*8*8 << " values" << std::endl;
    std::cout << "After Layer 3: [64, 64, 4, 4]    = " << 64 * 64*4*4 << " values" << std::endl;
    std::cout << "\nPattern: channels UP (3→16→32→64), spatial DOWN (32→16→8→4)" << std::endl;
    std::cout << "Next step: flatten to 1024 values → fully connected layer → 10 classes" << std::endl;
    
    // =========================================================
    // Cleanup
    // =========================================================
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    if(d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));

    destroy_layer(layer1);
    destroy_layer(layer2);
    destroy_layer(layer3);
    CHECK_CUDNN(cudnnDestroy(cudnn));

    std::cout << "\nDone!" << std::endl;

    return 0;
}
    

