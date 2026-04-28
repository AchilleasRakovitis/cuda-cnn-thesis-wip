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

#include "conv_layer.h"
#include "cuda_kernels.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cudnn.h>
#include "data_loader.h"
#include <cublas_v2.h>
#include "common.h"

// =========================================================
// Struct that holds everything for one CNN layer
// =========================================================

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
    

