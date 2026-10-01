#pragma once

#include "conv_kernels.h"
#include "conv_layer.h"

//Which convolution implementation to time
enum ConvImpl{
    CONV_CUDNN = 0,
    CONV_NAIVE,
    CONV_TILED,
    CONV_REGTILED,
    CONV_CUDNN_WINO,    // benchmark-only reference: cuDNN WINOGRAD (fused)
    CONV_CUDNN_WINO_NONFUSED,   // benchmark-only reference: cuDNN WINOGRAD_NONFUSED
    //REST
    CONV_IMPL_COUNT
};

const char* conv_impl_name(ConvImpl impl);

//Result of timing one implementation on the layer
struct BenchResult{
    ConvImpl impl;
    int layer_id;
    double ms_median;
    double ms_min;
    double gflops;
    double pct_peak;
    bool supported; // false id cudnn reports NOT_SUPPORTED for this shape
};

BenchResult bench_conv(cudnnHandle_t cudnn, convLayer& layer, int layer_id,
                        ConvImpl impl, float* d_input, void* d_workspace,
                        float* d_out, int warmup, int iters);

void run_all_benchmarks(cudnnHandle_t cudnn, convLayer& layer1, convLayer& layer2, convLayer& layer3,
                        float* d_input, void* d_workspace);