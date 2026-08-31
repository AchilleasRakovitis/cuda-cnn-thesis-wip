#pragma once

//Struct to group the dimensions of the convolution layer together
struct ConvDims{
    int N; // batch size (64)
    int C; // input channels
    int H; // input/output height (same due to padding)
    int W; // input/output width;
    int K; // number of filters same as output channels
    int R; // filter height (3)
    int S; // filter width (3);
    int pad; // padding (R / 2 = 1) 
};

//Host launcher helper function. Computes grid / block and launches the naive kernel
void launch_conv_naive(const float* d_input, const float* d_filter, float* d_output, const ConvDims& d);

__global__ void conv_forward_naive_kernel(const float* __restrict__ input, const float* __restrict__ filter,
                                          float* __restrict__ output, ConvDims d);