#include "conv_layer.h"
#include <vector>

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
    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;
    const float beta_accumulate = 1.0f;

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