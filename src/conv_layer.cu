#include "conv_layer.h"
#include "cuda_kernels.h"
#include <vector>

convLayer create_layer(cudnnHandle_t cudnn, int in_n, int in_c, int in_h,
                        int in_w, int num_filters, int kernel_size,
                        cudnnTensorDescriptor_t input_desc, unsigned seed){

    convLayer layer;
    layer.input_desc = input_desc;
    
    //assign and save the input dimentions for input gradient use.
    layer.in_n = in_n;
    layer.in_c = in_c;
    layer.in_h = in_h;
    layer.in_w = in_w;

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

    //Backward Filter algorithm selection, same logic as forward - dFilter from(input, dConvOut)
    int max_bwd_filter_algos;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
        cudnn, &max_bwd_filter_algos
    ));

    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_perf_filter(max_bwd_filter_algos);
    int bwd_returned_filter;
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn,
        input_desc,     // x   — this layer's input
        layer.output_desc,  // dy  — gradient at the conv output
        layer.conv_desc,
        layer.filter_desc,  // dw  — what we're solving for
        max_bwd_filter_algos,
        &bwd_returned_filter,
        bwd_perf_filter.data()    
    ));

    layer.bwd_filter_algo = bwd_perf_filter[0].algo;

    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn, 
        input_desc,
        layer.output_desc,
        layer.conv_desc,
        layer.filter_desc,
        layer.bwd_filter_algo,
        &layer.bwd_filter_workspace_bytes
    ));

    //Backward data algorithm, dInput from (filter, dConvOut)
    int max_bwd_data_algos;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
        cudnn, &max_bwd_data_algos
    ));

    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_perf_data(max_bwd_data_algos);
    int bwd_returned_data;
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn,
        layer.filter_desc,  // w   — the filter
        layer.output_desc,  // dy  — gradient at the conv output
        layer.conv_desc,
        input_desc,     // dx  — what we're solving for
        max_bwd_data_algos,
        &bwd_returned_data,
        bwd_perf_data.data()
    ));

    layer.bwd_data_algo = bwd_perf_data[0].algo;

    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        layer.filter_desc,
        layer.output_desc,
        layer.conv_desc,
        input_desc,
        layer.bwd_data_algo,
        &layer.bwd_data_workspace_bytes
    ));

    
    std::cout << "  fwd algo=" << layer.algo
              << " ws=" << layer.workspace_bytes
              << " | bwd_filter algo=" << layer.bwd_filter_algo
              << " ws=" << layer.bwd_filter_workspace_bytes
              << " | bwd_data algo=" << layer.bwd_data_algo
              << " ws=" << layer.bwd_data_workspace_bytes << std::endl;

    //Allocate GPU memory for this layer's weights and outputs
    layer.kernel_size = kernel_size;
    layer.filter_size = num_filters * in_c * kernel_size * kernel_size;
    const int conv_out_size = layer.out_n * layer.out_c * layer.out_h * layer.out_w;
    const int pool_out_size = layer.pool_n * layer.pool_c * layer.pool_h * layer.pool_w;

    CHECK_CUDA(cudaMalloc(&layer.d_filter, layer.filter_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_bias, num_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_conv_out, conv_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_pool_out, pool_out_size * sizeof(float)));

    //Gradient buffer sizes. Each gradient has the same shape as the tensor it grades
    //so i can use the same sizes as the forward process.
    const int grad_conv_out_size = conv_out_size;
    const int grad_filter_size = layer.filter_size;
    const int grad_bias_size = num_filters;
    const int grad_input_size = in_n * in_c * in_h * in_w;

    //Memory allocation for grad buffers in device memory
    CHECK_CUDA(cudaMalloc(&layer.d_grad_conv_out, grad_conv_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_grad_filter, grad_filter_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_grad_bias, grad_bias_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_grad_input, grad_input_size * sizeof(float)));



    //Initialize weights with he_init, bias with zeros
    std::vector<float> h_filter(layer.filter_size);
    std::vector<float> h_bias(num_filters, 0.0f);

    int fan_in = in_c * kernel_size * kernel_size;
    he_init(h_filter, fan_in, seed); 

    CHECK_CUDA(cudaMemcpy(layer.d_filter, h_filter.data(), layer.filter_size * sizeof(float),
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

void backward_conv_layer(cudnnHandle_t cudnn, convLayer& layer, float* d_input,
                         float* d_grad_pool_out, void* d_workspace){

    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;
    
    //Pooling backward, routes gradient to each 2x2 tile's winner
    //expands [N, K, H/2, H/2] to [N, K, H, W]. it needs pooling's forward
    //input and output
    CHECK_CUDNN(cudnnPoolingBackward(
        cudnn,
        layer.pool_desc,
        &alpha,
        layer.pool_out_desc, layer.d_pool_out, //y = pooling forward OUTPUT
        layer.pool_out_desc, d_grad_pool_out,   //dy incoming gradients (at pool output)
        layer.output_desc, layer.d_conv_out,    //x = pooling forward input(post ReLU)
        &beta_overwrite,
        layer.output_desc, layer.d_grad_conv_out //dx = expanded gradient
    ));

    //ReLU backward, its a mask y is > 0 where x > 0.same as fc layer
    CHECK_CUDNN(cudnnActivationBackward(
        cudnn,
        layer.relu_desc,
        &alpha,
        layer.output_desc, layer.d_conv_out,    // y post activation
        layer.output_desc, layer.d_grad_conv_out,   //dy = gradient from pooling backward
        layer.output_desc, layer.d_conv_out,    //x = pre activation
        &beta_overwrite,
        layer.output_desc, layer.d_grad_conv_out    //dx = masked, written back in place
    ));

    //Bias backward, sums gradient over N, H, W (one bias serves a whole feature map)
    CHECK_CUDNN(cudnnConvolutionBackwardBias(
        cudnn,
        &alpha,
        layer.output_desc, layer.d_grad_conv_out, //dy
        &beta_overwrite,
        layer.bias_desc, layer.d_grad_bias //db
    ));

    //Filter gradient, corrects each filter's weight, we use(input, dConvOut)
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(
        cudnn,
        &alpha,
        layer.input_desc, d_input,  //x
        layer.output_desc, layer.d_grad_conv_out, //dy
        layer.conv_desc,
        layer.bwd_filter_algo,
        d_workspace, layer.bwd_filter_workspace_bytes,
        &beta_overwrite,
        layer.filter_desc, layer.d_grad_filter //dw
    ));

    //Data Gradient: the parcel that travels back, we need(filter, dConvOut)
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn,
        &alpha,
        layer.filter_desc, layer.d_filter,  //w
        layer.output_desc, layer.d_grad_conv_out, //dy
        layer.conv_desc,
        layer.bwd_data_algo,
        d_workspace, layer.bwd_data_workspace_bytes,
        &beta_overwrite,
        layer.input_desc, layer.d_grad_input //dx
    ));
}

void update_conv_layer(convLayer& layer, float lr){
    const int threads = 256;
    
    // 1. Filter update — layer.filter_size elements (K*C*R*S)
    int filter_blocks = (layer.filter_size + threads - 1) / threads;
    sgd_update_kernel<<<filter_blocks, threads>>>(layer.d_filter, layer.d_grad_filter,
                                                  lr, layer.filter_size);
    
    CHECK_CUDA(cudaGetLastError());

    // 2. Bias update — one bias per output channel
    int bias_blocks = (layer.out_c + threads - 1) / threads;
    sgd_update_kernel<<<bias_blocks, threads>>>(layer.d_bias, layer.d_grad_bias,
                                                lr, layer.out_c);
    
    CHECK_CUDA(cudaGetLastError());
}

// =========================================================
// Cleanup one layer
// =========================================================
void destroy_layer(convLayer& layer){
    CHECK_CUDA(cudaFree(layer.d_filter));
    CHECK_CUDA(cudaFree(layer.d_bias));
    CHECK_CUDA(cudaFree(layer.d_conv_out));
    CHECK_CUDA(cudaFree(layer.d_pool_out));
    CHECK_CUDA(cudaFree(layer.d_grad_conv_out));
    CHECK_CUDA(cudaFree(layer.d_grad_filter));
    CHECK_CUDA(cudaFree(layer.d_grad_bias));
    CHECK_CUDA(cudaFree(layer.d_grad_input));
    // Don't destroy input_desc — it belongs to the previous layer or main
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.pool_out_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(layer.filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(layer.conv_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(layer.relu_desc));
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(layer.pool_desc));
}