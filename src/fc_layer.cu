#include "fc_layer.h"
#include <vector>
#include "cuda_kernels.h"

fcLayer create_fc_layer(cudnnHandle_t cudnn, int in_features, int out_features,
                        int batch_size, bool apply_relu, unsigned seed){
        
    // Copy parameters into the struct
    fcLayer layer;
    layer.in_features = in_features;
    layer.out_features = out_features;
    layer.batch_size = batch_size;
    layer.apply_relu = apply_relu;
    
    //Output tensor: shape [N, O, 1, 1] for cuDNN's 4D expectation
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
    layer.output_desc,
    CUDNN_TENSOR_NCHW,
    CUDNN_DATA_FLOAT,
    layer.batch_size,
    layer.out_features,
    1,
    1
));
    //ReLU activation descriptor (created even if unused, so destroy is uniform)
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&layer.relu_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        layer.relu_desc,
        CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN,
        0.0
    ));

    //Allocate GPU Buffers for this layers parameters
    const int weights_size = layer.out_features * layer.in_features;
    const int bias_size = layer.out_features;
    const int output_size = layer.batch_size * layer.out_features;
    

    CHECK_CUDA(cudaMalloc(&layer.d_weights, weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_bias, bias_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_output, output_size * sizeof(float)));

    //Memory allocation for this layer back prop.
    const int input_size = layer.batch_size * layer.in_features; // for d_grad_input
    CHECK_CUDA(cudaMalloc(&layer.d_grad_weights, weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_grad_bias, bias_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_grad_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer.d_grad_preact, output_size * sizeof(float)));

    //Initialize weights and bias
    std::vector<float> h_weights(weights_size);
    std::vector<float> h_bias(bias_size, 0.0f);

    he_init(h_weights, layer.in_features, seed);

    CHECK_CUDA(cudaMemcpy(layer.d_weights, h_weights.data(), weights_size*sizeof(float),
                            cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(layer.d_bias, h_bias.data(), bias_size * sizeof(float),
                            cudaMemcpyHostToDevice));
    
    return layer;
}

void forward_fc_layer(cudnnHandle_t cudnn, cublasHandle_t cublas, fcLayer& layer,
                        float* d_input){
    
    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;

    // Y = X · W^T  via cuBLAS SGEMM
    // We compute Y^T = W · X^T in cuBLAS's column-major view.
    // Row-major pointers are interpreted as their transposes by cuBLAS,
    // so CUBLAS_OP_T on W brings it back to its "logical" form.
    CHECK_CUBLAS(cublasSgemm(
        cublas,     // cuBLAS handle
        CUBLAS_OP_T,    // op(A): transpose W (logical [O, I] -> col-major view of W^T)
        CUBLAS_OP_N,    // op(B): no transpose on X (already X^T in col-major view)
        layer.out_features,     // m = rows of output in col-major view = O
        layer.batch_size,   // n = cols of output = N (batch size)
        layer.in_features,  // k = shared inner dimension = I (features per neuron)
        &alpha,      // scalar α = 1.0
        layer.d_weights,    // A pointer = weights matrix [O, I] row-major
        layer.in_features,  // lda = I (rows of A in col-major storage)
        d_input,    // B pointer = input matrix [N, I] row-major
        layer.in_features,  // ldb = I (rows of B in col-major storage)
        &beta_overwrite,    // scalar β = 0.0 → overwrite d_output
        layer.d_output,     // C pointer = output buffer [N, O] row-major
        layer.out_features  // ldc = O (rows of C in col-major storage)
    ));

    int total = layer.batch_size * layer.out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    add_bias_kernel<<<blocks, threads>>>(layer.d_output, layer.d_bias,
        layer.batch_size, layer.out_features);

    CHECK_CUDA(cudaGetLastError());
    
    // Y = ReLU(Y) — only if this layer applies ReLU
    if(layer.apply_relu){
        CHECK_CUDNN(cudnnActivationForward(
            cudnn,
            layer.relu_desc,
            &alpha,
            layer.output_desc,
            layer.d_output,
            &beta_overwrite,
            layer.output_desc,
            layer.d_output
        ));
    }
}

void backward_fc_layer(cudnnHandle_t cudnn, cublasHandle_t cublas, fcLayer& layer,
                       float* d_input, float* d_grad_output){
    
    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;

    //ReLU backwards if it exists.
    //dY right now is after ReLU
    //to take the gradient before ReLU activation we have to pass it back through ReLU
    //Lucky because with ReLU where x > 0 y is the same. 
    float* dY = d_grad_output; //FC3 with no ReLU
    if(layer.apply_relu){
        CHECK_CUDNN(cudnnActivationBackward(
            cudnn, layer.relu_desc,
            &alpha,
            layer.output_desc, layer.d_output,  // y  = post-activation output
            layer.output_desc, d_grad_output,   // dy = incoming gradient
            layer.output_desc, layer.d_output,  // x = pre activation input
            &beta_overwrite,    // beta = 0 so overwrite
            layer.output_desc, layer.d_grad_preact // dx = masked gradient
        ));
        dY = layer.d_grad_preact; // below this only dY gets used
    }

    // ---- 2. db = Σ_n dY[n][o] ----
    int threads = 256;
    int blocks = (layer.out_features + threads - 1) / threads;
    bias_backward_kernel<<<blocks, threads>>>(dY, layer.d_grad_bias, layer.batch_size, layer.out_features);
    CHECK_CUDA(cudaGetLastError());

    // dW = dY^T * X
    // cuBLAS: dW^T = X^T * dY
    CHECK_CUBLAS(cublasSgemm(
        cublas,
        CUBLAS_OP_N,    // A = X so dont change cublas sees X^T   
        CUBLAS_OP_T,    // B = dY, cublas sees dY^T so tranpose it 
        layer.in_features,  // m = rows or result dW^T
        layer.out_features, // n = cols of result
        layer.batch_size,   // k = inner dim, summed away
        &alpha,
        d_input, layer.in_features, // A = this layers forward input [N, I], lda = rows of A as cublas sees it
        dY, layer.out_features, // B = output gradient[N, O] ldb = rows of B as cublas sees it
        &beta_overwrite,
        layer.d_grad_weights, layer.in_features // C = dW [O, I], ldc = rows of C as cublas writes it 
    ));

    // dX = dY * W
    //cuBLAS:  dX^T = W^T * dY^T
    CHECK_CUBLAS(cublasSgemm(
        cublas,
        CUBLAS_OP_N,    // A = W, cublas sees W^T so we dont transpose it 
        CUBLAS_OP_N,    // B = dY cublas sees dY^T so we dont tranpose it 
        layer.in_features,  // m = rows of result dX^T
        layer.batch_size,   // n = cols of result
        layer.out_features, // k = inner dim. summed away
        &alpha,
        layer.d_weights, layer.in_features, // A = weights [O, I], lda = rows of A as cublas sees it
        dY, layer.out_features, // B = output gradient [N, O], ldb = rows of B as cublas sees it 
        &beta_overwrite, 
        layer.d_grad_input, layer.in_features   // C = dX [N, I], ldc = rows of C as cublas writes it 
    ));
}

void update_fc_layer(fcLayer& layer, float lr){
    const int threads = 256;

    int weight_size = layer.in_features * layer.out_features;

    int weight_blocks = (weight_size + threads - 1) / threads;
    sgd_update_kernel<<<weight_blocks, threads>>>(layer.d_weights, layer.d_grad_weights,
                                                  lr, weight_size);

    CHECK_CUDA(cudaGetLastError());

    int bias_blocks = (layer.out_features + threads - 1) / threads;
    sgd_update_kernel<<<bias_blocks, threads>>>(layer.d_bias, layer.d_grad_bias,
                                                lr,  layer.out_features);

    CHECK_CUDA(cudaGetLastError());
}

void destroy_fc_layer(fcLayer& layer){
    CHECK_CUDA(cudaFree(layer.d_weights));
    CHECK_CUDA(cudaFree(layer.d_bias));
    CHECK_CUDA(cudaFree(layer.d_output));
    CHECK_CUDA(cudaFree(layer.d_grad_weights));
    CHECK_CUDA(cudaFree(layer.d_grad_bias));
    CHECK_CUDA(cudaFree(layer.d_grad_input));
    CHECK_CUDA(cudaFree(layer.d_grad_preact));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(layer.output_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(layer.relu_desc));
}