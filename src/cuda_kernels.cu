#include "cuda_kernels.h"

// Adds a per-channel bias to a row-major [N, O] output tensor.
// One thread per output element. The bias vector bias has shape [out_features], so each
// element output[idx] gets bias[idx % out_features] added in. Used as Phase 1 plumbing in
// forward_fc_layer to follow the cuBLAS GEMM with a bias add.
__global__ void add_bias_kernel(float* output, const float* bias, int batch_size, int out_features){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_features * batch_size;
    if(idx < total){
        int neuron = idx % out_features;
        output[idx] += bias[neuron];
    }
}

// NLL gather: for each image in the batch, picks out the log-probability
// of the true class and negates it to produce the per-sample cross-entropy
// loss. One thread per sample. The d_logprobs buffer is laid out row-major
// as [batch_size, num_classes], so image idx's true-class log-prob is at
// d_logprobs[idx * num_classes + label].
__global__ void nll_kernel(const float* d_logprobs ,const uint8_t* d_labels, 
                            float* d_losses, const int batch_size, const int num_classes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size){
            int label = d_labels[idx];
        float logp = d_logprobs[idx * num_classes + label];
        float loss = -logp;
        d_losses[idx] = loss;
    }
}

// Single-block tree reduction in shared memory.
// Each thread loads one element (or 0.0 if past N), then iteratively pairs
// values from increasing strides. Final mean stored at out_sum[0].
// Assumes blockDim.x is a power of 2 and >= batch_size.
__global__ void mean_reduce_kernel(const float* in, float* out_sum, int batch_size){
        
     extern __shared__ float sdata[];

    int tid = threadIdx.x;

    //Load the elements from global memory into shared memory
    sdata[tid] = (tid < batch_size) ? in[tid] : 0.0f;
    __syncthreads();

    //Tree reduction in shared memory
    for(int s = blockDim.x / 2; s > 0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out_sum[0] = sdata[0] / batch_size;
    }

}
// Computes the loss gradient dz = (p - y) / N for every logit.
// One thread per element of the [batch, classes] grid (640 threads for 64x10).
//   p = expf(log_p)  ... 
//   y = 1 if this element's class is the image's true label, else 0 ...
__global__ void loss_backward_kernel(const float* d_logprobs, const uint8_t* d_labels,
                                     float* d_grad_logits, int batch_size, int num_classes){
        
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = batch_size * num_classes;
      if(idx < total){
        int sample = idx / num_classes;
        int cls = idx % num_classes;

        float p  = expf(d_logprobs[idx]);
        float y  = (cls == d_labels[sample]) ? 1.0f : 0.0f;
        d_grad_logits[idx] = (p - y) / (float)batch_size;
      }  

}

//computes the gradient of the bias: db[o] = Σ_n dY[n][o]
//a thread for each neuron, each threads sum its own column of dY[N, O]
//no /N because 1 / N is already in dz calculated at the loss layer.
__global__ void bias_backward_kernel(const float* d_grad_output, float* d_grad_bias,
                                     int batch_size, int out_features){
    
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if(o < out_features){
        float sum = 0.0f;
        for(int n = 0; n < batch_size; n++){
            sum += d_grad_output[n * out_features + o]; // stride = out_features
        }
        d_grad_bias[o] = sum;
    }
}