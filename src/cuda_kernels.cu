#include "cuda_kernels.h"

// Adds a per-channel bias to a row-major [N, O] output tensor.
// One thread per output element. The bias vector b has shape [O], so each
// element Y[idx] gets b[idx % O] added in. Used as Phase 1 plumbing in
// forward_fc_layer to follow the cuBLAS GEMM with a bias add.
__global__ void add_bias_kernel(float* Y, const float* b, int N, int O){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = O * N;
    if(idx < total){
        int col = idx % O;
        Y[idx] += b[col];
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