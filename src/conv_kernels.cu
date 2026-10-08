#include "conv_kernels.h"
#include <cmath>
#include <vector>
#include <cstdio>

//--- M3 shared-tiled constants ---
#define TILED_BLK 16 // threads per block dimension
#define TILED_SH 18 // shared tile = TILED_BLK + R - 1 (16 + 3 - 1)

//--- M4 register tiled constants ---
#define REG_TPT 2 // outputs per thread, per dimension (2x2 block)
#define REG_BLK 16 // threads per block dimension
#define REG_OUT 32 // output tile per block = REG_BLK * REG_TPT
#define REG_SH 34 // shared tile = REG_OUT + R - 1 (32 + 3 - 1)

//--- M5 outer-product kernel constants ---
#define OUTER_CH_PER_THREAD 4 // output channels per thread
#define OUTER_POS_PER_THREAD 4 // output positions per thread
#define OUTER_CH_PER_BLOCK 16 // output channels per block
#define OUTER_POS_PER_BLOCK 64 // output positions per block
#define OUTER_THREADS_CH 4 // threads along channels
#define OUTER_THREADS_POS 16 // threads along positions 
#define OUTER_SH_IN_MAX 256  // max input-tile floats(L1: 4 * 34 = 136)
#define OUTER_FILTER 3  // filter height = width (launcher enforces R == S == 3)
#define OUTER_TAPS (OUTER_FILTER * OUTER_FILTER)    // 9 taps per 3x3 filter

// Naive direct convolution, forward pass.
//
// Computes, for every output element:
//   out[n,k,p,q] = SUM over c,r,s of in[n,c,p+r-pad,q+s-pad] * filt[k,c,r,s]
//
// Parallelisation strategy: the output elements are independent of one another,
// so they map to threads; the (c,r,s) reduction is dependent and stays as a
// sequential loop inside each thread, accumulating in a register. One thread
// therefore produces exactly one output element with a single store.
//
// This is the reference implementation for Phase 2: unoptimised by design, it
// reads every operand from global memory with no reuse.
__global__ void conv_forward_naive_kernel(const float* __restrict__ input, const float* __restrict__ filter,
                                          float* __restrict__ output, ConvDims d){

        // CUDA provides a 3D grid but the output is 4D, so two axes must share one
        // grid dimension. p and q are kept separate because adjacent q values are
        // adjacent in memory, which later milestones exploit for tiling. n and k
        // have no such relationship, so they are folded together: pairs are
        // enumerated with k varying fastest (z = n*K + k), and the inverse recovers
        // n by integer division and k by remainder.
        int q = blockIdx.x * blockDim.x + threadIdx.x;
        int p = blockIdx.y * blockDim.y + threadIdx.y;
        int n = blockIdx.z / d.K;
        int k = blockIdx.z % d.K;


        if(p >= d.H ||q>= d.W ) return;

        float acc = 0;
        for(int c = 0; c <= d.C - 1; c++){
            for(int r = 0; r <= d.R - 1; r++){
                for(int s = 0; s <= d.S - 1; s++){
                    // -d.pad is needed to get the position inside the input image and not a wrong one because of padding
                    int ih = p + r - d.pad; 
                    int iw = q + s - d.pad;
                    if(ih >= 0 && ih < d.H && iw >= 0 && iw < d.W){
                        //flattening the input and the filter because in memory they got stored row-major and we have 4d tensors
                        int in_idx = ((n*d.C + c)*d.H + ih)*d.W + iw;
                        int flt_idx = ((k*d.C + c)*d.R + r)*d.S + s;
                        acc += input[in_idx] * filter[flt_idx];        
                    }
                }
            }
        }

        int out_idx = ((n*d.K + k)*d.H + p)*d.W + q;
        output[out_idx] = acc;
}

// Host-side launcher: chooses the execution configuration and starts the kernel.
//
// The block is 16x16 (256 threads, eight full warps) covering a 16x16 patch of
// the output. The grid covers the whole output: x and y are ceiling divisions so
// that extents not divisible by 16 are not truncated — plain integer division
// would give zero blocks for an 8-pixel extent and nothing would be computed.
// The z dimension holds every (image, filter) pair, N*K of them.
//
// Ceiling division over-provisions when the output is smaller than the block:
// layer 3 (8x8) launches a full 16x16 block, so 192 of its 256 threads exit at
// the bounds guard. A uniform block size is kept at this stage so the comparison
// across layers stays clean; per-layer tuning is a later refinement.
void launch_conv_naive(const float* d_input, const float* d_filter, float* d_output, 
                        const ConvDims& d){
    dim3 block(16, 16);
    dim3 grid( (d.W + 15) / 16, (d.H + 15) / 16, d.N * d.K);

    conv_forward_naive_kernel<<<grid, block>>>(d_input, d_filter, d_output, d);
    
}

// Correctness check: runs cuDNN and the custom kernel on identical inputs and
// compares the two outputs numerically.
//
// Both results go to buffers private to this function, so layer.d_conv_out is
// never touched and the check has no side effects on the training pipeline —
// it can be called at any point without disturbing state.
//
// The comparison is tolerance-based rather than bitwise. Floating-point addition
// is not associative, so an implementation that accumulates in a different order
// may produce small, entirely legitimate differences. A bitwise test would fail
// on correct code. This also matters because cuDNN algorithm selection was found
// to be nondeterministic on the shared server, and later kernels will reorder the
// summation deliberately.
//
// Three metrics are reported:
//   max abs diff  — the largest distance between corresponding elements, with its
//                   index. The index localises a failure: a low index points at
//                   the padding logic, an interior one at the address arithmetic.
//   max rel diff  — the same distance scaled by the magnitude of the values. An
//                   absolute difference of 0.1 is negligible near 1000 and severe
//                   near 0.001, so both measures are needed.
//   elems > tol   — how many elements exceed the tolerance. A handful suggests
//                   numerical noise at the borders; a large fraction suggests a
//                   structural bug.
void verify_conv_naive(cudnnHandle_t cudnn, convLayer& layer, float* d_input, void* d_workspace){
    int total_elements = layer.out_n * layer.out_c * layer.out_h * layer.out_w;
    size_t total_bytes = total_elements * sizeof(float);

    float* d_naive;
    float* d_cudnn;
    CHECK_CUDA(cudaMalloc(&d_naive, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_cudnn, total_bytes));

    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;
        
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
        d_cudnn
    ));

    // H and W come from the input dimensions and serve for both input and output,
    // because "same" padding preserves the spatial extent. K is the output channel
    // count (out_c), not the input channel count.
    ConvDims d;
    d.N = layer.in_n;
    d.C = layer.in_c;
    d.H = layer.in_h;
    d.W = layer.in_w;
    d.K = layer.out_c;
    d.R = layer.kernel_size;
    d.S = layer.kernel_size;
    d.pad = layer.kernel_size / 2;

    launch_conv_naive(d_input, layer.d_filter, d_naive, d);
    
    CHECK_CUDA(cudaDeviceSynchronize());


    std::vector<float> h_cudnn(total_elements);
    std::vector<float> h_naive(total_elements);

    CHECK_CUDA(cudaMemcpy(h_cudnn.data(), d_cudnn, total_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_naive.data(), d_naive, total_bytes, cudaMemcpyDeviceToHost));

    const float threshold = 1e-3f;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int max_abs_idx = -1;
    int n_over = 0;

    for(int i = 0; i < total_elements; i++){
        const float a = h_cudnn[i];
        const float b = h_naive[i];

        const float abs_diff = fabsf(a-b);
        const float rel_diff = abs_diff / (fabsf(a) + fabsf(b) + 1e-8f);

        if(abs_diff > max_abs){
            max_abs = abs_diff;
            max_abs_idx = i;
        }
        if (rel_diff > max_rel) max_rel = rel_diff;
        if (abs_diff > threshold) n_over++;
    }

    printf("\n=== VERIFY conv naive vs cuDNN ===\n");
    printf("  shape        : [%d, %d, %d, %d]  (%d elements)\n",
           layer.out_n, layer.out_c, layer.out_h, layer.out_w, total_elements);
    printf("  max abs diff : %.3e  (at index %d)\n", max_abs, max_abs_idx);
    printf("  max rel diff : %.3e\n", max_rel);
    printf("  elems > %.0e : %d\n", threshold, n_over);
    printf("  verdict      : %s\n", (max_abs < threshold ? "PASS" : "FAIL"));

    CHECK_CUDA(cudaFree(d_cudnn));
    CHECK_CUDA(cudaFree(d_naive));
}
// Shared-memory tiled direct convolution, forward pass.
//
// Computes the same operation as the naive kernel:
//   out[n,k,p,q] = SUM over c,r,s of in[n,c,p+r-pad,q+s-pad] * filt[k,c,r,s]
// but reduces global-memory traffic through reuse. Neighbouring output pixels
// share most of their input window (6 of 9 values for a 3x3 kernel); the naive
// kernel re-reads those overlaps from global memory once per thread, whereas
// here each block loads its input region into shared memory ONCE and all 256
// threads read it back from there.
//
// Each block computes a 16x16 output tile. To do so it needs an 18x18 input
// region: the 16x16 body plus a one-pixel halo on every side, because the 3x3
// filter of every border output reaches one pixel outside the tile
// (input_tile = output_tile + R - 1 = 16 + 3 - 1 = 18).
//
// Channels are processed ONE AT A TIME (an 18x18 tile per channel). Holding all
// C channels in shared memory at once would exceed the ~21 KiB per-block budget
// (18*18*32*4 bytes ~= 40 KiB at layer 3), so the kernel loops over channels,
// reloading the shared tile each iteration and accumulating into a register.
//
// Two __syncthreads() per channel are mandatory. The first, after loading,
// ensures the tile is fully populated before any thread reads it. The second,
// after computing, ensures every thread has finished reading the current channel
// before any thread overwrites the tile with the next one. Both barriers sit
// OUTSIDE the bounds guard: every thread in the block must reach them, or threads
// that returned early would deadlock those still waiting.
//
// The padding is handled once, during loading: out-of-image positions are written
// as zero into the halo. Because the halo is already present in shared memory,
// the compute step reads tile[(threadIdx.y + r) * 18 + (threadIdx.x + s)] with no
// bounds check at all -- the +pad from the halo and the -pad from the filter
// offset cancel, leaving a clean threadIdx + (r,s). This is a hidden benefit of
// tiling beyond bandwidth: the inner loop has no per-element boundary branches.
//
// This kernel corresponds to the T1 (shared-memory tile) level of the Lopes
// hierarchy. Technique guided by the reference implementation (github.com/
// paclopes/cuDconv); the loading loop, indexing and channel strategy are our own,
// without the reference's vectorized (float4) loads or register tiling.
__global__ void conv_forward_tiled_kernel(const float* __restrict__ input, const float* __restrict__ filter,
                                          float* __restrict__ output, ConvDims d){

    // assigns an index to each element while flattening from 2d to 1d
    int flat_id = threadIdx.y * TILED_BLK + threadIdx.x;
    
    // thread index for the output pixels, q = col, p = row
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y * blockDim.y + threadIdx.y;
    
    // making 4d compatible to 3d thread mapping, n = image, k = filter 
    int n = blockIdx.z / d.K;
    int k = blockIdx.z % d.K;

    // mapping where does the output tile start, needed for the shared tile
    int out_row_start = blockIdx.y * TILED_BLK;
    int out_col_start = blockIdx.x * TILED_BLK;

    // sum accumulator for convolution
    float acc = 0.0f;

    // declaring the shared memory buffer 
    __shared__ float tile[TILED_SH * TILED_SH];

    //loading each channel seperately
    for(int c = 0; c < d.C; c++){
        
        //load the shared tile of channel c in shared memory
        for(int idx = flat_id; idx < TILED_SH * TILED_SH; idx+= TILED_BLK*TILED_BLK){
            // converting 1d into 2d for the shared tile access
            int local_row = idx / TILED_SH;
            int local_col = idx % TILED_SH;
            
            //going from the shared tile to the input image
            int ih = out_row_start - d.pad + local_row;
            int iw = out_col_start - d.pad + local_col;

            // bounds check and NCHW flattening indexing
            if(ih >= 0 && ih < d.H && iw >= 0 && iw < d.W){
                tile[idx] = input[((n * d.C + c) * d.H + ih) * d.W + iw];
            }else{
                tile[idx] = 0.0f;
            }
        }

        __syncthreads();

        if(p < d.H && q < d.W){
            // scanning the filter
            for(int r = 0; r < d.R; r++){
                for(int s = 0; s < d.S; s++){
                    //read from the shared buffer
                    float in_val = tile[(threadIdx.y + r) * TILED_SH + (threadIdx.x + s)];
                    //same index flattening like in the naive for the filter
                    float w_val = filter[((k * d.C + c) * d.R + r) * d.S + s];
                    //conv MAC computation
                    acc += in_val * w_val;
                }
            }
        }

        __syncthreads();
    }

    //bounds check and write to the output 
    if(p < d.H && q < d.W){
        output[((n * d.K + k) * d.H + p) * d.W + q] = acc;
    }

}

void launch_conv_tiled(const float* d_input, const float* d_filter, float* d_output, const ConvDims& d){
    
    dim3 block(TILED_BLK, TILED_BLK);
    dim3 grid( (d.W + TILED_BLK - 1) / TILED_BLK, (d.H + TILED_BLK - 1) / TILED_BLK, d.N * d.K);

    conv_forward_tiled_kernel<<<grid, block>>>(d_input, d_filter, d_output, d);
}

void verify_conv_tiled(cudnnHandle_t cudnn, convLayer& layer, float* d_input, void* d_workspace){
    int total_elements = layer.out_n * layer.out_c * layer.out_h * layer.out_w;
    size_t total_bytes = total_elements * sizeof(float);

    float* d_tiled;
    float* d_cudnn;
    CHECK_CUDA(cudaMalloc(&d_tiled, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_cudnn, total_bytes));

    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;

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
        d_cudnn
    ));

    // H and W come from the input dimensions and serve for both input and output,
    // because "same" padding preserves the spatial extent. K is the output channel
    // count (out_c), not the input channel count.
    ConvDims d;
    d.N = layer.in_n;
    d.C = layer.in_c;
    d.H = layer.in_h;
    d.W = layer.in_w;
    d.K = layer.out_c;
    d.R = layer.kernel_size;
    d.S = layer.kernel_size;
    d.pad = layer.kernel_size / 2;

    launch_conv_tiled(d_input, layer.d_filter, d_tiled, d);

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_cudnn(total_elements);
    std::vector<float> h_tiled(total_elements);

    CHECK_CUDA(cudaMemcpy(h_cudnn.data(), d_cudnn, total_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_tiled.data(), d_tiled, total_bytes, cudaMemcpyDeviceToHost));

    const float threshold = 1e-3f;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int max_abs_idx = -1;
    int n_over = 0;

    for(int i = 0; i < total_elements; i++){
        const float a = h_cudnn[i];
        const float b = h_tiled[i];

        const float abs_diff = fabsf(a-b);
        const float rel_diff = abs_diff / (fabsf(a) + fabsf(b) + 1e-8f);

        if(abs_diff > max_abs){
            max_abs = abs_diff;
            max_abs_idx = i;
        }
        if (rel_diff > max_rel) max_rel = rel_diff;
        if (abs_diff > threshold) n_over++;
    }

    printf("\n=== VERIFY conv tiled vs cuDNN ===\n");
    printf("  shape        : [%d, %d, %d, %d]  (%d elements)\n",
           layer.out_n, layer.out_c, layer.out_h, layer.out_w, total_elements);
    printf("  max abs diff : %.3e  (at index %d)\n", max_abs, max_abs_idx);
    printf("  max rel diff : %.3e\n", max_rel);
    printf("  elems > %.0e : %d\n", threshold, n_over);
    printf("  verdict      : %s\n", (max_abs < threshold ? "PASS" : "FAIL"));

    CHECK_CUDA(cudaFree(d_cudnn));
    CHECK_CUDA(cudaFree(d_tiled));
}

// Register-tiled direct convolution, forward pass.
//
// Same operation as the naive and tiled kernels, but each thread computes a
// 2x2 block of outputs instead of one. A 16x16 thread block therefore covers
// a 32x32 output tile, loaded with its halo into a 34x34 shared tile
// (REG_SH = REG_OUT + R - 1), one channel at a time as in the tiled kernel.
//
// The four outputs of a thread use the same weight at each filter position
// (r,s), so the weight is read once and multiplied into four accumulators held
// in registers. The compute reads need no bounds check (halo-padded shared
// tile); each of the four writes is guarded separately, because a thread's
// 2x2 block can straddle the image edge.
//
// This kernel corresponds to the T4 (thread register tile) level of the Lopes
// hierarchy. Technique guided by the reference implementation (github.com/
// paclopes/cuDconv); the 2x2 tile, indexing and variables are our own,
// without the reference's vectorized (float4) loads or analytic tile sizes.
__global__ void conv_forward_regtiled_kernel(const float* __restrict__ input, const float* __restrict__ filter,
                                             float* __restrict__ output, ConvDims d){

    int flat_id = threadIdx.y * REG_BLK + threadIdx.x;

    // base output position of THIS thread's 2x2 block (in image coords)
    int base_row = blockIdx.y * REG_OUT + threadIdx.y * REG_TPT;
    int base_col = blockIdx.x * REG_OUT + threadIdx.x * REG_TPT;

    int n = blockIdx.z / d.K;
    int k = blockIdx.z % d.K;

    //where the block's shared tile starts(in image coords), same for all threads
    int out_row_start = blockIdx.y * REG_OUT;
    int out_col_start = blockIdx.x * REG_OUT;

    //four accumulators for the 2x2 output block
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    __shared__ float tile[REG_SH * REG_SH];

    for(int c = 0; c < d.C; c++){
        
        
        for(int idx = flat_id; idx < REG_SH * REG_SH; idx+= REG_BLK * REG_BLK){

            int local_row = idx / REG_SH;
            int local_col = idx % REG_SH;

            int ih = out_row_start - d.pad + local_row;
            int iw = out_col_start -d.pad + local_col;

            if(ih >= 0 && ih < d.H && iw >= 0 && iw < d.W){
                tile[idx] = input[((n * d.C + c) * d.H + ih) * d.W + iw];
            }else{
                tile[idx] = 0.0f;
            }
        }

        __syncthreads();

        for(int r = 0; r < d.R; r++){
            for(int s = 0; s < d.S; s++){
                //The weight is the same for every accumulator
                float w_val = filter[((k * d.C + c) * d.R + r) * d.S + s];

                // the four accumulator for the 2x2 output of each thread
                acc00 += tile[(threadIdx.y*2 + 0 + r) * REG_SH + (threadIdx.x*2 + 0 + s)] * w_val;
                acc01 += tile[(threadIdx.y*2 + 0 + r) * REG_SH + (threadIdx.x*2 + 1 + s)] * w_val;
                acc10 += tile[(threadIdx.y*2 + 1 + r) * REG_SH + (threadIdx.x*2 + 0 + s)] * w_val;
                acc11 += tile[(threadIdx.y*2 + 1 + r) * REG_SH + (threadIdx.x*2 + 1 + s)] * w_val;
            }
        }

        __syncthreads();

    }

    //output (0,0) -> (base_row + 0, base_col + 0)
    if(base_row + 0 < d.H && base_col + 0 < d.W){
        output[((n * d.K + k) * d.H + (base_row + 0)) * d.W + (base_col + 0)] = acc00;
    }

    if(base_row + 0 < d.H && base_col + 1 < d.W){
        output[((n * d.K + k) * d.H + (base_row + 0)) * d.W + (base_col + 1)] = acc01;
    }

    if(base_row + 1 < d.H && base_col + 0 < d.W){
        output[((n * d.K + k) * d.H + (base_row + 1)) * d.W + (base_col + 0)] = acc10;
    }

    if(base_row + 1 < d.H && base_col + 1 < d.W){
        output[((n * d.K + k) * d.H + (base_row + 1)) * d.W + (base_col + 1)] = acc11;
    }
}

void launch_conv_regtiled(const float* d_input, const float* d_filter, float* d_output, const ConvDims& d){
    dim3 block(REG_BLK, REG_BLK);
    dim3 grid((d.W + REG_OUT - 1) / REG_OUT, (d.H + REG_OUT - 1) / REG_OUT, d.N * d.K); // ceil(W / 32)

    conv_forward_regtiled_kernel<<<grid, block>>>(d_input, d_filter, d_output, d);
}

void verify_conv_regtiled(cudnnHandle_t cudnn, convLayer& layer, float* d_input, void* d_workspace){
    int total_elements = layer.out_n * layer.out_c * layer.out_h * layer.out_w;
    size_t total_bytes = total_elements * sizeof(float);

    float* d_reg;
    float* d_cudnn;
    CHECK_CUDA(cudaMalloc(&d_reg, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_cudnn, total_bytes));

    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;

    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn, &alpha, layer.input_desc, d_input,
        layer.filter_desc, layer.d_filter, layer.conv_desc, layer.algo,
        d_workspace, layer.workspace_bytes,
        &beta_overwrite, layer.output_desc, d_cudnn));

    ConvDims d;
    d.N = layer.in_n;  d.C = layer.in_c;
    d.H = layer.in_h;  d.W = layer.in_w;
    d.K = layer.out_c;
    d.R = layer.kernel_size;  d.S = layer.kernel_size;
    d.pad = layer.kernel_size / 2;

    launch_conv_regtiled(d_input, layer.d_filter, d_reg, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_cudnn(total_elements);
    std::vector<float> h_reg(total_elements);
    CHECK_CUDA(cudaMemcpy(h_cudnn.data(), d_cudnn, total_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_reg.data(), d_reg, total_bytes, cudaMemcpyDeviceToHost));

    const float threshold = 1e-3f;
    float max_abs = 0.0f, max_rel = 0.0f;
    int max_abs_idx = -1, n_over = 0;

    for(int i = 0; i < total_elements; i++){
        const float a = h_cudnn[i];
        const float b = h_reg[i];
        const float abs_diff = fabsf(a-b);
        const float rel_diff = abs_diff / (fabsf(a) + fabsf(b) + 1e-8f);
        if(abs_diff > max_abs){ max_abs = abs_diff; max_abs_idx = i; }
        if(rel_diff > max_rel) max_rel = rel_diff;
        if(abs_diff > threshold) n_over++;
    }

    printf("\n=== VERIFY conv regtiled vs cuDNN ===\n");
    printf("  shape        : [%d, %d, %d, %d]  (%d elements)\n",
           layer.out_n, layer.out_c, layer.out_h, layer.out_w, total_elements);
    printf("  max abs diff : %.3e  (at index %d)\n", max_abs, max_abs_idx);
    printf("  max rel diff : %.3e\n", max_rel);
    printf("  elems > %.0e : %d\n", threshold, n_over);
    printf("  verdict      : %s\n", (max_abs < threshold ? "PASS" : "FAIL"));

    CHECK_CUDA(cudaFree(d_cudnn));
    CHECK_CUDA(cudaFree(d_reg));
}

// Outer-product direct convolution, forward pass (M5).
//
// Computes the same operation as the previous kernels:
//   out[n,k,p,q] = SUM over c,r,s of in[n,c,p+r-pad,q+s-pad] * filt[k,c,r,s]
// but tiles over OUTPUT CHANNELS and POSITIONS instead of over the rows and
// columns of a single output channel. This is the Y = W X' view of Lopes
// (Eq. 2-4): rows of Y are output channels k, columns are output positions.
//
// Why: arithmetic intensity A_I = MADs per shared-memory load must reach
// A_G = M_M / L_M to keep shared memory from being the bottleneck. On the
// TITAN RTX (Turing) M_M = 64 FP32 cores per SM and L_M = 32 banks, so A_G = 2.
// The M3/M4 kernels reach only A_I = 1 (one shared read per MAD). Here each
// thread computes 4 channels x 4 positions as an OUTER PRODUCT: per (c,r,s) it
// loads 4 weights + 4 inputs from shared memory and performs 4*4 = 16 MADs,
// so A_I = 16 / 8 = 2 = A_G (Lopes Sec. 4, Eq. 16-19).
//
// Why channels and positions: CIFAR feature maps shrink (32x32 -> 16x16 -> 8x8)
// while K grows (16 -> 32 -> 64), so large SPATIAL tiles leave most threads
// idle on small layers (M4 finding). The block tile is sized to fit every
// layer exactly: 16 channels (largest divisor of K = 16, 32, 64) x 64 positions
// (largest divisor of H*W = 1024, 256, 64), all from ONE image. 64 is a
// multiple of W, so a block always covers whole output rows.
//
// Mapping:
//   grid  = (H*W / 64, K / 16, N): x = group of 64 positions, y = group of 16
//           channels, z = image n
//   block = (16, 4): threadIdx.x picks 4 consecutive positions (same row,
//           since W % 4 == 0), threadIdx.y picks 4 consecutive channels
//   position = flat index p*W + q;  p = pos / W, q = pos % W
//
// Shared memory, per input channel c (channels processed one at a time, as in
// M3/M4):
//   sh_weights[16 * 9]  : the 3x3 filters of the block's 16 output channels
//                         for channel c, flattened as r*3 + s
//   sh_input[(rows+2) * (W+2)] : the block's input rows plus a one-pixel halo
//                         on every side; out-of-image cells are written as 0
//                         (zero padding, applied once at load time)
// Because shared row 0 / column 0 already sit one pixel up-left of the first
// output, the compute step indexes sh_input with (row_in_blk + r, col + s) and
// no -pad: the halo offset and the filter offset cancel, as in M3/M4.
//
// Two __syncthreads() per channel: after the loads (tiles complete before any
// read) and after the compute (all reads done before the next channel
// overwrites the tiles). Every thread must reach both, so no early return.
// No write guard is needed: the launcher only accepts shapes for which every
// block is full (K % 16 == 0, (H*W) % 64 == 0, 64 % W == 0, W % 4 == 0,
// R == S == 3).
//
// Technique (A_I >= A_G tile sizing, outer-product thread tile, channel x
// position tiling) follows Lopes 2026 and the reference implementation
// (github.com/paclopes/cuDconv, MIT). Tile sizes, index mapping, variables and
// loading strategy are our own; one input channel per barrier (P1 = 9) instead
// of the reference's multi-channel P1, and without vectorized (float4) loads.
__global__ void conv_forward_outer_kernel(const float* __restrict__ input, const float* __restrict__ filter,
                                          float* __restrict__ output, ConvDims d){

    // ---- BLOCK: which image, channels, positions ----
    int n = blockIdx.z;
    int blk_ch_start = blockIdx.y * OUTER_CH_PER_BLOCK;
    int blk_pos_start = blockIdx.x * OUTER_POS_PER_BLOCK;
    int blk_out_row_count = OUTER_POS_PER_BLOCK / d.W;
    int blk_out_row_start = blk_pos_start / d.W;

    // ---- THREAD: which channels, positions ----
    int thr_pos_start = blk_pos_start + threadIdx.x * OUTER_POS_PER_THREAD;
    int thr_out_row = thr_pos_start / d.W;
    int thr_out_col = thr_pos_start % d.W;
    int thr_row_in_blk = thr_out_row - blk_out_row_start;
    int thr_ch_start = blk_ch_start + threadIdx.y * OUTER_CH_PER_THREAD;

    // ---- SHARED input tile shape (output rows/cols + halo) ----
    int sh_in_rows =  blk_out_row_count + 2;
    int sh_in_cols =  d.W + 2;

    // ---- SHARED tiles, refilled for each input channel c ----
    __shared__ float sh_weights[OUTER_CH_PER_BLOCK * 9]; //[16 channels][9 taps r*3+s]
    __shared__ float sh_input[OUTER_SH_IN_MAX]; // // [sh_in_rows][sh_in_cols], flattened

    // ---- 16 accumulators: acc[i][j] = channel thr_ch_start+i, position thr_pos_start+j ----
    float acc[OUTER_CH_PER_THREAD][OUTER_POS_PER_THREAD];
    #pragma unroll
    for(int i = 0; i < OUTER_CH_PER_THREAD; i++){
        #pragma unroll
        for(int j = 0; j < OUTER_POS_PER_THREAD; j++){
            acc[i][j] = 0.0f;
        }
    }

    // ---- flat thread id, used by the cooperative loads (like flat_id in M3/M4) ----
    int thr_flat_id = threadIdx.y * OUTER_THREADS_POS + threadIdx.x;
    const int blk_thread_count = OUTER_THREADS_POS * OUTER_THREADS_CH;

    // ---- reduction over input channels: one channel per pair of barriers ----
    for(int c = 0; c < d.C; c++){
        // (a) WEIGHTS: 3x3 filters of the 16 block channels, for input channel c
        //     sh_weights is [16][9] flattened: idx = w_ch * 9 + w_tap
        for(int idx  = thr_flat_id; idx < OUTER_CH_PER_BLOCK * 9; idx += blk_thread_count){
            int w_ch = idx / 9; // which of the 16 block channels (row)
            int w_tap = idx % 9;    // which tap r*3+s (column)
            int k = blk_ch_start + w_ch;    // absolute output channel
            // filter[k][c][r][s] = filter[((k*C + c)*R + r)*S + s], and r*S + s == w_tap
            sh_weights[idx] = filter[(k * d.C + c) * OUTER_TAPS + w_tap];
        }

        // (b) INPUTS: sh_in_rows x sh_in_cols of channel c. Shared row 0 is input row
        //     (blk_out_row_start - pad), shared col 0 is input col -pad. Out-of-image
        //     cells are written as 0: the zero padding, done once here.
        for(int idx = thr_flat_id; idx < sh_in_rows * sh_in_cols; idx += blk_thread_count){
            int sh_row = idx / sh_in_cols;
            int sh_col = idx % sh_in_cols;
            int in_row = blk_out_row_start - d.pad + sh_row;
            int in_col = -d.pad + sh_col;

            if(in_row >= 0 && in_row < d.H && in_col >= 0 && in_col < d.W){
                sh_input[idx] = input[((n * d.C + c) * d.H + in_row) * d.W + in_col];
            }else{
                sh_input[idx] = 0.0f;
            }
        }

        __syncthreads(); // both tiles complete before anyone reads them

        // (c) COMPUTE: per filter tap (r,s), one outer product per thread:
        //     4 weights (one per channel) x 4 inputs (one per position) = 16 MADs
        //     from 8 shared loads -> A_I = 2 = A_G.
        #pragma unroll
        for(int r = 0; r < OUTER_FILTER; r++){
            #pragma unroll
            for(int s = 0; s < OUTER_FILTER; s++){
                float w[OUTER_CH_PER_THREAD];
                float x[OUTER_POS_PER_THREAD];

                #pragma unroll
                for(int i  = 0; i < OUTER_CH_PER_THREAD; i++){
                    // row = this thread's i-th channel inside the block, column = tap
                    w[i] = sh_weights[(threadIdx.y * OUTER_CH_PER_THREAD + i) * OUTER_TAPS 
                                      + r * OUTER_FILTER +s];
                }

                #pragma unroll
                for(int j = 0; j < OUTER_POS_PER_THREAD; j++){
                    // no -pad: the halo offset is built into where the tile starts
                    x[j] = sh_input[(thr_row_in_blk + r) * sh_in_cols + (thr_out_col + j + s)];
                }

                #pragma unroll
                for(int i = 0; i < OUTER_CH_PER_THREAD; i++){
                    #pragma unroll
                    for(int j = 0; j < OUTER_POS_PER_THREAD; j++){
                        acc[i][j] += w[i] * x[j];
                    }
                }
            }
        }

        __syncthreads(); // all reads done before the next c overwrites the tiles
    }

    // ---- WRITE the 16 outputs. No bounds guard: the launcher only accepts shapes
    //      for which every block lies completely inside the output. ----
    #pragma unroll
    for(int i = 0; i < OUTER_CH_PER_THREAD; i++){
        int k = thr_ch_start + i;
        #pragma unroll
        for(int j = 0; j < OUTER_POS_PER_THREAD; j++){
            output[((n * d.K + k) * d.H + thr_out_row) * d.W + (thr_out_col + j)] = acc[i][j];
        }
    }
}

// Host launcher for the outer-product kernel. The kernel has no bounds guards,
// so the launcher refuses any shape for which a block would not be completely
// full. All three Mini-VGG layers satisfy these conditions.
void launch_conv_outer(const float* d_input, const float* d_filter, float* d_output, const ConvDims& d){
    bool ok = (d.R == OUTER_FILTER && d.S == OUTER_FILTER && d.pad == 1)
           && (d.K % OUTER_CH_PER_BLOCK == 0)                    // whole channel groups
           && ((d.H * d.W) % OUTER_POS_PER_BLOCK == 0)           // whole position groups
           && (OUTER_POS_PER_BLOCK % d.W == 0)                   // a block = whole output rows
           && (d.W % OUTER_POS_PER_THREAD == 0)                  // a thread's 4 positions in one row
           && ((OUTER_POS_PER_BLOCK / d.W + 2) * (d.W + 2) <= OUTER_SH_IN_MAX); // input tile fits

    if(!ok){
        fprintf(stderr, "launch_conv_outer: unsupported shape C=%d H=%d W=%d K=%d R=%d S=%d pad=%d\n",
                d.C, d.H, d.W, d.K, d.R, d.S, d.pad);
        exit(EXIT_FAILURE);
    }

    dim3 block(OUTER_THREADS_POS, OUTER_THREADS_CH);    // 16 x 4 = 64 threads
    // groups of 64 positions, // groups of 16 channels, // one image per z
    dim3 grid((d.H *d.W) / OUTER_POS_PER_BLOCK, d.K / OUTER_CH_PER_BLOCK, d.N);
    conv_forward_outer_kernel<<<grid, block>>>(d_input, d_filter, d_output, d);
}


// Correctness check vs cuDNN (same method as verify_conv_regtiled).
// Expect a tiny nonzero max abs diff (~1e-6): the summation order differs
// from cuDNN, which is legitimate floating-point behaviour, not an error.
void verify_conv_outer(cudnnHandle_t cudnn, convLayer& layer, float* d_input, void* d_workspace){
    int total_elements = layer.out_n * layer.out_c * layer.out_h * layer.out_w;
    size_t total_bytes = total_elements * sizeof(float);

    float* d_outer;
    float* d_cudnn;
    CHECK_CUDA(cudaMalloc(&d_outer, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_cudnn, total_bytes));

    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;

    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn, &alpha, layer.input_desc, d_input,
        layer.filter_desc, layer.d_filter, layer.conv_desc, layer.algo,
        d_workspace, layer.workspace_bytes,
        &beta_overwrite, layer.output_desc, d_cudnn));

    ConvDims d;
    d.N = layer.in_n;  d.C = layer.in_c;
    d.H = layer.in_h;  d.W = layer.in_w;
    d.K = layer.out_c;
    d.R = layer.kernel_size;  d.S = layer.kernel_size;
    d.pad = layer.kernel_size / 2;

    launch_conv_outer(d_input, layer.d_filter, d_outer, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_cudnn(total_elements);
    std::vector<float> h_outer(total_elements);
    CHECK_CUDA(cudaMemcpy(h_cudnn.data(), d_cudnn, total_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_outer.data(), d_outer, total_bytes, cudaMemcpyDeviceToHost));

    const float threshold = 1e-3f;
    float max_abs = 0.0f, max_rel = 0.0f;
    int max_abs_idx = -1, n_over = 0;

    for(int i = 0; i < total_elements; i++){
        const float a = h_cudnn[i];
        const float b = h_outer[i];
        const float abs_diff = fabsf(a-b);
        const float rel_diff = abs_diff / (fabsf(a) + fabsf(b) + 1e-8f);
        if(abs_diff > max_abs){ max_abs = abs_diff; max_abs_idx = i; }
        if(rel_diff > max_rel) max_rel = rel_diff;
        if(abs_diff > threshold) n_over++;
    }

    printf("\n=== VERIFY conv outer vs cuDNN ===\n");
    printf("  shape        : [%d, %d, %d, %d]  (%d elements)\n",
           layer.out_n, layer.out_c, layer.out_h, layer.out_w, total_elements);
    printf("  max abs diff : %.3e  (at index %d)\n", max_abs, max_abs_idx);
    printf("  max rel diff : %.3e\n", max_rel);
    printf("  elems > %.0e : %d\n", threshold, n_over);
    printf("  verdict      : %s\n", (max_abs < threshold ? "PASS" : "FAIL"));

    CHECK_CUDA(cudaFree(d_cudnn));
    CHECK_CUDA(cudaFree(d_outer));
}