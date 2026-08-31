# Phase 2 — Progress Tracker

Anchored to: Lopes, "Open CUDA convolution neural network inference
implementation", Cluster Computing (2026) 29:105.

Scope: replace cuDNN forward convolution with custom CUDA kernels
(naive → shared-tiled → register-tiled → mutex/T0), benchmarked against a
pinned cuDNN baseline. Backward pass stays on cuDNN. Forward-only for now.

To resume in a new chat: pull the repo, read this file, continue from the
first milestone not marked DONE.

## Baseline (pinned, sm_75)
- Forward algo pinned to CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
- All 3 layers report fwd algo=1 (was 0/6/0 via nondeterministic Find*)
- Build now targets -arch=sm_75 (was defaulting to sm_52 / JIT)
- Test acc: ~67% (small run-to-run variation is FP-order noise, expected)
- Full forward timing: 0.191 ms/pass
  NOTE: whole forward (convs + bias + ReLU + pool + FC). Per-layer conv
  timing is still needed for fair comparison — that is M2.

## Milestones
- [DONE] M0  Pin cuDNN algo — stable baseline (5%)
- [DONE] M1  Naive direct conv kernel + verification (20%)
- [ ]    M2  Benchmark harness: per-layer conv timing, CSV, tolerance compare (30%)
- [ ]    M3  Shared-memory tiled (tile + halo in shared) (45%)
- [ ]    M4  Register tiled (multiple outputs/thread, occupancy check) (60%)
- [ ]    M5  Lopes techniques: AI>=AG derivation, float4 loads, ptxas verify (72%)
- [ ]    M6  Mutex / T0 split (partial-sum accumulation) (85%)
- [ ]    M7  Layer sweep + writeup data (VGG/ResNet shapes, final tables) (100%)

## Current: 20% (M0, M1 done)

## M1 results
Files: include/conv_kernels.h, src/conv_kernels.cu
- conv_forward_naive_kernel: one thread per output element (n,k,p,q)
- Thread mapping: q,p from block/thread x,y; n = blockIdx.z / K, k = blockIdx.z % K
- Launch: block(16,16), grid(ceil(W/16), ceil(H/16), N*K)
- ptxas @ sm_75: 33 registers, 0 bytes smem, 0 spill stores, 0 spill loads

Verification vs cuDNN (verify_conv_naive), all three layers:
| Layer | Shape          | Elements  | max abs diff | Verdict |
|-------|----------------|-----------|--------------|---------|
| L1    | [64,16,32,32]  | 1,048,576 | 0.000e+00    | PASS    |
| L2    | [64,32,16,16]  |   524,288 | 0.000e+00    | PASS    |
| L3    | [64,64, 8, 8]  |   262,144 | 0.000e+00    | PASS    |

Exact (bit-for-bit) agreement across 1,835,008 elements. The naive kernel
accumulates in the same c→r→s order as IMPLICIT_PRECOMP_GEMM, so no
floating-point reordering occurs. L3 is the important case: output is 8x8
while the block is 16x16, so 192 of 256 threads exit at the bounds guard —
this exercises the guard that L1/L2 do not.

## Session PDFs produced
- Phase2_M1_naive.pdf

## Key decisions log
- Scope C + mutex: staged progression that adopts Lopes's analytical method,
  plus the full mutex/T0 accumulation as the faithful-reproduction piece.
- Baseline algo = IMPLICIT_PRECOMP_GEMM (closest to Lopes Y=WX', fair comparison).
  Winograd kept separately as "cuDNN's best" number, not the main baseline.
- Find* block kept commented in conv_layer.cu as documented nondeterminism finding.
- ConvDims struct chosen over flat int args so tile parameters can be added in M3-M6.
- verify_conv_naive writes to its own buffers, never to layer.d_conv_out, so it
  has no side effects on the training pipeline.
- Naive kernel is the degenerate case of Lopes tiling (all tile sizes = 1):
  same arithmetic, zero reuse, everything from global memory.