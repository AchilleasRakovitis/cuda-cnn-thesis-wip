# Phase 2 — Progress Tracker

Anchored to: Lopes, "Open CUDA convolution neural network inference
implementation", Cluster Computing (2026) 29:105.

Scope: replace cuDNN forward convolution with custom CUDA kernels
(naive → shared-tiled → register-tiled → mutex/T0), benchmarked against a
pinned cuDNN baseline. Backward pass stays on cuDNN. Forward-only for now.

To resume in a new chat: pull the repo, read this file, continue from the
first milestone not marked DONE.

## Baseline (pinned)
- Forward algo pinned to CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
- All 3 layers report fwd algo=1 (was 0/6/0 via nondeterministic Find*)
- Test acc after pin: 67.47% (was 67.69%; diff is FP-order noise, expected)
- Full forward timing: 0.194 ms/pass (IMPLICIT_PRECOMP_GEMM is slower than the
  Find* mix that used Winograd on L2 — deliberate: honest apples-to-apples baseline)

## Milestones
- [DONE] M0  Pin cuDNN algo — stable baseline (5%)
- [ ]    M1  Naive direct conv kernel + enum + conv_kernels files (20%)
- [ ]    M2  Benchmark harness: per-layer conv timing, CSV, tolerance compare (30%)
- [ ]    M3  Shared-memory tiled (tile + halo in shared) (45%)
- [ ]    M4  Register tiled (multiple outputs/thread, occupancy check) (60%)
- [ ]    M5  Lopes techniques: AI>=AG derivation, float4 loads, ptxas verify (72%)
- [ ]    M6  Mutex / T0 split (partial-sum accumulation) (85%)
- [ ]    M7  Layer sweep + writeup data (VGG/ResNet shapes, final tables) (100%)

## Current: 5% (M0 done)

## Session PDFs produced
- (none yet)

## Key decisions log
- Scope C + mutex: staged progression that adopts Lopes's analytical method,
  plus the full mutex/T0 accumulation as the faithful-reproduction piece.
- Baseline algo = IMPLICIT_PRECOMP_GEMM (closest to Lopes Y=WX', fair comparison).
  Winograd kept separately as "cuDNN's best" number, not the main baseline.
- Find* block kept commented in conv_layer.cu as documented nondeterminism finding.
