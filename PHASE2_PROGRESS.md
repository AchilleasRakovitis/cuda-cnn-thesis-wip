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
- Build targets -arch=sm_75 and -std=c++14
- Test acc: ~67% (small run-to-run variation is FP-order noise, expected)

## Milestones
- [DONE] M0  Pin cuDNN algo — stable baseline (5%)
- [DONE] M1  Naive direct conv kernel + verification (20%)
- [DONE] M2  Benchmark harness: per-layer conv timing, CSV, tolerance compare (30%)
- [ ]    M3  Shared-memory tiled (tile + halo in shared) (45%)
- [ ]    M4  Register tiled (multiple outputs/thread, occupancy check) (60%)
- [ ]    M5  Lopes techniques: AI>=AG derivation, float4 loads, ptxas verify (72%)
- [ ]    M6  Mutex / T0 split (partial-sum accumulation) (85%)
- [ ]    M7  Layer sweep + writeup data (VGG/ResNet shapes, final tables) (100%)

## Current: 30% (M0, M1, M2 done)

## M2 results — benchmark harness
Files: include/benchmark.h, src/benchmark.cu

Design:
- enum ConvImpl { CONV_CUDNN, CONV_NAIVE, CONV_IMPL_COUNT } selects which
  implementation is timed; adding a kernel later = one enum value + one switch case.
- run_conv_once(): fires one convolution (cuDNN or naive), no sync inside — the
  only synchronisation is the timing event, so measurement is not disturbed.
- bench_conv(): warmup (20) discarded, then 100 timed iterations, each wrapped in
  its own CUDA events; results sorted for median and min. GFLOP/s and %peak
  computed from FLOPs = 2*N*K*H*W*C*R*S against 16300 GFLOP/s (TITAN RTX FP32).
- run_all_benchmarks(): loops 3 layers x all impls, prints a table and writes
  benchmark_results.csv (for LaTeX tables). Called from main.cu behind a
  RUN_BENCHMARK flag with early return, so it does not run the training loop.

First results (single run, median ms):
| Layer | cuDNN | naive | naive slowdown |
|-------|-------|-------|----------------|
| L1 (C=3)  | 0.067 | 0.078 | 1.17x |
| L2 (C=16) | 0.070 | 0.178 | 2.56x |
| L3 (C=32) | 0.092 | 0.404 | 4.38x |

Interpretation: naive slowdown grows with input channel count (3 -> 16 -> 32),
because naive re-reads each operand from global memory C times with no reuse —
exactly the arithmetic-intensity argument, measured. Even cuDNN reaches only
5-13% of peak (3x3 conv on small images is memory-bound). naive is stable
(median ~ min); cuDNN shows more run-to-run variance, which justifies using the
median. NOTE: cuDNN numbers need a stability re-check (L2 median > L3 median is
suspicious; likely shared-server noise).

## Session PDFs produced
- Phase2_M1_naive.pdf
- M1_Theoria_Perilipsi.pdf (Greek study sheet)
- Phase2_M2_benchmark.pdf  (to be added)

## Key decisions log
- Scope C + mutex: staged progression adopting Lopes's analytical method plus the
  full mutex/T0 accumulation as the faithful-reproduction piece.
- Baseline algo = IMPLICIT_PRECOMP_GEMM (closest to Lopes Y=WX', fair comparison).
- Find* block kept commented in conv_layer.cu as documented nondeterminism finding.
- ConvDims struct chosen over flat args so tile parameters can be added in M3-M6.
- verify_conv_naive / benchmark write to their own buffers — no training side effects.
- Naive kernel is the degenerate case of Lopes tiling (all tile sizes = 1).
- Build needs -std=c++14: CUDA 11.5 + GCC 11 + cudnn.h fails to parse <functional>
  under the default C++17 (variadic std::function signatures). Same toolchain root
  cause as the known Thrust incompatibility.
- benchmark.cu is host-only code with .cu extension; median (not mean) chosen
  because the shared GPU server introduces outliers.
- main.cu NOT cleaned up yet — verify calls, manual forward/backward, old full-
  forward timing all still present. Deferred until Phase 1 implementation chapters
  are written, so nothing described in the thesis is deleted prematurely.

## Deferred (after implementation chapters written)
- Clean up main.cu (remove debug forward/backward, old timing, gate verify calls).
- Refactor: make run_conv_once the single call site, so verify_conv_naive stops
  duplicating the cudnnConvolutionForward call.
- Possibly increase iters / add percentiles for more stable cuDNN numbers.