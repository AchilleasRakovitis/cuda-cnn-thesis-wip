# Phase 2 — Progress Tracker

Anchored to: Lopes, "Open CUDA convolution neural network inference
implementation", Cluster Computing (2026) 29:105.

Scope: replace cuDNN forward convolution with custom CUDA kernels
(naive → shared-tiled → register-tiled → mutex/T0), benchmarked against a
pinned cuDNN baseline. Backward pass stays on cuDNN. Forward-only for now.

To resume in a new chat: pull the repo, read this file, continue from the
first milestone not marked DONE.

## Methodology (agreed with supervisor)
Reference implementation: Lopes, cuDconv — github.com/paclopes/cuDconv (MIT),
the code annex to the anchor paper.

Approach for M3-M6: independent implementation, guided by the reference.
- Where the reference implements a functionality (mutex/T0 accumulation, float4
  vectorized loads, T0-T4 tile hierarchy), we follow HIS technique/algorithm.
- We write it with OUR OWN variables, structures (ConvDims), naming, file
  organisation, and integration into the MiniVGG pipeline.
- Every technique taken from the reference is cited in a code comment above the
  relevant function and in the thesis text.
- The naive kernel (M1) is NOT from the reference: Lopes has no naive kernel, he
  starts from the optimised form. Ours is a first-principles derivation, the
  known-correct baseline and the degenerate case (all tile sizes = 1).
- The reference is used as an "answer key": attempt each milestone from
  understanding first, then consult cuDconv to verify the approach.

## Baseline (pinned, sm_75)
- Forward algo pinned to CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
- All 3 layers report fwd algo=1 (was 0/6/0 via nondeterministic Find*)
- Build targets -arch=sm_75 and -std=c++14

## Milestones
- [DONE] M0  Pin cuDNN algo — stable baseline (5%)
- [DONE] M1  Naive direct conv kernel + verification (20%)
- [DONE] M2  Benchmark harness: per-layer conv timing, CSV (30%)
- [DONE] M3  Shared-memory tiled (tile + halo in shared) (45%)
- [ ]    M4  Register tiled (multiple outputs/thread, occupancy check) (60%)
- [ ]    M5  Lopes techniques: AI>=AG derivation, float4 loads, ptxas verify (72%)
- [ ]    M6  Mutex / T0 split (partial-sum accumulation) (85%)
- [ ]    M7  Layer sweep + writeup data (VGG/ResNet shapes, final tables) (100%)

## Current: 45% (M0-M3 done)

## M3 results — shared-memory tiled kernel (T1 level)
Files: src/conv_kernels.cu (conv_forward_tiled_kernel, launch_conv_tiled,
verify_conv_tiled), include/conv_kernels.h, benchmark.h/.cu (CONV_TILED added).

Design:
- Each block computes a 16x16 output tile; loads an 18x18 input region into
  shared memory (16x16 body + 1-pixel halo; input_tile = output + R - 1 = 18).
- Channels processed ONE AT A TIME (18x18 per channel). All-at-once would need
  18*18*32*4 ~= 40 KiB > 21.3 KiB budget, so we loop over C (1296 B in shared).
- Load: grid-stride loop (256 threads, 324 cells), idx -> (local_row,col) via
  /18 %18, then -> image (ih,iw) via -pad offset, bounds check writes value or 0.
- Two __syncthreads() per channel (after load, after compute), OUTSIDE the
  bounds guard so all threads reach them (no deadlock).
- Compute reads tile[(threadIdx.y+r)*18 + (threadIdx.x+s)], NO bounds check:
  halo's +pad and filter's -pad cancel. Simpler inner loop.
- Corresponds to Lopes T1. Guided by cuDconv tensor_tile; no float4 / register
  tiling / intx yet (M4/M5).

Compilation (ptxas -v):
| Kernel | Registers | Shared mem | Spills |
|--------|-----------|------------|--------|
| naive  | 33        | 0 bytes    | 0 / 0  |
| tiled  | 61        | 1296 bytes | 0 / 0  |

Verification vs cuDNN: PASS on all 3 layers, max abs diff 0.000e+00
(1,835,008 elements total, exact). L3 (8x8 under 16x16 block) exercises the
bounds guard and barrier discipline hardest.

Benchmark (median ms, warmup 20, 100 iters):
| Layer | Channels | cuDNN | naive | tiled | tiled vs naive |
|-------|----------|-------|-------|-------|----------------|
| L1    | 3        | 0.058 | 0.078 | 0.083 | 1.06x SLOWER   |
| L2    | 16       | 0.062 | 0.178 | 0.199 | 1.12x SLOWER   |
| L3    | 32       | 0.084 | 0.404 | 0.348 | 1.16x FASTER   |

### THE FINDING (key defence point)
Tiling wins ONLY at L3, and is slower at L1/L2. Tiling has a fixed cost paid
every run (2C barriers, grid-stride load, extra indexing); its benefit (reduced
global traffic via reuse) grows with channel count. At 3 channels there is too
little reuse to amortise the sync cost, so tiling loses; at 32 channels reuse
dominates and tiling wins. Engineering principle: a shared-memory optimisation
is worthwhile only when the reuse it enables amortises its synchronisation and
loading overhead — knowing WHEN to apply it (and when not) is as much the point
as the optimisation itself. Secondary: at L3 tiled is also more STABLE
(median 0.348 ~ min 0.344) vs naive (median 0.404, min 0.295).
This motivates M4 (register tiling, spread cost over more arithmetic per thread)
and M5 (vectorized loads, intensity-tuned params) — evidence they are necessary,
not decorative.

## Session PDFs produced
- Phase2_M1_naive.pdf
- M1_Theoria_Perilipsi.pdf (Greek study sheet)
- Phase2_M2_benchmark.pdf
- Phase2_M3_tiled.pdf

## Key decisions log
- Scope C + mutex: staged progression adopting Lopes's method plus the full
  mutex/T0 accumulation as the faithful-reproduction piece.
- Baseline algo = IMPLICIT_PRECOMP_GEMM (closest to Lopes Y=WX', fair comparison).
- Find* block kept commented in conv_layer.cu as documented nondeterminism finding.
- ConvDims struct over flat args so tile parameters can be added in M4-M6.
- verify_* and benchmark write to their own buffers — no training side effects.
- Build needs -std=c++14: CUDA 11.5 + GCC 11 + cudnn.h fails to parse
  <functional> under default C++17. Same root cause as the Thrust incompatibility.
- tiled kernel hardcodes 16 (block) and 18 (shared tile); block(16,16) is now
  MANDATORY, not optional. Consider named constants (TILE, HALO, SH) later.

## Environment note
- Server GPU once went down mid-session with a driver/library version mismatch
  (nvidia-smi: NVML 580.178, stale kernel module). Fixed by admin reboot/module
  reload. If cuDNN fails at cudnnCreate with CUDNN_STATUS_NOT_INITIALIZED, check
  nvidia-smi first — it is an environment issue, not a code bug.

## Deferred (after implementation chapters written)
- Clean up main.cu (remove debug forward/backward, old timing, gate verify calls).
- Refactor: run_conv_once as single call site; verify_* stop duplicating the
  cudnnConvolutionForward call.
- Make tiled tile/block sizes named constants (TILE, HALO, SH).
- Possibly increase iters / add percentiles for more stable cuDNN numbers.