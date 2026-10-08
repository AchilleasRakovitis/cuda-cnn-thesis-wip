## Milestones
- [DONE] M0  Pin cuDNN algo (5%)
- [DONE] M1  Naive direct conv kernel + verification (20%)
- [DONE] M2  Benchmark harness (30%)
- [DONE] M3  Shared-memory tiled (45%)
- [DONE] M4  Register tiled (60%)
- [DONE] M5  Lopes techniques: AI>=AG derivation, outer-product kernel, ptxas verify (72%)
           (float4 vector loads NOT done -> future work)
- [ ]    M6  Mutex / T0 split -- OUT OF SCOPE (time), future work
- [ ]    M7  Layer sweep -- OUT OF SCOPE (time), future work

## Current: CODE FROZEN at M5 (thesis writing + exam presentation).

## M4 results — register-tiled kernel (Lopes T4/T3)
Files: src/conv_kernels.cu (conv_forward_regtiled_kernel, launch_conv_regtiled,
verify_conv_regtiled), benchmark.h/.cu (CONV_REGTILED added).
Constants: REG_TPT=2 (outputs/thread/dim), REG_BLK=16, REG_OUT=32, REG_SH=34.

- Each thread computes a 2x2 output block; 16x16 block covers a 32x32 output tile.
- 4 accumulators, ONE 3x3 loop, weight read once and shared across all 4 (the reuse).
- Per-output bounds guards on the 4 writes; compute needs none (halo-padded shared).
- base_row/col = block term + thread term (write position); out_start = block term only (load).

Compilation (ptxas): 64 registers, 4624 B smem, 0 spills. Barely up from tiled's 61,
zero spills — reuse gained essentially free in occupancy. Far under Lopes 170 limit.

Verification vs cuDNN: PASS all 3 layers, max abs diff 0.000e+00 (exact).

### BUG FIX (commit 84cef32) — M4 benchmark was wrong
CONV_TILED case in run_conv_once had no `break` -> fell through into CONV_REGTILED.
The "tiled" column at commit 11af971 was tiled+regtiled (e.g. L1: 0.083 + 0.035 = 0.116).
Introduced when CONV_REGTILED was inserted after the tiled case (at M3 the fall-through
hit `default: break` and was harmless). Verification unaffected: verify_* does not use
run_conv_once, and regtiled wrote d_out last with a correct result.
The 11af971 commit-message claim "beats tiled everywhere (2.3-3.3x)" is WRONG.

Benchmark (median ms, corrected, single run, all impls in the same session):
| Layer | Output | cuDNN* | naive | tiled | regtiled | vs tiled    |
|-------|--------|--------|-------|-------|----------|-------------|
| L1    | 32x32  | 0.057  | 0.078 | 0.083 | 0.035    | 2.3x faster |
| L2    | 16x16  | 0.061  | 0.178 | 0.199 | 0.287    | 1.4x slower |
| L3    | 8x8    | 0.084  | 0.405 | 0.491 | 1.083    | 2.2x slower |
*cuDNN = pinned IMPLICIT_PRECOMP_GEMM (same algorithm class)

### FINDINGS (defence points)
1. Register tiling wins ONLY when the tile fits the layer. Idle threads per block:
   regtiled (32x32 output tile): L1 0%, L2 75%, L3 94%
   tiled    (16x16 output tile): L1 0%, L2 0%,  L3 75%
   At every layer, the kernel with fewer idle threads wins. (L2: 8x8 of 16x16 threads busy
   = 64/256 = 25%; same as counting outputs 16x16/32x32 — the 2x2/thread factor cancels.)
2. At L1, beats pinned cuDNN by 1.6x (9.8% vs 6.0% peak). HONEST FRAMING: baseline is
   IMPLICIT_PRECOMP_GEMM, not cuDNN's fastest; Winograd would likely win; Lopes did not
   beat cuDNN overall. Correct claim: "beats cuDNN's implicit-GEMM at L1", NOT "beats cuDNN".
3. A fixed tile size cannot serve all layer shapes: large thread tile helps big layers,
   hurts small ones -> motivates M5 (analytic tile size).
4. Methodology: L2/L3 timings vary up to ~40% BETWEEN sessions (e.g. naive L3 0.286 at
   11af971 vs 0.405 now; L1 stable). Compare only numbers from the same run. Investigate
   (repeat runs, nvidia-smi for other jobs/clocks) before producing final tables.

## cuDNN Winograd reference (DONE)
Files: benchmark.h/.cu — CONV_CUDNN_WINO, CONV_CUDNN_WINO_NONFUSED (benchmark-only).
run_conv_once now takes (algo, ws, ws_bytes) as params; bench_conv sets defaults
(pinned algo + shared ws) and, for Winograd, overrides them: algo, workspace size
queried (status checked manually, NOT CHECK_CUDNN, so NOT_SUPPORTED is reportable),
private cudaMalloc before warmup, freed at cleanup. Pinned baseline untouched.

Benchmark (median ms, single run, same session):
| Layer | C  | pinned | wino  | wino_nf | best ours        | ours vs best cuDNN |
|-------|----|--------|-------|---------|------------------|--------------------|
| L1    | 3  | 0.058  | 0.081 | 0.126   | regtiled 0.035   | 1.6x faster        |
| L2    | 16 | 0.062  | 0.047 | 0.074   | tiled 0.200      | 4.2x slower        |
| L3    | 32 | 0.084  | 0.056 | 0.055   | naive 0.404      | 7.3x slower        |
All Winograd variants supported on all 3 layers.

### FINDINGS (defence points)
1. Winograd beats implicit-GEMM at L2/L3 (1.3-1.5x), below Lavin's 2.25x: transforms cost.
2. At L1 (C=3) Winograd is SLOWER than implicit-GEMM. Saving = 20 mults per input channel
   per 2x2 tile (scales with C); transform cost is ~fixed -> at C=3 they cancel out.
   Stage-2 GEMM with inner dim C=3 is also inefficient. wino_nf worst (global-memory round trips).
3. At L1 regtiled beats all 3 measured cuDNN algos. HONEST FRAMING: "faster than the
   cuDNN algorithms measured" — FFT/GEMM not timed; NOT "faster than cuDNN".
4. Prediction for M7: at C=256/512 (VGG) Winograd advantage should approach 2.25x.

### Caveats
- GFLOP/s column uses direct-conv FLOP count -> for Winograd it is EFFECTIVE GFLOP/s.
  Compare on time, or label "effective" in thesis.
- Winograd output not verified (timing only). Optional: one-time tolerance compare
  vs pinned (expect ~1e-5, not exact — different arithmetic order).
- Optional step 4: print "N/A" row when supported == false (not triggered on these shapes).

## M5 results — outer-product kernel (Lopes Sec. 4, Y = W X')
Files: src/conv_kernels.cu (conv_forward_outer_kernel, launch_conv_outer, verify_conv_outer),
include/conv_kernels.h, benchmark.h/.cu (CONV_OUTER, with its break), main.cu (verify calls).
Constants: OUTER_CH_PER_THREAD=4, OUTER_POS_PER_THREAD=4, OUTER_CH_PER_BLOCK=16,
OUTER_POS_PER_BLOCK=64, OUTER_THREADS_CH=4, OUTER_THREADS_POS=16, OUTER_SH_IN_MAX=256,
OUTER_FILTER=3, OUTER_TAPS=9. Theory: Phase2_M5_theory.pdf.

Design (derived, not tuned):
- A_G = M_M / L_M = 64 FP32 cores / 32 banks = 2 on Turing (Lopes's Pascal GPU: 4).
- M3/M4 have A_I = 1 (one shared load per MAD; M4 reuses the GLOBAL weight, not shared input).
- Thread = 4 output channels x 4 positions, outer product per (c,r,s): 4 weights + 4 inputs
  from shared, 16 MADs -> A_I = 16/8 = 2 = A_G.
- Block = 16 channels x 64 positions of one image (largest tile dividing K = 16/32/64 and
  H*W = 1024/256/64) -> 0 idle threads on every layer. 64 threads (16 x 4).
- Grid = (H*W/64, K/16, N). Shared per input channel: sh_weights[16*9] + sh_input[(64/W+2)*(W+2)].
- One input channel per pair of barriers (P1 = 9; Lopes uses multi-channel P1).
- Launcher rejects shapes where a block would not be full (no bounds guards in the kernel).

Compilation (ptxas): 64 registers, 1600 B smem, 0 spills, 0 stack frame
-> 65536/64 = 1024 threads = 16 blocks of 64 = full occupancy (theory).
Verification vs cuDNN: PASS all 3 layers, max abs diff 0.000e+00 (exact; same c,r,s summation order).
Index logic also checked by CPU emulation vs naive conv (exact) before the GPU run.

Benchmark: two consecutive runs in one session (median ms):
| Layer | outer         | pinned cuDNN  | wino          | wino_nf       | regtiled | outer vs best cuDNN | vs pinned   |
|-------|---------------|---------------|---------------|---------------|----------|---------------------|-------------|
| L1    | 0.0179/0.0179 | 0.0584/0.0580 | 0.0819/0.0817 | 0.1259/0.1268 | 0.0355   | 3.2x faster         | 3.2x        |
| L2    | 0.0299/0.0388 | 0.0620/0.0615 | 0.0413/0.0471 | 0.0668/0.0752 | 0.2863   | 1.2-1.4x faster     | 1.6-2.1x    |
| L3    | 0.0431/0.0430 | 0.0714/0.0840 | 0.0472/0.0471 | 0.0490/0.0491 | 0.8069   | 1.1x faster         | 1.7-2.0x    |
Peak (effective for wino): outer 19% / 24-31% / 22% of 16.3 TFLOP/s.

### FINDINGS (defence points)
1. outer is the fastest implementation on EVERY layer, in both runs, against all three measured
   cuDNN algorithms, and 2.0x / 4.6-5.9x / 6.6-9.4x faster than the best earlier own kernel
   per layer (regtiled at L1, naive at L2 and L3).
2. Why: A_I 1 -> 2 (= A_G), weights staged in shared once per block per channel instead of a
   global load per inner iteration, no idle threads on any layer, 64 regs / 0 spills -> full occupancy.
3. Resolves the M4 lesson: the tile is sized from the layer shapes (channels x positions), not
   spatially, so it fits all three layers.
4. HONEST FRAMING: "faster than the cuDNN algorithms measured (IMPLICIT_PRECOMP_GEMM, WINOGRAD,
   WINOGRAD_NONFUSED) on these small CIFAR-10 shapes". NOT "faster than cuDNN": GEMM, FFT and
   IMPLICIT_GEMM were not timed; cuDNN is tuned for large layers; Lopes did not beat cuDNN on his
   larger VGG/ResNet shapes. The L3 margin over Winograd (~10%) is small.

### Caveats
- Run-to-run variance (finding 4 of M4) reappeared WITHIN the session: L2 outer 0.030 vs 0.039,
  L3 cuDNN 0.071 vs 0.084, L3 naive 0.285 vs 0.404 (the same two values seen in earlier sessions:
  bimodal, likely GPU clock/shared-server state). L1 stable (<1%). The ranking is identical in both
  runs, so report both runs / ranges, not a single number. Not investigated further (time).
- Kernel code written by Claude at the student's request after the joint design and index setup;
  the student must be able to explain every line (walkthrough planned before the write-up).
- Not done: float4 vector loads, T0 mutex split, multi-channel P1, layer sweep (future work).

## Session PDFs
- Phase2_M1_naive.pdf, M1_Theoria_Perilipsi.pdf, Phase2_M2_benchmark.pdf,
  Phase2_M3_tiled.pdf, Phase2_M4_regtiled.pdf, Phase2_M5_theory.pdf, Phase2_M5_outer.pdf
- Phase2_Master_Reference.html (intuition of every M, updated per milestone)