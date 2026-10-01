## Milestones
- [DONE] M0  Pin cuDNN algo (5%)
- [DONE] M1  Naive direct conv kernel + verification (20%)
- [DONE] M2  Benchmark harness (30%)
- [DONE] M3  Shared-memory tiled (45%)
- [DONE] M4  Register tiled (60%)
- [ ]    M5  Lopes techniques: AI>=AG derivation, float4 loads, ptxas verify (72%)
- [ ]    M6  Mutex / T0 split (85%)
- [ ]    M7  Layer sweep + writeup data (100%)

## Current: 60% (M0-M4 done, M4 benchmark corrected)

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

## TODO next: measure cuDNN Winograd separately
Add a WINOGRAD cuDNN entry to the benchmark to get the full picture: "vs implicit-GEMM we win,
vs Winograd we lose by X". ~10 lines. Closes the "what about Winograd?" question definitively.
Watch out: new switch case needs its own `break` (see bug fix above).

## Session PDFs
- Phase2_M1_naive.pdf, M1_Theoria_Perilipsi.pdf, Phase2_M2_benchmark.pdf,
  Phase2_M3_tiled.pdf, Phase2_M4_regtiled.pdf
- Phase2_Master_Reference.html (intuition of every M, updated per milestone)