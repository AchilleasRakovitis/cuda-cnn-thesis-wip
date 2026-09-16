## Milestones
- [DONE] M0  Pin cuDNN algo (5%)
- [DONE] M1  Naive direct conv kernel + verification (20%)
- [DONE] M2  Benchmark harness (30%)
- [DONE] M3  Shared-memory tiled (45%)
- [DONE] M4  Register tiled (60%)
- [ ]    M5  Lopes techniques: AI>=AG derivation, float4 loads, ptxas verify (72%)
- [ ]    M6  Mutex / T0 split (85%)
- [ ]    M7  Layer sweep + writeup data (100%)

## Current: 60% (M0-M4 done)

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

Benchmark (median ms, 2 runs, regtiled stable <2%):
| Layer | cuDNN* | naive | tiled | regtiled | vs tiled |
|-------|--------|-------|-------|----------|----------|
| L1    | 0.059  | 0.078 | 0.116 | 0.035    | 3.3x faster |
| L2    | 0.062  | 0.178 | 0.484 | 0.213    | 2.3x faster |
| L3    | 0.071  | 0.286 | 1.118 | 0.772    | 1.4x faster |
*cuDNN = pinned IMPLICIT_PRECOMP_GEMM (same algorithm class)

### FINDINGS (defence points)
1. Register tiling beats shared tiling on EVERY layer (2.3-3.3x), 0 spills — theory confirmed.
2. At L1, beats pinned cuDNN by 1.7x (9.8% vs 5.9% peak). HONEST FRAMING: baseline is
   IMPLICIT_PRECOMP_GEMM, not cuDNN's fastest; Winograd would likely win; Lopes did not
   beat cuDNN overall. Correct claim: "beats cuDNN's implicit-GEMM at L1", NOT "beats cuDNN".
3. At L3, naive still beats regtiled: 32x32 block over 8x8 output = 94% idle threads.
   Large thread tile helps big layers, hurts small ones -> motivates M5 (analytic tile size).

## TODO next: measure cuDNN Winograd separately
Add a WINOGRAD cuDNN entry to the benchmark to get the full picture: "vs implicit-GEMM we win,
vs Winograd we lose by X". ~10 lines. Closes the "what about Winograd?" question definitively.

## Session PDFs
- Phase2_M1_naive.pdf, M1_Theoria_Perilipsi.pdf, Phase2_M2_benchmark.pdf,
  Phase2_M3_tiled.pdf, Phase2_M4_regtiled.pdf
- Phase2_Master_Reference.html (intuition of every M, updated per milestone)