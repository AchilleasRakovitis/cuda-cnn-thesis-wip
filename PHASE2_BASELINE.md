# Phase 1 Baseline — reference before Phase 2

Captured before any Phase 2 (custom CUDA convolution) code was written.
This is the "before" state that all Phase 2 work is measured against.

## Configuration
- Network: Mini-VGG, 3 conv layers (3→16→32→64), all 3×3, "same" padding
- Batch size N = 64
- CIFAR-10, 50000 train / 10000 test, 781 batches/epoch
- lr = 0.01, 20 epochs, plain SGD (no momentum)
- GPU: TITAN RTX, CUDA 11.5, cuDNN 9.3.0

## Layer shapes
| Layer | In → Out | Conv output | After pool |
|-------|----------|-------------|------------|
| L1 | 3 → 16 | [64,16,32,32] | [64,16,16,16] |
| L2 | 16 → 32 | [64,32,16,16] | [64,32,8,8] |
| L3 | 32 → 64 | [64,64,8,8] | [64,64,4,4] |

## Accuracy (final)
- epoch 19: avg train loss = 0.644203, **test acc = 67.6883%**
- Overfitting onset ~epoch 16-17 (train loss keeps falling, test acc slows)
- Single-batch initial loss: 2.43104

## Timing
- Full forward pass: **0.160199 ms/pass** (16.0199 ms / 100 iterations)
- NOTE: this is the WHOLE forward (all convs + bias + ReLU + pool + FC).
  Phase 2 replaces ONLY the conv operation, so per-layer conv timing
  is still needed for a fair comparison (built in the benchmark harness).

## cuDNN algorithm selection (via Find* — NONDETERMINISTIC)
| Layer | fwd algo | bwd_filter | bwd_data | fwd workspace |
|-------|----------|------------|----------|---------------|
| L1 | 0 (IMPLICIT_GEMM) | 0 | 0 | 0 |
| L2 | 6 (WINOGRAD_NONFUSED) | 0 | 4 | 52352 |
| L3 | 0 (IMPLICIT_GEMM) | 0 | 4 | 0 |

Shared workspace allocated: 204928 bytes

### Key point
Find* picked DIFFERENT algorithms per layer (0, 6, 0) and is not guaranteed
to pick the same on the next run. This is the documented nondeterminism finding.
Phase 2 benchmarking must PIN the forward algorithm so the cuDNN baseline is
stable run-to-run, and compare numerically with tolerance (not bitwise).
