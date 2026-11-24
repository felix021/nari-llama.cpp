# INT8 prefill path for NariQ (experimental)

An optional prefill kernel that uses INT8 tensor cores (`mma.m16n8k32`) instead of
FP16 (`mma.m16n8k16`). It is a pure speed change: weights are decoded to the same
integer grid and activations are quantized to int8 with a per-row scale.

Measured on one RTX 5060 Ti 16GB (sm_120), Qwen3.8-27B NariQ 3.0bpw, `llama-bench -p 8192 -n 128 -b 2048 -ub 512 -fa on`:

| | pp8192 | tg128 |
|---|---:|---:|
| default (FP16 m16n8k16) | 674.8 tok/s | 31.7 tok/s |
| `TQ_I8=1` (INT8 m16n8k32) | **1110.3 tok/s** | 31.7 tok/s |

Quality against the FP16 path on 2 chunks of source text (`llama-perplexity --kl-divergence`):
`Same top p: 98.2%`, `RMS Δp: 1.36%`. The approximation comes only from the activation
quantization (per-row scale).

## Why it works

- The trellis grid is integer: `v = sum(four 6-bit digits) - 126`, so `v ∈ [-126, 126]`
  and the weights fit in signed int8 **exactly** (`common.cuh:72`). The FP16 path
  multiplies `4v` and scales by `0.25` in the epilogue; the INT8 path uses `v` directly.
- INT8 tensor throughput on this GPU is 4x the FP16-with-FP32-accumulate rate
  (measured: 207 TOPS vs 51.8 TFLOPS), which is what makes 1000+ tok/s possible at
  all — the FP16 path already runs at ~97% of its own peak.
- The bare FLOP count for the trellis linear layers is 48.2 GFLOP/token, so 1000 tok/s
  needs 48 TFLOP/s in the FP16 path — equal to the hardware peak. INT8 leaves ample room.

## What changed

1. `prologue_k` writes int8 activations (sequential `[k-tile][token][16 int8]` layout)
   with the existing per-row power-of-two scale `2^e`, shifted to `2^(e-7)` so the row
   max lands in `[64, 128)` before rounding/clamping to `[-127, 127]`.
2. `pf_k_i8` decodes both k-tiles with the existing `decode_ints`, repacks the ring
   layout by warp shuffle, and issues `mma.m16n8k32`.
   The ring layout gives each lane columns `2q, 2q+1, 2q+8, 2q+9` (m16n8k16 fragment);
   INT8 needs `4q..4q+3`. The repack takes 4 `__shfl_sync` per k-tile (the prototype
   measured 87% of INT8 peak with this included).
3. The epilogue converts the int32 accumulator to float, scales by 512 (`2^7 * 4`), and
   reuses the existing `finish_row` scales.
4. `TQ_I8` env var selects the path; default behavior is unchanged. Matrices with
   `splits > 1` (split-K partials go through `reduce_k`) stay on the FP16 path.

Also adds:
- `TQ_TILING=0|1|2` to force a tile shape (`T64X8` / `T128X8` / `T128X4`) for experiments.
  The default picker already selects the fastest shape on a 36-SM part (672.6 vs 635.7 vs 661.5).

## Use

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build -j --target llama-bench llama-server

TQ_I8=1 ./build/bin/llama-server -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj ... \
    -ngl 99 -c 131072 -fa on
```

## Limits / next steps

- Activation quantization is per-row; per-block scales (e.g. one per 32–64 K elements)
  should recover most of the remaining 1.8% top-1 gap at a small epilogue cost.
- At 1110 tok/s the inner loop is no longer MMA-bound; the remaining time is decode,
  cp.async traffic and the prologue. Further gains need pipelining work, not bigger MMA.
- `splits > 1` matrices stay FP16; they are the small projections (few percent of time).
