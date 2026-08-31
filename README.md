# Skye

**Portable low-bit CPU kernels written in Rust, currently focused on NVFP4 × FP16 inference on x86-64.**

Skye explores how low-bit formats that are normally consumed by specialized hardware can be executed efficiently on ordinary CPUs.

The current MVP implements an NVFP4 weight × FP16 activation GEMV path using AVX2, F16C, and FMA without first expanding the full weight matrix into FP16 or FP32.

## Motivation

Low-bit inference reduces model storage and memory traffic, but a quantized format is only useful when the target hardware can consume it efficiently.

A naive CPU implementation can erase much of the benefit by performing:

```text
packed low-bit weights
        ↓
full dequantization
        ↓
temporary FP16 / FP32 weights
        ↓
matrix-vector multiplication
```

Skye instead performs decode and computation together:

```text
packed NVFP4 weights
        │
        ├─ unpack E2M1 nibbles
        ├─ decode block scale
        └─ load FP16 activations
                    ↓
             vector multiply
                    ↓
               FP32 accumulate
                    ↓
              global scaling
```

The dequantized weight matrix is never materialized as a separate full-size buffer.

## Current NVFP4 representation

`NvFp4Matrix` stores:

```text
weights.data
    two E2M1 FP4 values / byte

weights.scales
    one E4M3 scale / 16 weights

weights.global_scale
    one FP32 global scale
```

For a matrix with `rows × cols` weights:

```text
packed weight bytes = rows × cols / 2
block scales        = rows × cols / 16
```

The current format requires the column dimension to be divisible by 16.

## AVX2 kernel

The optimized x86-64 path requires:

- AVX2
- F16C
- FMA

Each 16-weight block is processed by:

1. loading 8 packed bytes,
2. splitting low/high FP4 nibbles,
3. decoding E2M1 through SIMD lookup,
4. converting FP16 activations to FP32,
5. multiplying and accumulating with FMA,
6. applying the E4M3 block scale,
7. applying the matrix global scale.

The implementation keeps partial sums in AVX registers and performs horizontal reduction only after the row has been processed.

## Implemented paths

| Path | Weight | Activation | Accumulation | Purpose |
|---|---|---|---|---|
| `gemv_scalar` | NVFP4 | FP16 | FP32 | reference implementation |
| `gemv_avx2` | NVFP4 | FP16 | FP32 | optimized fused path |
| `gemv_fp16_avx2` | FP16 | FP16 | FP32 | FP16 comparison baseline |

The current MVP is single-threaded so that the low-level SIMD behavior can be evaluated independently from thread scheduling.

## Correctness

The test suite checks:

- all 16 E2M1 values,
- known E4M3 scale values,
- scalar GEMV behavior,
- randomized NVFP4 matrices,
- AVX2 output against the scalar reference.

Because scalar and SIMD accumulation orders differ, AVX2 correctness is checked with a numerical tolerance rather than bit-identical equality.

Run:

```bash
cargo test
```

## Benchmark

The Criterion benchmark compares:

```text
NVFP4 scalar
vs.
NVFP4 AVX2 fused
vs.
FP16 AVX2
```

for the following matrix sizes:

```text
1024 × 1024
4096 × 4096
8192 × 8192
```

Run:

```bash
cargo bench --bench gemv
```

Benchmark throughput is reported in logical matrix elements processed per second.

### Interpreting the benchmark

The important comparison is not only scalar vs. SIMD.

Skye also includes an AVX2 FP16 path so the NVFP4 kernel can be evaluated against a conventional higher-precision representation on the same CPU instruction set.

This separates:

- the benefit of SIMD acceleration,
- the cost of FP4 decoding,
- and the memory-traffic advantage of packed low-bit weights.

## Project structure

```text
src/
├── format.rs
│   └── NvFp4Matrix
│
└── kernel/
    ├── scalar.rs
    │   └── scalar NVFP4 reference
    │
    ├── avx.rs
    │   └── AVX2 + F16C + FMA NVFP4 GEMV
    │
    └── fp16.rs
        └── AVX2 FP16 GEMV baseline

tests/
└── correctness.rs

benches/
└── gemv.rs
```

## Scope and limitations

Skye is currently a kernel-level prototype, not a complete inference runtime.

The current implementation does **not** claim:

- end-to-end LLM inference acceleration,
- optimized multithreading,
- AVX-512 or ARM support,
- full GEMM coverage,
- or integration with an existing inference engine.

The present goal is narrower: demonstrate that packed NVFP4 data can be consumed directly by a portable CPU SIMD kernel without expanding the full weight matrix first.

## Build

Requires Rust and an x86-64 CPU for the optimized path.

```bash
cargo build --release
cargo test
cargo bench --bench gemv
```
