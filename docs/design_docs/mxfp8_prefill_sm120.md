# [RFC] MXFP8 block-scaled prefill attention for SM120a (consumer Blackwell)

## Summary

Add an open-source, source-level **MXFP8** prefill / ragged attention kernel for
**SM120a / SM121a** (consumer Blackwell — GeForce RTX 50 series, GB20x).
MXFP8 here means the OCP microscaling FP8 format: `e4m3` data with per-32-element
`ue8m0` block scales, accumulating in FP32.

FlashInfer currently has no block-scaled FP8 attention that runs on SM120. This RFC
proposes one, motivated by a concrete hardware property of consumer Blackwell measured
below.

## Motivation

### 1. The block-scaled tensor path escapes the consumer FP32-accumulate throttle (measured)

On consumer Blackwell the FP32-accumulate tensor throughput is halved **for the legacy
warp-MMA instructions** — but the **block-scaled tensor instruction is not throttled**.

Register-resident MMA microbenchmark, RTX 5060 Ti (SM120, GB206, 36 SMs, CUDA 13.3),
pure tensor-core issue rate, no memory traffic:

| inputs / accumulate | SASS instruction | TFLOP/s |
|---|---|---:|
| FP16 / FP16 acc | `HMMA.16816.F16` | ~103 |
| FP16 / FP32 acc | `HMMA.16816.F32` | ~51 |
| FP8 e4m3 / FP32 acc (plain) | `QMMA.16832.F32.E4M3.E4M3` | ~102 |
| FP8 e4m3 / FP16 acc (plain) | `QMMA.16832.F16.E4M3.E4M3` | ~206 |
| **MXFP8 e4m3 + ue8m0 block-scaled / FP32 acc** | **`QMMA.SF.16832.F32.E4M3.E4M3.E8`** | **~202** |

Key ratios:

- **MXFP8 block-scaled / BF16(FP32-acc) = 3.95×** (202 vs 51 TFLOP/s).
- Legacy FP32-acc throttle = 2.0× (`HMMA.16816.F16` vs `HMMA.16816.F32`), and it equally
  halves *plain* FP8 (`QMMA...F32` 102 vs `QMMA...F16` 206) — but **not** the block-scaled
  path (`QMMA.SF` stays at ~202 with FP32 accumulate).

In other words: on consumer Blackwell the block-scaled MXFP8 MMA is the only way to get
full-rate tensor throughput *together with* FP32 accumulation. A per-tensor / plain FP8
attention kernel (plain `mma.sync`) only reaches 2× because it pays the FP32-acc throttle.

The measurement is verified, not inferred:
- SASS shows `QMMA.SF.16832.F32.E4M3.E4M3.E8` inside the timed loop with real accumulator
  read-modify-write dependency chains (no dead-code elimination).
- A correctness check (A=B=1.0, scale=1.0) returns the exact `K = 32` reduction on all
  output lanes.
- Stable across ILP (8 / 16 independent accumulators) and loop counts.

Repro is included at the end of this RFC.

### 2. There is no source-level block-scaled FP8 attention for SM120 today

| existing path | source? | runs on SM120? | scaling |
|---|---|---|---|
| `hopper/quantization/prefill_sm90` | source | no (SM90 only) | dequant → BF16, not native FP8 MMA |
| `fmha_v2` `e4m3_fp32_*_sm120` | source | kernel compiles, **disabled in the Python API** | per-tensor scalar (`scale_bmm1/2`) |
| trtllm-gen FMHA | prebuilt cubin | **no** — runner asserts `mSM == kSM_100 \|\| kSM_103` | block-scaled, datacenter-only |
| `mxfp8_gemm_cutlass_sm120` | source | yes | `ue8m0` block-32 — **GEMM only, never wired to attention** |

So the large installed base of consumer Blackwell GPUs has no open, tunable, block-scaled
FP8 attention — even though the MMA atoms, block-scaled layout and `ue8m0` scale handling
already exist in-tree (in the MXFP8 GEMM) and a working block-scaled SM120a attention exists
as a reference (SageAttention3, in NVFP4).

## Proposed design

**Target:** SM120a / SM121a only. Warp-level block-scaled
`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`
(SASS `QMMA.SF`), 1×1×1 cluster — the consumer-Blackwell warp path, *not* the tcgen05 /
tensor-memory path used on SM100/103.

### Implementation substrate: CUTLASS C++ (CuTe), not CuTe DSL

The exact atom we need already exists upstream in CUTLASS C++:

```cpp
cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,   // A = Q / P
    cutlass::float_e4m3_t,   // B = K / V
    float,                   // FP32 accumulate
    cutlass::float_ue8m0_t,  // block scale
    /*SFVecSize=*/32>
```

(`cute/arch/mma_sm120.hpp`, with the full `{e2m1,e2m3,e3m2,e4m3,e5m2}²` matrix and
`sm120_make_smem_layout_sfa/sfb` helpers.) The CuTe **DSL**, by contrast, only wires the FP4
block-scaled ops (`MmaMXF4Op`, `MmaMXF4NVF4Op`) on the SM120 *warp* path; its `MmaMXF8Op`
lives on the SM100 *tcgen05* path and is not reachable on SM120. So the MXFP8 atom is
callable from C++ but not from the DSL today — the kernel will be C++ CuTe.

**Toolchain confirmed on consumer Blackwell:** the upstream SM120 MXFP8 GEMM example
(`79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm`, which uses `mx_float8_t<e4m3>` +
`OpClassBlockScaledTensorOp` + `Sm120` + the `LayoutSFA/SFB` interleaved scale layout)
compiles and runs `Disposition: Passed` on an RTX 5060 Ti with CUDA 13.3 — so the atom,
the `ue8m0` scale-factor layout, and the element types are all proven working on the
target hardware.

### Architecture: hybrid (block-scaled core + FlashInfer shell)

| layer | source | notes |
|---|---|---|
| block-scaled fused mainloop — TMA warp-specialized load → QKᵀ block-scaled MMA → online softmax → **in-kernel P requant to `e4m3` + `ue8m0`** → PV block-scaled MMA | adapt **SageAttention3**'s SM120a attention mainloop | the only existing SM120a block-scaled *attention*; retarget NVFP4 → MXFP8 |
| MMA atom + SF smem layout | **upstream CUTLASS** (`SM120_16x8x32_TN_VS<e4m3,…>`, `sm120_make_smem_layout_sfa/sfb`) | replaces SageAttention's hand-rolled inline-PTX NVFP4 atom |
| K/V load, scheduler / plan, op registration, paged-KV (later) | **FlashInfer** (`blackwell/collective/*load*`, `gather_tensor.hpp`, `plan.cuh`) | the parts SageAttention lacks; paged-KV deferred to Phase 2 |

- **Data types:** Q/K/V in `e4m3` with `ue8m0` block-32 scales; P quantized in-kernel to
  `e4m3` + `ue8m0` block scales between the two matmuls; FP32 accumulation throughout (both
  QKᵀ and PV use the `QMMA.SF` path, which is why the kernel keeps the full-rate throughput).
- **Tiling note:** the atom shape differs from SageAttention3's FP4 path
  (FP4 is `16x32x64`, `scale_vec::4X`; MXFP8 is `16x8x32`, `scale_vec::1X`), so the
  `TiledMMA`, the SF layout, and the K-loop are re-derived rather than type-swapped.
- **SMEM budget:** SM120 exposes ~99 KB shared memory (vs 160+ KB on SM100/A100); tile sizes
  follow SageAttention3's SM120a-tuned shapes, not SM100's.

## API & integration

Following `CONTRIBUTING.md`:

- kernel: `include/flashinfer/attention/blackwell/quantization/`
- registration / binding: `csrc/`
- Python interface: `flashinfer/`
- JIT module registered with `supported_major_versions=[12]`
- tests under `tests/`, benchmarks under `benchmarks/`

## Accuracy

Validate against a BF16 reference across head_dim ∈ {64, 128}, causal and non-causal
(relative error / cosine similarity). MX block-32 scaling is expected to track K/V outliers
substantially better than the per-tensor scalar scaling used by the existing `fmha_v2` path.

## Scope / non-goals (Phase 1)

- **In:** single + ragged prefill, `e4m3`, head_dim 128, SM120a / SM121a.
- **Later:** paged-KV, SM100/103, decode, FP6/FP4 mixed-input.

## Open questions

- **P quantization between the two matmuls** — the main engineering risk: granularity and
  block layout for requantizing softmax probabilities P to `e4m3` + `ue8m0` so the scale
  vectors align with the PV MMA's SFB layout, and how much the in-kernel requant + scale
  computation costs.
- **K-loop / SF layout re-derivation** for the `16x8x32` MXFP8 atom (SageAttention3 is built
  around the `16x32x64` FP4 atom).
- **Integration with FlashInfer's prefill scheduler / plan kernel** and the ragged/varlen
  KV-load path; whether the K/V load stage can be factored so paged-KV (Phase 2) drops in via
  `gather_tensor` + page table.
- **head_dim 64 vs 128 tile shapes** under the ~99 KB SMEM budget.

## Reproducing the microbenchmark

```cuda
// nvcc -gencode arch=compute_120a,code=sm_120a -O3 mma_peak.cu -o mma_peak
// NOTE: -arch=sm_120a does NOT set the ptxas target to sm_120a; the block-scaled
//       instruction then fails with "not supported on .target 'sm_120'".
//       You must use -gencode arch=compute_120a,code=sm_120a.
asm volatile(
  "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col"
  ".f32.e4m3.e4m3.f32.ue8m0 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
  "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
  : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
  : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
    "r"(sfa),"h"(bidA),"h"(tidA), "r"(sfb),"h"(bidB),"h"(tidB));
```

Loop this over many independent accumulators with no memory traffic and divide
`2·M·N·K·(warp-MMAs)` by elapsed time to get the TFLOP/s in the table above.
