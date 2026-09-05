# SM120 QK MXFP8 / PV NVFP4 Attention

This document specifies the low-precision recipe and public contract of
`qk_mxfp8_pv_nvfp4_attention_sm120`. The implementation is an inference-only
forward kernel for SM120 and SM121 with head dimension 128.

## Quantization recipe

The preprocessing helper accepts contiguous FP16 or BF16 tensors in HND
layout:

- Q: `[batch, num_qo_heads, qo_len, 128]`
- K and V: `[batch, num_kv_heads, kv_len, 128]`

Query heads are mapped to compact K/V heads in the kernel. K/V are not
materialized at the query-head count.

The recipe is fixed as follows:

1. For each batch and K/V head, subtract the per-channel mean over the logical
   K sequence. Subtracting the same vector from every key shifts all valid
   logits in one query row by the same scalar, so it does not change the
   softmax probabilities or attention output.
2. Pad Q and K/V independently with zeros to multiples of 128 tokens.
3. Quantize Q and centered K to E4M3 values. Each contiguous group of 32 head
   features uses one UE8M0 scale. The scale is the smallest power of two that
   is at least `amax / 448`; conversion uses round-to-nearest, saturating
   finite E4M3 conversion. An all-zero group uses a zero scale and zero data.
4. Quantize V to E2M1 values. Each contiguous group of 16 sequence elements at
   a fixed value channel uses an E4M3 scale derived from `amax / 6`. V data and
   scales are stored transposed and in the SM120 block-scaled MMA layout.
5. The kernel computes QK with MXFP8 block-scaled MMA. Softmax probabilities
   are normalized and quantized to E2M1 in groups of 16 with UE4M3 scales,
   then P and V are multiplied with NVFP4 block-scaled MMA.

The scale tensors exposed by the low-level API have logical shapes, but their
bytes use the SM120 block-scaled MMA permutation. They must be produced by the
quantization helper or by code implementing the same physical layout.

Inputs are expected to be finite. Head dimensions other than 128, dropout,
arbitrary masks, backward propagation, and architectures other than SM120 or
SM121 are not supported.

## LSE semantics

`return_lse=False` is the default and selects a compile-time specialization
that neither allocates nor writes log-sum-exp values. `return_lse=True`
returns FP32 LSE for the effective, K-centered logits consumed by the kernel.
K centering does not change the attention output, but it shifts LSE by the
query-dependent centering constant relative to uncentered Q/K. The first API
version intentionally does not materialize that correction because inference
callers normally consume only the attention output.

## Sequence lengths and masking

The physical Q and K/V extents are padded multiples of 128. Callers must pass
`unpadded_q_len` and `unpadded_k_len` when the logical lengths differ from the
physical extents. Tail keys are excluded from softmax. Causal attention uses a
bottom-right aligned mask, including when query and key lengths differ:

```text
key_index <= query_index + kv_len - qo_len
```

Both aligned and partial lengths support MHA, GQA, and MQA as long as
`num_qo_heads % num_kv_heads == 0`.

## Kernel schedule

The noncausal mainloop uses the symmetric N64/N64 score-slot schedule derived
from the optimized SM120 NVFP4 attention path:

1. compute one N128 score tile and establish its row maximum;
2. quantize and consume the first N64 score slot in PV;
3. reuse the retired score registers for the next QK tile while consuming the
   second N64 slot; and
4. repeat with the two slots exchanged.

This shortens score-fragment live ranges and overlaps QK, softmax/P
quantization, and PV. The causal path retains guarded masking and softmax
handling for empty or partial score regions. Its noninitial tiles generate
normalized P directly and rescale the existing output accumulator in place,
avoiding a second output fragment. Long causal workloads use the single-tile
scheduler to reduce the imbalance caused by different per-row causal work.
Output-only and LSE paths are separate compile-time instantiations.

## Timing

`benchmarks/bench_qk_mxfp8_pv_nvfp4_attention_sm120.py` reports three distinct
measurements:

- attention-only latency on prequantized inputs, using CUDA Graph replay by
  default;
- Q/K/V preprocessing and quantization latency; and
- inclusive quantization plus attention latency.

The optional pure-NVFP4 comparison is informational: its quantization recipe
and QK tensor-core precision differ from this kernel. Performance claims for
this implementation should use the same QK-MXFP8/PV-NVFP4 recipe and identical
GPU clocks, inputs, software stack, and timing protocol.
