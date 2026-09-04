# ulysses_lowp: Low-Precision Ulysses All-to-All Primitives

*Design Document · v1.0 · September 2026*

> For API reference see `flashinfer.comm.ulysses_lowp`.
> For benchmark methodology and numbers see `benchmarks/comm/bench_ulysses_lowp.py`.

## 1. Motivation

Sequence-parallel Ulysses attention scatters the local Q/K/V shard to all
ranks before attention and gathers the output after. On eight ranks with
sequence length 8 192 and head dim 128, each all-to-all moves roughly 1.2 GB
of BF16 per rank. On NVLink this is typically the dominant pre-attention
cost.

SageAttention2 already quantizes its Q/K/V inputs before attention; the
baseline pipeline dequantizes→BF16-A2A→requantizes, discarding the numerical
work done on the sender side. `ulysses_lowp` eliminates this round-trip:

- The **sender** quantizes on the final global scales (which depend on the
  gathered statistics) and packs into a sectioned, destination-major payload.
- The **A2A** moves the quantized payload — roughly half the bytes of BF16.
- The **receiver** unpacks directly into pre-quantized SageAttention2
  operands, with no dequantize/requantize step.

The result is approximately the same BF16 payload as naive A2A but at half
the bytes, while the SageAttention2 quality level is preserved because the
quantization was done on global scales.

## 2. Design Principles

- **FlashInfer provides only compute.** Transport is the caller's
  responsibility (NCCL, torch.distributed, or any other communicator). The
  stats all-gather and payload all-to-all are explicit caller-controlled
  steps, not hidden inside FlashInfer calls.
- **Byte-exact output by construction.** The quantized payload is guaranteed
  to be identical to what would result from gathering the full BF16 sequence
  and quantizing in one shot. The design is validated by a bit-exact anchor
  test against a reference implementation.
- **Protocol is auto-selected, not caller-specified.** `stats_protocol_for(L,
  world_size)` returns the optimal protocol for the given sequence length and
  world size; callers do not need to reason about protocol choice.
- **Fused-projection inputs are first-class.** The common `[B, L, H, 3, D]`
  interleaved QKV tensor from fused projections is accepted without copies;
  Q and K/V are exposed as separate pack entry points to allow pipelining Q
  packing over the stats all-gather.

## 3. Stats Protocol

Two stats protocols handle the case where a quantization group straddles a
rank boundary. Protocol is chosen automatically by
`stats_protocol_for(L, world_size)`.

### aligned path — ALIGN-128 (fused single-pass)

**Condition:** `L % 128 == 0`, where `L` is the **local shard length** at
forward time (i.e. `global_sequence / world_size` for the current request).

Every rank's local shard starts and ends on a 128-token boundary, so no
quantization group straddles a rank edge. A single fused
amax + quant + pack kernel is sufficient. The flow:

```
local_stats(local_kv)         →  k_sum_local, v_amax_local   [per rank]
all_gather(local_stats)        →  k_sum_all,  v_amax_all      [1 round]
finalize_stats(k_sum_all, ...)  →  k_mean, v_scale            [per rank]
quant_and_pack(q, kv, stats)  →  payload                     [per rank, fused]
```

The protocol is selected **per request** by `stats_protocol_for(L,
world_size)` at forward time, not at server startup. Whether aligned path is
reached depends entirely on the packed sequence length of each incoming
request: if the caller pads the global sequence to a multiple of
`128 × world_size`, the local shard is always a multiple of 128 and the
aligned path is always taken. The environment variable
`SGLANG_MINIMAX_H3_PACKED_ALIGNMENT` (or its equivalent server argument) is
an optional performance hint that controls this padding — it does not switch
protocols globally; requests that happen to be aligned take aligned path
regardless.

### boundary-merge path — boundary-amax merge

**Condition:** `L % 128 != 0`

The first and last 64-token K groups on each rank may straddle boundaries.
boundary-merge path adds a boundary all-gather step to collect partial amax values
across the adjacent ranks and merge them before quantizing boundary groups:

```
local_stats(local_kv)                    →  k_sum_local, v_amax_local,
                                             boundary_amax_local
all_gather(local_stats + boundary_amax)  →  full stats + all boundary amaxes
finalize_stats(...)                      →  k_mean, v_scale, merged boundary amax
quant_and_pack(q, kv, stats)             →  payload  [boundary groups use merged amax]
```

On 128-aligned shards, boundary-merge path and aligned path produce byte-identical
payloads (covered by the test suite).

## 4. Payload Layout

The payload is a flat bytes buffer with a fixed-order section layout. All
sections are contiguous with no inter-section padding; the buffer ends with a
128-byte zero tail for alignment.

The layout is determined by three compiled-in constants: `Q_GROUP = 32`,
`K_GROUP = 64`, `HEAD_DIM = 128`. The Python layer verifies these match the
compiled kernel at import time via `capability()`; a mismatch means the
kernel was built with different grouping parameters and the path is disabled.

```
Section       Dtype       Shape                         Notes
──────────────────────────────────────────────────────────────────────────────
Q int8        int8        [B, P, H, D]                  destination-major
K int8        int8        [B, P, H, D]
V fp8 E4M3    uint8       [B, 128, H, P×D]              V transposed; seq dim
                                                        padded to 64 multiple
Q scales      float32     [B, H, ceil(S/128), 4]        per-32-token warp scale
K scales      float32     [B, H, ceil(S/64)]            per-64-token group scale
Zero tail     uint8       128                           alignment sentinel
```

Where `S` is the local sequence length before A2A, `P` is the world size, and
`D` is head dim (128). `capability()` verifies at import time that the
compiled kernel's `Q_GROUP`, `K_GROUP`, and `HEAD_DIM` match these values.

Section offsets and sizes are computed by `payload_spec(B, S, H, P)` so
callers never hand-compute byte offsets. `payload_buffer(B, S, H, P)` returns
a pre-allocated `torch.uint8` buffer of the correct size.

## 5. API Surface

All public symbols live in `flashinfer.comm.ulysses_lowp`.

### Capability gate

```python
capability() -> bool
```

Returns `True` when the compiled kernel's `Q_GROUP`, `K_GROUP`, and
`HEAD_DIM` constants match the Python-side values and the device is SM120.
Import succeeds on all platforms; callers that depend on the lowp path
should gate on `capability()` rather than catching import errors.

### Stats flow

```python
k_sum_v_amax(kv)  ->  (k_sum_local, v_amax_local)
local_stats(kv)   ->  StatsContext         # wraps k_sum and boundary amax
finalize_stats(gathered_stats) -> V2GStats  # k_mean, v_scale, merged boundary
```

`V2GStats` is a frozen dataclass; it is the only object passed between the
stats and pack phases and is safe to pickle or send across processes.

### Pack and unpack

```python
quant_and_pack(q, kv, stats, payload) -> None   # writes into caller-owned payload buffer
# or, for pipelining Q over the stats all-gather:
quant_q_into_payload_fused(q, payload) -> None
quant_kv_into_payload_fused(kv, stats, payload) -> None

unpack_for_sage(payload, dst_q, dst_k, dst_v, dst_q_scale, dst_k_scale) -> None
```

`unpack_for_sage` applies SageAttention2's sequence permutation and writes
into caller-allocated pre-quantized operand tensors. The `scale_sequence`
parameter selects the scale layout expected by the consumer
(`'per_warp'` or `'per_head'`; default `'per_warp'`).

### Routing helpers

```python
stats_protocol_for(L, world_size) -> Literal[2, 3]
required_alignment(world_size)    -> int      # minimum L multiple for aligned path
aligned_length(L, world_size)     -> int      # smallest L' >= L satisfying alignment
payload_spec(B, S, H, P)         -> PayloadSpec
payload_buffer(B, S, H, P)       -> torch.Tensor   # dtype=uint8
```

## 6. Kernel Chain

Six CUDA kernels are exposed through TVM FFI; each has a corresponding
`TraceTemplate` for `flashinfer trace`:

| Kernel | Purpose | PDL |
|--------|---------|-----|
| `k_sum_v_amax` | Two-stage deterministic K-sum and V per-channel amax | ✓ |
| `k_boundary_minmax` | Partial amax at rank-boundary K groups (boundary-merge path) | ✓ |
| `merge_boundary_amax` | All-reduce of boundary amaxes across ranks | — |
| `derive_k_boundary_amax` | Final per-boundary-group amax after merge | ✓ |
| `quant_and_pack` | Fused int8 Q/K + fp8 V quantize-and-pack into payload | ✓ |
| `unpack_for_sage` | Payload unpack into SageAttention2 pre-quantized operands | ✓ |

PDL (`cudaLaunchKernelEx` with `enable_pdl=True`) is on by default when
`device_support_pdl()` returns `True`. It can be overridden per-call for
profiling.

The K-sum reduction is deterministic by construction: fixed 256-token chunks,
fixed reduction order, no atomics. All ranks derive identical K means and V
scales from the same gathered stats buffer.

## 7. Platform Scope

The kernel byte-anchoring is validated under `sm_120a` (RTX PRO 6000
Blackwell; `compute_120f` FMA semantics). JIT compilation pins
`-gencode sm_120a` on SM 12.0 devices. AOT prebuilds for SM120 only.

`head_dim == 128` is a hard requirement. `world_size ∈ {2, 4, 6, 8}` is the
tested range (P = 6 is admissible).

## 8. Out of Scope

This module does not provide:

- **Transport.** The caller supplies NCCL handles, process groups, or any
  other communicator. FlashInfer does not call `nccl*` or
  `torch.distributed.*` internally.
- **Attention.** After unpack, the caller invokes SageAttention2 (or any
  consumer of the same pre-quantized layout). The attention kernel is not
  part of this module.
- **Multi-node or cross-NIC paths.** The payload layout is designed for
  intra-node NVLink all-to-all. Cross-node payload compression or alternate
  quantization formats are out of scope.
- **Hopper (SM 9.0) native WGMMA kernel support.** The scale granularity
  (Q_GROUP=32 / K_GROUP=64) is matched to the `_qattn_sm89`
  grid used by SageAttention2 on both SM89 and SM120. A future layout variant
  targeting the SM90 WGMMA kernel (Q per-16-token, K per-128-token) would
  require new kernel instances and is not part of this PR.
- **Dynamic sequence lengths within a batch.** All batch entries share the
  same `S`; variable-length packing is not implemented.
