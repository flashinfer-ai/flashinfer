# Experimental MiniMax-H3 packed-varlen attention (Cake backend)

Both the APIs and their Cake backend are experimental and may change without
backward compatibility. Calling an API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no
automatic backend selection. Tracking: flashinfer-ai/flashinfer#4532
(MiniMax-H3 dense attention request; candidates 3A/3C), tracker #4254.

Two public entry points in `flashinfer.prefill` serve **noncausal
packed-varlen multi-head self-attention** over THD tensors on SM100 (B200) and
SM103 (B300/GB300):

| API | Program family | QK MMA | PV MMA |
| --- | --- | --- | --- |
| `minimax_h3_varlen_attention(q, k, v, cu_seqlens, *, softmax_scale=None, out=None)` | `bf16` | `kind::f16` (BF16 x BF16, FP32 accumulate) | `kind::f16` (BF16 P x BF16 V) |
| `minimax_h3_varlen_nvfp4_attention(..., pv_mode="fp8")` | `nvfp4_fp8pv` | `kind::mxf4nvf4` E2M1 x E2M1, UE4M3 block-16 scales | `kind::f8f6f4` E4M3 P (TMEM) x dense E4M3 V, one per-tensor scale `448 / amax(V)` |
| `minimax_h3_varlen_nvfp4_attention(..., pv_mode="fp4")` | `nvfp4_fp4pv` | `kind::mxf4nvf4` E2M1 x E2M1, UE4M3 block-16 scales | `kind::mxf4nvf4` E2M1 P (score-tile maximum as the shared anchor of the eight K16 scales) x E2M1 V^T with per-16-token UE4M3 scales |

Softmax statistics and accumulation are FP32 in every variant; the output is
rounded to BF16 once. Segments never attend across boundaries, empty and
sub-tile segments are allowed, and there is no causal mask, GQA sharing,
bias, dropout, sliding window or LSE output.

| Tensor | Shape | dtype |
| --- | --- | --- |
| `query`, `key`, `value` | `[T, H, 128]` packed THD, contiguous | bfloat16 |
| `cu_seqlens` | `[B + 1]`, `cu_seqlens[0] == 0`, non-decreasing, `cu_seqlens[B] == T` | int32 (CUDA) |
| `out` | `[T, H, 128]` (optional, caller-owned) | bfloat16 |

`T` and every segment length are arbitrary (no 64/128/512 alignment). `H` is
the number of local heads (56 / Ulysses degree for MiniMax-H3, but any
positive count works). `softmax_scale` defaults to `1 / sqrt(128)`.

```python
import torch
from flashinfer.prefill import minimax_h3_varlen_attention, minimax_h3_varlen_nvfp4_attention

cu = [0, 133, 300, 900]
T, H = cu[-1], 7
q = torch.randn(T, H, 128, dtype=torch.bfloat16, device="cuda")
k, v = torch.randn_like(q), torch.randn_like(q)
cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")

out = minimax_h3_varlen_attention(q, k, v, cu_seqlens)                     # BF16
out8 = minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode="fp8")  # NVFP4 QK, FP8 PV
out4 = minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode="fp4")  # NVFP4 QK, NVFP4 PV
```

Both one-shot APIs read `cu_seqlens` back to the host (one synchronization)
to build the segment plan; pass `cu_seqlens_host=[...]` to avoid it. For
repeated launches or CUDA Graph capture use the prepared form from this
package:

```python
from flashinfer.experimental.minimax_h3_varlen_attention.cake_backend import (
    prepare_minimax_h3_varlen_attention,
    prepare_minimax_h3_varlen_nvfp4_attention,
)

runner = prepare_minimax_h3_varlen_attention(q, k, v, cu_seqlens, out=out)
runner()            # launches on the current stream, returns out (no allocation)
nv = prepare_minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode="fp8", out=out8)
nv.quantize()       # quantizer stage only (fp8: amax reduction + one fused launch)
nv.attention()      # attention launch only
nv()                # complete pipeline (what the one-shot API times)
```

A runner is bound to one `cu_seqlens`, one shape set and one set of tensor
bindings; values may change freely between launches (the NVFP4 runner
re-quantizes on every launch). Prepare a new runner when segments, shapes or
bindings change.

## Host planning

### BF16 (`bf16`)

The kernel is one persistent launch of `num_clusters = min(num_SMs / 2,
total_tiles)` 2-CTA clusters (`2 * num_clusters` CTAs). A *unit* is one head x
one cluster tile, and a cluster tile covers **four consecutive 128-row Q tiles
(512 rows) of one segment**, so `total_tiles = num_heads * sum(ceil(seg_len /
512))`. The grid is statically strided: cluster `i` runs the units at slots
`i, i + num_clusters, i + 2 * num_clusters, ...`, and every role reads its
unit with one table lookup. The host builds, from `cu_seqlens` (empty segments
dropped) and `num_heads`:

* `seg_begin[s]`, `seg_len[s]` (int32, `num_segments` entries),
* `unit_table` (int32, `4 * total_tiles` entries): per slot the segment index,
  `head << 16 | cluster_in_segment`, `kv_block_begin << 16 | kv_blocks` (the
  unit's K/V block range inside the segment) and its partial slot (`-1` for an
  unsplit unit, which writes the BF16 output directly),
* `combine_table` (int32, `4 * num_combine_units` entries): per K/V-split
  unit the segment index, `head << 16 | cluster_in_segment`, its first partial
  slot and the number of splits, plus the plan's partial workspace
  `partial_O` (FP16, `num_partial_slots * 512 * 128`) and `partial_ML` (FP32,
  `num_partial_slots * 512 * 2`).

Units are enumerated segment-major (heads slow, clusters fast, so a cluster's
Q tiles reuse the segment's K/V from L2) and placed into slots
longest-processing-time first over the per-unit cost `ceil(seg_len / 128) + 2`
K/V blocks (`assign_unit_slots`): the partial tail round receives the cheapest
units and, within every full round, the clusters that also own a tail unit
receive that round's cheapest units; equal costs keep the enumeration order.

**K/V splits for the partial wave.** When the unit count leaves a partial
wave on the persistent grid (`total_units mod num_clusters != 0`, fewer than
four waves), the planner (`choose_kv_splits`) simulates the slot assignment
for splitting the `total_units mod num_clusters` most expensive units (and,
as the fallback, every unit) `k = 2..8` ways into near-equal K/V block ranges
and keeps the candidate with the lowest makespan plus combine cost when it
beats the unsplit plan by more than 3 %. A split unit's ranges are separate
units of the table; each runs the full online softmax over its range and
writes its rows normalized by its own softmax sum as FP16 into its partial
slot together with the FP32 `(scaled log2 row max, row sum)`. The `combine`
stage (one warp per output row, `128` CTAs of 128 threads per split unit)
then merges the slots with the exact FlashAttention formula
`O = sum_i 2^(m_i - m) l_i O_i / sum_i 2^(m_i - m) l_i` into the BF16 output.
It is launched after the attention kernel on the same stream and skipped when
the plan has no split units. Plans with at least as many units as clusters
per wave are unchanged (unsplit units are bitwise identical to the previous
kernel).
The plan is a host-side function of `(cu_seqlens, num_heads, num_SMs)` and
reproduces the Cake production plan table for table (`num_heads < 2^15`,
fewer than `2^16` clusters per segment). K/V TMA loads that run past a segment
are masked to `-inf` before the softmax; ragged Q tail tiles are staged with
predicated zero-filling copies;
output rows are stored straight to global memory predicated on the row lying
inside its segment.

### NVFP4 (`nvfp4_fp4pv`, `nvfp4_fp8pv`)

Quantization is part of the pipeline: one fused quantizer launch (`quantize`;
`minimax_h3_varlen_nvfp4_quantize_qkv` for `pv_mode="fp4"`,
`minimax_h3_varlen_nvfp4_quantize_qk_fp8v` for `pv_mode="fp8"`), then one
attention launch. The attention launch uses one of two generated programs,
chosen at preparation from the plan: the dense `attention` program (no
K/V-split code at all) for plans without split units, `attention_split` for
plans with them (`runner.attention_stage`, `route_metadata["attention_variant"]`).
Both are the same kernel specialised at build time; keeping the dense program
free of the split-unit epilogue keeps the softmax block loop of unsplit units
at the schedule of the single-program kernel (a split path in the epilogue
costs 1-2 % on every long row). The quantizer writes a
**head-major, per-segment 128-token-padded packed layout**: segment `s`
(non-empty segments only) owns `ceil(len / 128)` packed 128-token blocks
starting at packed block `seg_tile_base[s]`, `PB = sum(ceil(len / 128))`, and
token `t` of segment `s` lands at packed row `seg_tile_base[s] * 128 + (t -
seg_begin[s])` of its head. Padded rows inside a segment's region are
zero-filled with the minimum scale `2^-9`. Because the packed buffers look
exactly like the dense NVFP4 kernel's `[head * S_pad, .]` buffers with
`S_pad = PB * 128`, every TMA map and scale-factor address formula of the dense
kernel is unchanged. Host tables (int32):

* quantizer (built once per `cu_seqlens`), `PB` entries: `block_token[p]`
  (first THD token of packed block `p`), `block_valid[p]` (valid rows,
  `0..128`). Grid `heads * PB * 4` CTAs of 256 threads: CTA `b` quantizes Q,
  K and V of the 32-token slice `b % 4` of packed block `(b // 4) % PB` for
  head `(b // 4) // PB`.
* attention scheduler (built once per `(cu_seqlens, heads)`,
  `build_tile_tables`), one entry per scheduled unit (`heads * sum(ceil(len /
  512))` cluster tiles plus the extra K/V-split ranges): `cl_head[t]`,
  `cl_seg_begin[t]`, `cl_seg_len[t]`, `cl_kv_base[t] = seg_tile_base[s]`,
  `cl_q_block[t]` (first Q block of the cluster tile inside its segment, a
  multiple of four), `cl_kv_begin[t]` / `cl_kv_blocks[t]` (the unit's K/V
  block range inside the segment) and `cl_ws_slot[t]` (partial slot, `-1` for
  an unsplit unit). Units are enumerated segment-major with the longest
  segments first (ties keep segment order), then head, then the segment's
  cluster tiles (consecutive units stream one head's K/V), split by the same
  `choose_kv_splits` planner as the BF16 family, and placed into the
  statically strided persistent grid (`grid = 2 * min(num_SMs / 2,
  total_tiles)` CTAs, cluster `i` runs units `i, i + G, ...`)
  longest-processing-time first (`assign_unit_slots`). CTA rank `r` of a
  cluster owns blocks `cl_q_block + 2r` and `cl_q_block + 2r + 1`. The tables
  also carry the `combine_table`, `seg_begin` / `seg_len` and the FP16 / FP32
  partial workspace of the `combine` stage, which runs after the attention
  launch (ordinary serial launch) when the plan has split units.

The attention launch is a **programmatic dependent launch**: the quantizer
signals `griddepcontrol.launch_dependents` at its end, the attention prologue
runs its barrier/TMEM setup and then waits with `griddepcontrol.wait` before
reading the packed operands. The launch attribute
(`cudaLaunchAttributeProgrammaticStreamSerialization`) is baked into both
generated attention host bindings by the export, so the runner only enqueues
the launches in order on the current stream (quantizer, the bound attention
program and, for plans with K/V-split units, the `combine` stage); the `fp8`
`amax(V)` reduction runs before the quantizer launch.

Packed operands (`q_fp4`, `k_fp4`, `q_scale`, `k_scale`, plus `v_fp4_t`,
`v_scale_lo`, `v_scale_hi` for `fp4` or `v_fp8`, `v_amax` for `fp8`) cost
about 180 bytes per (token, head) and are allocated at preparation
(`nvfp4_workspace_shapes(heads, PB, pv_mode)`); a caller-owned set may be
passed through `workspace=`. For `pv_mode="fp8"` the per-tensor
`amax(V)` is one device reduction into `v_amax` before the fused quantizer
launch; the scalar never visits the host, so the pipeline is CUDA-Graph
capturable.

## Supported hardware and limitations

* SM100 (`sm_100a`, B200) and SM103 (`sm_103a`, B300/GB300) only. Separate
  exact-arch programs are registered per architecture (`ROUTES["<variant>__<arch>"]`,
  selected from the device's compute capability): the BF16 program is a
  different trace of the same schedule per architecture (on SM103 the score
  drain uses `tcgen05.ld.red`, every exp2 runs on MUFU and the lazy rescale
  threshold is `2^-8`; on SM100 a quarter of the exp2 work runs as a packed
  FMA polynomial). The NVFP4 attention programs differ per architecture the
  same way (`USE_TMEM_LD_RED`: SM103 drains scores with `tcgen05.ld.red` and
  runs every exp2 on MUFU; SM100 emulates the last quarter of the softmax
  exp2 pairs with a packed polynomial). SM120/SM121 lack the tcgen05/TMEM
  path and are not supported; SM90 is not a target.
* BF16 THD inputs and BF16 output only, head dimension 128, noncausal,
  self-attention (`H_q == H_kv`), no bias/mask/window/LSE/dropout.
* The NVFP4 variants trade accuracy for speed: validated tolerance against an
  FP32 reference is `atol=1.0, rtol=0.1` (BF16: `atol=rtol=1e-2`). The
  `fp8` PV mode is the default because its error is lower at similar speed;
  `fp4` PV is the fastest mode for long segments.
* One-shot APIs synchronize once to read `cu_seqlens` unless
  `cu_seqlens_host` is passed. The prepared runners never allocate or
  synchronize.
* Generated sources live under `csrc/cake_minimax_h3_varlen_attention/<arch>/`
  and are registered in `cake_jit.py` (`MODULES`, `ROUTES`) by the Cake
  generated-program export; the JIT builds them on first use. Until the export
  has run, the entry points raise `NotImplementedError` naming the missing
  program.

See `tests/experimental/test_cake_minimax_h3_varlen_attention.py` for the FP32
reference and validated shape set (the contract rows of the Cake evaluation:
production-center single segments of 4k-110k tokens per Ulysses degree, ragged
tails, trailing padding, multi-segment packs and sub-tile smoke rows) and
`benchmarks/bench_cake_minimax_h3_varlen_attention.py` for the CUPTI benchmark
that reports complete-call, quantization-only and attention-only times per
row, optionally against `BatchPrefillWithRaggedKVCacheWrapper` on the same
tensors.
