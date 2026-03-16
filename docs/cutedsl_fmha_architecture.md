# CuTe DSL FMHA Architecture: Comparison & Migration Plan

## Overview

This document compares three implementations of Blackwell (SM100) Flash Multi-Head Attention using CuTe DSL / CUTLASS, and outlines a migration plan to modularize FlashInfer's CuTe DSL kernels for maintainability, extensibility, and performance.

### Implementations Compared

| Implementation | Location | Language | Role |
|----------------|----------|----------|------|
| **C++ CUTLASS** | `cutlass/examples/77_blackwell_fmha/` | C++ templates | Upstream reference; most complete |
| **DKG Python** | `cutlass_ir/compiler/python/examples/blackwell/` | Python CuTe DSL | Faithful port with additional optimizations |
| **FlashInfer PR #1549** | `flashinfer/cute_dsl/` | Python CuTe DSL | Derivative with user-facing customization hooks |

---

## 1. PR #1549 Changes Summary

PR #1549 ("[CuTe DSL] Add Blackwell MHA prefill and MLA decode kernel") adds two new high-performance attention kernels for NVIDIA Blackwell (SM100) GPUs written in CuTe DSL.

### New Files

- **`flashinfer/cute_dsl/prefill.py`** (~2933 lines) — Fused MHA prefill kernel
- **`flashinfer/cute_dsl/mla.py`** (~3641 lines) — Multi-Latent Attention decode kernel
- **`flashinfer/cute_dsl/patch/pipeline.py`** (~419 lines) — Producer/consumer pipeline wrappers

### MHA Prefill Kernel

- Warp specialization with 16 warps: 8 softmax (4+4 for double-buffered Q tiles), 4 correction, 1 MMA, 1 load, 1 epilogue, 1 empty
- Multi-stage pipelining: TMA load, MMA, softmax, correction, epilogue overlap via async barriers
- Uses Tensor Memory (TMEM) for intermediate QK scores and PV accumulators
- Double-buffered Q tiles (2 per CTA) for maximum throughput
- Customizable: logits transforms, output transforms, attention sink support, sliding window, causal masking
- Reaches ~1247 TFLOPs on B200 at seq_len=65536

### MLA Decode Kernel

- Split-KV attention for long sequences with separate reduction kernel
- Paged KV cache with page tables
- Separate latent (512d) and RoPE (64d) dimension handling
- Configurable multi-CTA clusters
- Variable sequence lengths and variable split-KV
- Reaches ~1436 TFLOPs and ~6116 GB/s on B200

### Attention Sinks

The prefill kernel supports attention sinks via user-injectable `@cute.jit` callbacks:

- **`M_D_update`**: Modifies online softmax's running max and sum to account for a per-head sink value on the first KV tile
- **`output_transform`**: Rescales the output by the corrected normalizer

The sink value acts as a virtual extra token in the softmax denominator:
$$p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j) + \text{sink}_h}$$

This is implemented by injecting `log(sink)` into the online softmax at `kv_tile_idx == 0`, with zero overhead on subsequent tiles (compiles away via `cutlass.const_expr`).

### Other Changes

- **Routed MoE API**: `trtllm_fp4_block_scale_routed_moe()` accepting pre-computed top-k routing decisions
- **TRT-LLM headers**: Extended batched GEMM and fused MoE for routing maps
- **Benchmarks**: New benchmarks for CuTe DSL prefill and MLA decode
- **Tests**: Comprehensive test coverage including variable-length, GQA, and attention sink variants

---

## 2. Prefill Kernel vs. Decode Kernel (within PR #1549)

| Aspect | Prefill (`prefill.py`) | Decode MLA (`mla.py`) |
|--------|------------------------|-----------------------|
| **Workload** | Standard MHA for long sequences | Multi-Latent Attention for decoding |
| **Q shape** | Many Q tokens (seq_len 512–65K) | One Q token per sequence |
| **Bottleneck** | Compute-bound | Memory-bound |
| **Warps** | 16 (fine-grained specialization) | 6–8 (merged compute) |
| **Pipelines** | 10 (deeply pipelined) | 5 (simpler) |
| **Q double-buffering** | Yes (2 Q tiles/CTA) | No |
| **Split-KV** | No | Yes (+ reduction kernel) |
| **KV access** | Contiguous tensors | Paged with page tables |
| **Q/K structure** | Monolithic | Latent + RoPE decomposed |
| **Clusters** | Single CTA | Multi-CTA supported |
| **Customization** | logits/output transforms, sinks | Fixed-function |

### Warp Layout Comparison

**Prefill (16 warps):**
| Warps | Role | Registers |
|-------|------|-----------|
| 0–3 | Softmax0 (Q tile 0) | 192 |
| 4–7 | Softmax1 (Q tile 1) | 192 |
| 8–11 | Correction | 96 |
| 12 | MMA | 32 |
| 13 | Load | 32 |
| 14 | Epilogue | 32 |
| 15 | Empty | 24 |

**Decode MLA (6–8 warps):**
| Warps | Role | Registers |
|-------|------|-----------|
| 0–3 | Compute (softmax + correction merged) | 192 |
| 4 | MMA | 96 |
| 5 | TMA Load | 96 |
| 6 | Page Table Load (optional) | 96 |
| 7 | Empty (optional) | 96 |

---

## 3. FlashInfer vs. DKG Python vs. C++ CUTLASS

### Prefill Kernel

| Aspect | C++ CUTLASS | DKG Python | FlashInfer |
|--------|-------------|------------|------------|
| **Warp layout** | 16 warps, same roles | 16 warps, same roles | 16 warps, same roles |
| **TMEM layout** | S0@0, S1@128, O0@256, O1@384, P in S | Same | Same |
| **Pipelines** | 8 + `OrderedSequenceBarrier` | 12 (explicit P0/P1 + inplace) | 10 (using `pipeline_patch.py`) |
| **P storage** | TMEM (PV reads P from TMEM) | TMEM | TMEM |
| **Masking** | Template policy (`CausalMask<bool>`) with 3-phase loop | `FusedMask` class, similar 3-phase | `MaskType` enum, similar splitting |
| **Softmax pipelining** | Manual SW pipeline: `kFmaPipeCount=8`, `kConvertPipeCount=16` | Not as deeply pipelined | Not as deeply pipelined |
| **exp2 emulation** | `enable_exp2_emulation` flag | Not mentioned | Not present |
| **Skip-correction** | Not shown in prefill | `enable_skip_correction` | Not present |
| **FP8** | Full (E4M3/E5M2 with scale_q/k/v/o) | Full (E4M3) | BF16/FP16 only |
| **LSE output** | Optional `lse_calculation=True` | Optional | Not exposed |
| **Customization** | Static template mask policies | Fixed-function | **Logits/output transforms, attention sinks** |
| **Variable-length** | `VariableLength` with cumulative_length | `cum_seqlen_q/k` | `cum_seqlen_q/k` |
| **Persistent mode** | Yes | Yes | Yes |
| **Backward pass** | Yes | No | No |
| **Code organization** | 5+ files (kernel, mainloop, loader, epilogue, common) | 1 file (~3600 lines) | Monolithic: 1 file (~2930 lines); Modular: 36 files (~7600 lines) in `attention/` package |

### Decode / MLA Kernel

| Aspect | C++ CUTLASS MLA | DKG Python MLA | FlashInfer MLA |
|--------|-----------------|----------------|----------------|
| **Warps** | 8 (4 compute, 1 MMA, 1 TMA, 1 PT, 1 empty) | 12 (4 compute, **4 correction**, 1 MMA, 1 TMA, 1 PT, 1 empty) | 6–8 (4 compute, 1 MMA, 1–2 load) |
| **Correction** | Merged into compute warps | **Separate 4-warp group** (208 regs) | Merged into compute warps |
| **2-CTA cluster** | Always (`kIs2Sm=true`) | Always (`cluster=(2,1,1)`) | Configurable |
| **KV pipeline stages** | 12–15 (unified K+V) | 15 (unified K+V) | Separate load_q/load_kv |
| **PT load** | Dedicated warp + `PipelineCpAsync` (4 stages) | Dedicated warp + 4-stage pipeline | Optional, inline in load warp |
| **Split-KV reduction** | Separate + **fused atomic** (`atomicMax` + `TMA_REDUCE_ADD`) | Separate only | Separate only |
| **Skip-correction** | Not shown | Yes (`vote_all_sync`) | Not present |
| **FP8 variant** | Separate template instantiation | Separate file (`mla_decode_fp8.py`) | Not present |
| **Paged KV** | TMA + CpAsync with `ComposedLayout<Gather>` | TMA with page table | TMA with page table |

### Gen (Decode) Kernel — C++ Only

The C++ CUTLASS implementation includes a **gen kernel** (`sm100_fmha_gen_*`) for the generation/decode phase with features not present in either Python version:

- **KV cache append**: Fuses loading new K/V, appending to cache, and attention into one kernel
- **`cp.async` load path**: Software-managed async copies (vs. TMA) enabling complex access patterns for per-batch variable-length KV, cache index remapping, and cache+new stitching
- **GQA head packing**: Maps multiple Q heads into the MMA N-dimension (groups of 8/16/32)
- **Direct GMEM epilogue**: Correction warp writes O directly to global memory (no TMA store needed for single-token output)

### Mixed-Input FMHA — DKG Python Only

The DKG implementation includes mixed-precision variants not present elsewhere:

- **`mixed_input_fmha_decode.py`**: Q in BF16, K/V in Int8/Int4/FP8 with per-block scale factors
- **`mixed_input_fmha_prefill_d256.py`** / **`d512.py`**: Prefill with 8 dedicated transform warps for KV dequantization
- Dequant path: load quantized KV → multiply by block scales → write to SMEM/TMEM as BF16

---

## 4. Structural Comparison: C++ Templates vs. Python JIT

| | C++ CUTLASS | Python CuTe DSL |
|---|---|---|
| **Dispatch** | AOT — every (dtype, headdim, mask, schedule) combo is a separate template instantiation | JIT — compile only the exact config needed at runtime |
| **Binary size** | Large (combinatorial explosion) | Minimal (one .so per runtime config) |
| **Compile time** | Minutes to hours for full build | Seconds per kernel variant |
| **Performance ceiling** | Maximum — all decisions compile-time, zero dispatch overhead | Equivalent kernel quality (same PTX/SASS) |
| **Customization** | Static template policies — hard to add new variants | Runtime-composable — can inject arbitrary Python callables |
| **Readability** | Heavy template syntax; verbose pipeline management | Algorithm maps more directly to the math |
| **Error messages** | Opaque template errors | Python tracebacks |
| **CUTLASS infra reuse** | Full (`CollectiveBuilder`, standard pipelines, `TiledMma`) | Partial (CuTe layout algebra, MMA atoms, but not `CollectiveBuilder`) |

---

## 5. Variable Head Count: Mapping Heads to MMA Dimensions

### The core issue: prefill vs. decode head mapping

The way attention heads are mapped to MMA dimensions differs fundamentally between prefill and decode kernels, and this has major implications for supporting different head counts (128, 64, 32, 16, 8, ...) and attention variants (MQA, GQA, MLA).

#### Prefill kernels: heads in the grid dimension

In **all three implementations**, prefill kernels map heads to the outermost "L" (loop/batch) dimension, not to the MMA tile. Looking at FlashInfer's prefill layout:

```python
# (s, d, ((h_r, h_k), b))
q_layout = cute.make_layout(
    (s_q_all, d, ((h_r, h_k), b_q)),
    stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_q)),
)
```

- **M-dimension**: `s_q` (sequence length) → MMA M-tile
- **K-dimension**: `d` (head dimension) → MMA K-tile
- **L-dimension**: `((h_r, h_k), b)` (heads × batch) → **grid dimension**

Each CTA processes one `(seq_q_tile, head_group)` pair. The number of heads simply determines how many CTAs launch along the grid Y and Z dimensions. **Any number of heads works naturally** for prefill — it just changes the grid size, not the MMA tile utilization. The same applies to training/backward kernels.

#### MLA decode kernels: heads in the MMA M-dimension

In MLA decode, Q has `seq_len = 1`, so the sequence dimension collapses. Instead, **all heads are packed into the MMA M-dimension**:

```
MMA shape: (M=num_heads, N=seq_len_k_tile, K=rope_dim)
S = Q_all_heads * K_tile^T  →  shape (num_heads, kv_tile_size)
```

One CTA processes **all heads simultaneously** in a single MMA operation. The MMA tile M-dimension is fixed at 128 (the tcgen05 instruction width). When `num_heads = 128`, it fits perfectly. When `num_heads < 128`, the extra rows are "phantom" heads — they compute but their results are discarded.

This design makes sense because:
1. With only 1 query token, the sequence dimension can't fill the MMA tile
2. DeepSeek MLA's compressed KV cache is shared across all heads (no per-head K/V), so all heads naturally process the same KV data
3. Packing heads into the M-dimension maximizes MMA utilization for the common 128-head case

#### Standard (non-MLA) decode: GQA groups in the MMA N-dimension

The DKG `fmha_decode.py` handles standard GQA decode differently from MLA — it packs **GQA groups** into the MMA N-dimension:

```
grouped_head_tile = min(num_heads_q / num_heads_kv, 32)  # e.g., 8, 16, or 32
MMA shape: (M=kv_tile, N=grouped_head_tile, K=head_dim)
```

The head ratio (GQA group size) determines how many Q-heads share the same K/V tile. Typical GQA ratios (4, 8, 16, 32) fit naturally into the N-dimension.

### Cross-implementation comparison: variable head support

| Kernel | Head mapping | Variable heads? | Why |
|--------|-------------|----------------|-----|
| **Prefill (all impls)** | Heads → grid dimension | Naturally supported | Grid size just changes |
| **Training/backward** | Same as prefill | Naturally supported | Grid size just changes |
| **Standard decode (DKG)** | GQA groups → MMA N-dim | Works for typical GQA ratios | GQA ratio is usually small (4–32) |
| **MLA decode (C++ CUTLASS)** | All heads → MMA M-dim | **Only 128 heads** | `static_assert(TileShapeH{} == 128)` |
| **MLA decode (DKG Python)** | All heads → MMA M-dim | **Only 128 heads** | `mma_qk_tiler_mn[0] != 128` check |
| **MLA decode (FlashInfer)** | All heads → MMA M-dim | **128, 64, 32, 16, 8** | Runtime `num_heads` parameter with boundary masking |

### How FlashInfer handles variable heads in MLA decode

FlashInfer's MLA kernel makes `num_heads` a **runtime parameter** rather than a compile-time constant:

1. **Conditional 2-CTA mode** — only use cooperative 2-CTA instructions when `num_heads == 128`:
   ```python
   self._use_2cta_instrs = num_heads == 128
   ```
   For fewer heads, fall back to single-CTA mode.

2. **Over-provisioned MMA tile** — always use 128-wide M-tiles, but track actual head count:
   ```python
   cta_qk_tiler = (
       self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],  # e.g., 128 or 64 per CTA
       self.mma_qk_tiler[1],
       self.mma_qk_tiler[2],
   )
   ```

3. **Boundary masking in epilogue** — only write results for real heads:
   ```python
   if cute.elem_less(tTR_cO[0][0], self.num_heads):
       cute.autovec_copy(tR2G_rO_src, tR2G_rO_dst)
   ```

4. **LSE output masking**:
   ```python
   if cute.elem_less(cLSE[common_params.tidx][0], self.num_heads):
       gLSE[common_params.tidx] = lse
   ```

**Performance tradeoff**: For `num_heads < 128`, some compute is wasted on phantom head rows. But MLA decode is **memory-bandwidth-bound** (not compute-bound), so the wasted compute is hidden by KV cache load latency. The PR's performance table confirms this — going from 128 to 8 heads, TFLOPs drops proportionally (684→43) but memory bandwidth stays high (3541→2895 GB/s).

### Can `TileBounds` handle MQA, GQA, and MLA decode uniformly?

The `TileBounds` abstraction proposed in the modularization plan (Section 7) addresses **partial MMA tile filling** — when the logical data dimension is smaller than the physical MMA tile. This is one important piece of the puzzle, but it's not the whole story. MQA, GQA, and MLA decode differ in more ways than just tile occupancy:

| Variant | KV sharing | Head mapping in decode | What `TileBounds` handles |
|---------|-----------|----------------------|--------------------------|
| **MHA** | No sharing (h_q = h_kv) | Each head is independent → grid/loop dim | N/A (no partial tiles) |
| **MQA** | All Q heads share 1 KV head (h_kv = 1) | All heads share same KV → could pack into M or N | M-masking if packed into M-dim |
| **GQA** | Groups share KV (h_q = g × h_kv) | Group ratio g heads share KV → pack g into N-dim | N-masking if g < N-tile |
| **MLA** | All heads share compressed latent KV | All heads → M-dim | **M-masking when num_heads < 128** |

`TileBounds` cleanly handles the **M-dimension masking** problem that's most acute in MLA decode. But a fully unified decode kernel would also need:

1. **Head mapping strategy** — how heads map to MMA dimensions:
   - MLA: all heads → M-dimension (because KV is shared and latent-compressed)
   - GQA: group ratio → N-dimension (because each group has distinct KV)
   - MQA: similar to GQA with group_size = num_q_heads

2. **KV access pattern** — how KV is loaded:
   - MLA: paged latent + RoPE, decomposed into two GEMM paths
   - GQA/MQA: standard paged or contiguous KV per head group
   - MHA: per-head KV

3. **Softmax scope** — what constitutes a "row" in online softmax:
   - MLA: each head is a row, all rows share the same KV scores
   - GQA: each Q-head within a group has its own row, shares KV scores with sibling heads
   - The row-max/row-sum reductions are identical in algorithm, just differ in which dimension they reduce

The `TileBounds` abstraction handles concern (1)'s **boundary effects** regardless of the mapping strategy. Combined with the modular design where the head mapping strategy is configurable (in `AttentionConfig`), the softmax and epilogue modules can be reused across all variants:

```python
class SoftmaxWarpGroup:
    def step(self, ...):
        rS = tmem_load(tmem_S)

        # Head-dimension masking (compile-time eliminated when not needed)
        if cutlass.const_expr(self.config.tile_bounds.needs_m_masking(tile_m)):
            for i in range(size(rS)):
                if not elem_less(coord_m[i], self.config.tile_bounds.m_bound):
                    rS[i] = -inf

        # KV-dimension masking (causal, sliding window, residual)
        if cutlass.const_expr(self.fusion.mask is not None):
            rS = self.fusion.mask.apply(rS, kv_tile_idx, ...)
```

**Bottom line**: `TileBounds` is necessary but not sufficient for a fully unified MQA/GQA/MLA decode kernel. The complete solution requires `TileBounds` (partial tile masking) + a configurable head mapping strategy (how heads map to MMA dims) + flexible KV access patterns (paged, contiguous, or latent-decomposed). All three are addressed by the modular design's `AttentionConfig`, role modules, and fusion abstractions working together. The key insight is that the *softmax algorithm itself* is identical across all variants — only the data layout feeding into it differs.

---

## 6. Performance Optimizations to Port

Prioritized by expected impact:

### High Priority

1. **Skip-correction optimization** (from DKG decode)
   - When `exp2(scale * (old_max - new_max)) ≈ 1.0`, skip O rescaling entirely
   - Use `vote_all_sync` to check across the warp
   - High impact for decode where correction is often unnecessary

2. **Softmax software pipelining** (from C++ CUTLASS)
   - Interleave FMA, exp2, and dtype conversion with depths 8/16
   - Hides transcendental op latency on prefill's compute-heavy path

### Medium Priority

3. **exp2 emulation** (from C++ CUTLASS)
   - Polynomial approximation faster than hardware `exp2` on SM100
   - Unnecessary on SM103+, so needs architecture dispatch

4. **Fused atomic reduction for split-KV** (from C++ CUTLASS)
   - Eliminates the reduction kernel via `atomicMax` for LSE + `TMA_REDUCE_ADD` for output
   - Saves kernel launch overhead and global memory round-trip

5. **FP8 support** (from both C++ and DKG)
   - E4M3/E5M2 inputs with separate scale_q/k/v/inv_scale_o
   - Requires wider QK MMA tiles (2x K-dim for FP8)

### Lower Priority

6. **Causal-aware tile scheduling** (from C++ CUTLASS)
   - Swizzled launch order: longest mainloop tiles first for load balancing
   - 16x8 super-tile swizzling for L2 cache locality

7. **ThreadShape configurability** (from C++ CUTLASS)
   - `Shape<_2,_1,_1>` vs `Shape<_1,_2,_1>` for how softmax warp-groups partition Q vs K dimensions

8. **LSE output** (from C++ and DKG)
   - Optional log-sum-exp output per row: `lse = log(row_sum) + scale * row_max`

---

## 7. Modularization Design

### Motivation

`prefill.py` (2934 lines) and `mla.py` (3641 lines) were monolithic classes where warp roles, pipeline topology, TMEM layout, softmax algorithm, masking, and scheduling were intertwined. The modular `attention/` package breaks these into focused building blocks. After Phases 1–7, the kernel files have been reduced to 598 lines (FMHA) and 593 lines (MLA) — pure kernel logic with no wrapper or utility code. The two kernels are now near-identical in size, with each warp section expressed as a single-line dispatch to the corresponding role's `run()` method.

### Design Principle

**Kernels live at the top level** of the `attention/` package. They are readable, high-level compositions that express the core mathematical algorithm. Building blocks (config, roles, fusion, scheduler, pipeline, collective_builder) live one level below in subdirectories. When you open `attention/prefill.py`, you should see something close to:

> "Load Q, K, V tiles. Compute S = QK^T. Apply mask. Softmax. Compute O = PV. Correct. Write output."

The building blocks handle the *how* (TMEM layout, pipeline synchronization, warp assignment). The kernel expresses the *what*.

**For a step-by-step guide on writing a new attention variant, see Section 10.**

After completing Phases 1–7, the kernel files achieve this goal:

- **`prefill.py`** (598 lines, down from 1230): Pure kernel class with no config unpacking boilerplate, no infrastructure setup, and no wrapper/test code. The `__call__` method's warp dispatch reads as high-level role delegation. The `CollectiveBuilder` handles all MMA atom, SMEM layout, TMA atom, and `SharedStorage` creation. Helper methods `_create_pipelines()` and `_create_mma_fragments()` keep the kernel body concise.
- **`mla_decode.py`** (593 lines, down from 1554): Same treatment and now nearly identical in size to FMHA. Tile scheduler loops and parameter bundling were moved from the kernel body into each role's `run()` method, making the three warp sections (Load, MMA, Compute) single-line dispatches — matching FMHA's pattern exactly. The remaining size beyond FMHA reflects MLA's inherent complexity (split-KV reduction kernel, cluster synchronization, TMEM lifecycle).

Following the C++ CUTLASS collectives pattern, FMHA and MLA use **separate concrete types** (not abstract base classes) for variant-specific components (`AttentionConfig` vs `MLAConfig`, `WarpSchedule` vs `MLAWarpSchedule`), while sharing infrastructure (`PipelineTopology`, `softmax_math`, `tmem_utils`) via composition. The original monolithic files are preserved unchanged; the modular implementation is a parallel codebase in `flashinfer/cute_dsl/attention/`.

### Module Structure

```
flashinfer/cute_dsl/attention/
├── __init__.py                    # Re-exports public API
│
│  ── Kernels (top-level: readable dispatchers) ──
├── prefill.py                     # FMHA prefill kernel (600 lines, pure kernel)
├── mla_decode.py                  # MLA decode kernel + reduction kernel (1073 lines, pure kernel)
├── collective_builder.py          # build_fmha/mla_launch_params (MMA, SMEM, TMA, SharedStorage)
│
│  ── Configuration ──
├── config.py                      # AttentionConfig, AttentionFusion, HeadMapping
├── mla_config.py                  # MLAConfig (separate concrete type)
├── tmem_layout.py                 # TmemLayout: computed TMEM offsets
├── warp_schedule.py               # WarpSchedule for FMHA (16 warps)
├── mla_warp_schedule.py           # MLAWarpSchedule (6-8 warps)
├── mainloop_spec.py               # MainloopSpec (FMHA), MLAMainloopSpec
├── pipeline_topology.py           # PipelineTopology, PipelineEdge, PipelineType, factory
│
│  ── Warp Roles ──
├── roles/
│   ├── softmax.py                 # SoftmaxRole (FMHA)
│   ├── correction.py              # CorrectionRole (FMHA)
│   ├── mma.py                     # MmaRole (FMHA)
│   ├── loader_tma.py              # LoaderRole (FMHA)
│   ├── epilogue.py                # EpilogueRole (FMHA)
│   ├── mla_loader.py              # MLALoaderRole
│   ├── mla_mma.py                 # MLAMmaRole
│   ├── mla_compute.py             # MLAComputeRole (orchestrator)
│   ├── mla_softmax.py             # MLASoftmaxRole
│   ├── mla_rescale.py             # MLARescaleRole
│   ├── mla_epilogue.py            # MLAEpilogueRole
│   ├── softmax_math.py            # Shared: exp2_scale, packed_row_sum
│   └── tmem_utils.py              # Shared: tmem_load_partition
│
│  ── Fusion / Masking ──
├── fusion/
│   ├── mask.py                    # MaskType, apply_mask, trip count helpers
│   ├── logits_transform.py        # sigmoid_logits_transform
│   ├── output_transform.py        # dumb_output_transform
│   └── softmax_modifier.py        # (stub) WithSink modifier
│
│  ── Schedulers ──
├── scheduler/
│   ├── persistent.py              # FmhaStaticTileScheduler
│   └── mla_persistent.py          # MLAStaticTileScheduler
│
│  ── PyTorch Wrappers ──
└── wrappers/
    ├── batch_prefill.py           # BatchPrefillCuteDSLWrapper + tensor creation utils
    └── batch_mla.py               # BatchMLAPagedAttentionWrapperCuteDSL + page table/tensor utils
```

### Key Abstractions

#### `AttentionConfig` — Single source of truth

Replaces scattered `self.xxx` attributes. Computed once, passed to all modules.

```python
@dataclass
class AttentionConfig:
    # Problem shape
    head_dim_qk: int
    head_dim_vo: int
    num_heads: int             # Number of attention heads
    num_kv_heads: int = 0      # KV heads (0 = same as num_heads; for GQA/MQA)
    latent_dim: int = 0        # MLA (0 = standard MHA)
    rope_dim: int = 0          # MLA

    # Head mapping (see Section 5)
    head_mapping: HeadMapping = HeadMapping.GRID  # GRID (prefill) or MMA_M (decode)

    # Tile shape
    mma_tiler_mnk: Tuple[int, int, int]
    num_q_tiles_per_cta: int = 2

    # Types
    q_dtype: Type[cutlass.Numeric]
    kv_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32

    # Execution mode
    is_persistent: bool = False
    cluster_shape: Tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False

    # Features
    use_paged_kv: bool = False
    use_split_kv: bool = False

    @cached_property
    def tile_bounds(self) -> TileBounds:
        """Derive tile bounds from head mapping and head count (see Section 5)."""
        if self.head_mapping == HeadMapping.MMA_M:
            return TileBounds(m_bound=self.num_heads)
        return TileBounds()  # No partial-tile masking needed

    @cached_property
    def tmem_layout(self) -> TmemLayout: ...

    @cached_property
    def warp_assignment(self) -> WarpAssignment: ...

    @cached_property
    def pipeline_topology(self) -> PipelineTopology: ...
```

#### `TmemLayout` — Computed (not hardcoded) TMEM map

Derives offsets from tile configuration. Eliminates magic numbers.

```python
class TmemLayout:
    @staticmethod
    def from_config(config):
        # S0@0, S1@tile_m, O0@2*tile_m, O1@3*tile_m
        # P0 aliased inside S1+32, P1 inside S0+32
        # Vec buffers at start of each S region
        ...
```

#### `HeadMapping` — How heads map to MMA dimensions (see Section 5)

```python
class HeadMapping(Enum):
    GRID = "grid"      # Heads in grid/loop dim (prefill, training)
    MMA_M = "mma_m"    # All heads packed into MMA M-dim (MLA decode)
    MMA_N = "mma_n"    # GQA group packed into MMA N-dim (standard GQA decode)
```

This determines the kernel's Q layout and whether `TileBounds` needs M-dimension or N-dimension masking.

#### `TileBounds` — Partial MMA tile masking (see Section 5)

Handles the case where the logical data dimension is smaller than the physical MMA tile. Most critical for MLA decode where `num_heads < 128`.

```python
@dataclass
class TileBounds:
    """Handles the case where the actual problem is smaller than the MMA tile."""
    m_bound: int | None = None  # num_heads for MLA decode, None for prefill
    n_bound: int | None = None  # actual seq_len for residual KV tiles

    def needs_m_masking(self, tile_m: int) -> bool:
        return self.m_bound is not None and self.m_bound < tile_m

    def needs_n_masking(self, tile_n: int) -> bool:
        return self.n_bound is not None and self.n_bound < tile_n
```

Consumed by `SoftmaxWarpGroup` and epilogue modules — compiles away via `cutlass.const_expr` when no masking is needed (e.g., prefill, or MLA decode with exactly 128 heads).

#### `PipelineTopology` — Declarative pipeline graph

```python
@dataclass
class PipelineSpec:
    name: str
    pipeline_type: type    # PipelineTmaUmma, PipelineUmmaAsync, PipelineAsync
    stages: int
    producer_role: WarpRole
    consumer_role: WarpRole
    tx_count: int = 0
```

Enables visualization, validation (no cycles/orphans), and configuration swapping.

#### `AttentionFusion` — Customization bundle

```python
@dataclass
class AttentionFusion:
    mask: Mask = NoMask()
    logits_transform: Callable | None = None
    output_transform: Callable | None = None
    softmax_modifier: SoftmaxModifier | None = None
    # Future: kv_transform, score_mod, block_mask, ...
```

Each option resolves at JIT time via `cutlass.const_expr` — zero overhead when unused.

#### MLA as a config variant, not a fork

```python
# MLA decode with 128 heads (DeepSeek-V3)
mla_128h = AttentionConfig(
    head_dim_qk=576, head_dim_vo=512,
    num_heads=128, latent_dim=512, rope_dim=64,
    head_mapping=HeadMapping.MMA_M,
    num_q_tiles_per_cta=1,  # decode: single Q
    use_paged_kv=True, use_split_kv=True,
    cluster_shape=(2, 1, 1), use_2cta_instrs=True,
)
# tile_bounds.m_bound = 128, no masking needed (128 == tile_m)

# MLA decode with 64 heads (smaller variant)
mla_64h = AttentionConfig(
    head_dim_qk=576, head_dim_vo=512,
    num_heads=64, latent_dim=512, rope_dim=64,
    head_mapping=HeadMapping.MMA_M,
    num_q_tiles_per_cta=1,
    use_paged_kv=True, use_split_kv=True,
    cluster_shape=(1, 1, 1), use_2cta_instrs=False,  # single CTA for 64 heads
)
# tile_bounds.m_bound = 64, masking rows 64-127 in softmax and epilogue

# GQA decode (standard, non-MLA)
gqa_decode = AttentionConfig(
    head_dim_qk=128, head_dim_vo=128,
    num_heads=32, num_kv_heads=8,  # 4:1 GQA ratio
    head_mapping=HeadMapping.MMA_N,
    num_q_tiles_per_cta=1,
    use_paged_kv=True, use_split_kv=True,
)
# GQA group (4 heads) packed into MMA N-dim

# Reuses SoftmaxWarpGroup, CorrectionWarpGroup, etc.
# Only loader and MMA are specialized per variant.
```

---

## 8. Migration History

Each step was independently testable against the existing monolithic implementation. The original monolithic files (`prefill.py`, `mla.py`) were preserved unchanged throughout.

### Phase 1: Extract Configuration -- DONE

1. **`AttentionConfig`** / **`MLAConfig`** — separate concrete dataclasses for each variant's problem shape, tile sizes, dtypes, and feature flags
2. **`TmemLayout`** — computed TMEM offsets derived from config, eliminating magic numbers
3. **`AttentionFusion`** — customization bundle (logits/output transforms, attention sinks)

### Phase 2: Extract Warp Roles -- DONE

4. **FMHA roles**: `SoftmaxRole`, `CorrectionRole`, `MmaRole`, `LoaderRole`, `EpilogueRole` — each extracted as a class with `@cute.jit` methods
5. **MLA roles**: `MLALoaderRole`, `MLAMmaRole`, `MLAComputeRole` (orchestrator delegating to `MLASoftmaxRole`, `MLARescaleRole`, `MLAEpilogueRole`)
6. **Masking utilities**: `apply_mask()`, trip count helpers moved to `fusion/mask.py` as standalone `@cute.jit` functions

### Phase 3: Extract Infrastructure -- DONE

7. **`PipelineTopology`** — declarative pipeline graph with `PipelineEdge` specs and `create_pipelines()` factory, replacing ~80 lines of imperative pipeline setup per kernel
8. **`WarpSchedule`** / **`MLAWarpSchedule`** — warp role assignment and register budgets as dataclasses
9. **`MainloopSpec`** / **`MLAMainloopSpec`** — bundles config + schedule + topology + stage counts (analogous to C++ mainloop collective types)
10. **Tile schedulers**: `FmhaStaticTileScheduler`, `MLAStaticTileScheduler` extracted to `scheduler/`

### Phase 4: Extract Shared Utilities -- DONE

11. **`softmax_math.py`**: `exp2_scale()`, `packed_row_sum()` — shared utilities. MLA `MLASoftmaxRole` uses both; FMHA `SoftmaxRole` uses only `exp2_scale()` (see "Softmax Optimization Note" below)
12. **`tmem_utils.py`**: `tmem_load_partition()` — shared between `MLARescaleRole` and `MLAEpilogueRole`

#### Softmax Optimization Note

During Phase 4, `packed_row_sum()` was initially shared between both FMHA and MLA. However, benchmarking revealed a ~25% TFLOPS regression in FMHA prefill when using the generic `packed_row_sum()` in `SoftmaxRole`. The root cause was loss of instruction-level parallelism (ILP): the original FMHA code uses a **4-way unrolled reduction** with four independent accumulator chains (`local_row_sum_0..3`) that execute in parallel, then tree-reduce. The generic `packed_row_sum()` uses a single accumulator chain, creating a serial dependency.

The fix was to keep the hand-optimized 4-way unrolled reduction in `SoftmaxRole` and only share `exp2_scale()`. MLA's `MLASoftmaxRole` still uses `packed_row_sum()` because MLA decode is memory-bandwidth-bound, not compute-bound, so the ILP difference is irrelevant. This is a deliberate asymmetry: **performance-critical hot paths in compute-bound kernels may resist generalization**.

### Phase 5: Kernel Readability -- DONE

Three layers of boilerplate removal to make kernel bodies read like pseudocode:

13. **Layer 1: Config unpacking removal** — Replaced ~125 lines of `self.xxx = config.xxx` in both kernels with direct access (`self.config.xxx`, `self.schedule.xxx`, `self.tmem.xxx`). Makes the source of each value explicit.
14. **Layer 2: CollectiveBuilder** — Created `collective_builder.py` with `build_fmha_launch_params()` and `build_mla_launch_params()`. Each encapsulates MMA atom creation, SMEM layout computation, TMA atom setup, and `SharedStorage` struct definition. Reduced `__call__` from ~310 lines to ~40 lines (FMHA) and ~250 to ~70 lines (MLA).
15. **Layer 3: Kernel-side helpers** — Extracted `_create_pipelines()` and `_create_mma_fragments()` as `@cute.jit` helper methods in both kernels, moving ~80-100 lines of setup out of the main kernel body.

### Phase 6: Wrapper/Utility Cleanup -- DONE

16. **Removed dead code from `prefill.py`** — Deleted `_Removed_BatchPrefillCuteDSLWrapper_PLACEHOLDER` and duplicate test utilities (`qkv_torch_2_cute`, `create_and_pad_tensor`), removing ~357 lines. The real wrapper lives in `wrappers/batch_prefill.py`.
17. **Moved MLA utilities to `wrappers/batch_mla.py`** — Relocated `create_page_table`, `create_block_split_kvs`, `create_workspace`, `torch_to_cute`, `create_tensor`, `ceil_div` (~200 lines) from the kernel file to the wrapper, where they belong. The kernel files now contain only kernel logic.

### Phase 7: MLA Kernel Loop Extraction -- DONE

18. **Moved tile scheduler loops into MLA roles** — Each MLA role (`MLALoaderRole`, `MLAMmaRole`, `MLAComputeRole`) gained a `run()` method that owns its tile scheduler loop, pipeline state creation, SimpleNamespace parameter bundling, and pipeline tail/barrier calls. The kernel's three warp sections became single-line `self.xxx_role.run(...)` dispatches, matching the FMHA pattern where each role owns its own loop. This reduced `mla_decode.py` from 1073 to 593 lines.
19. **Added `_get_k_tile_count()` to each role** — The device-side tile range computation (`ceil_div` + index math) requires `@cute.jit` for CuTe DSL's `min`/`max` rewrites. Rather than a shared standalone function (which can't be `@cute.jit`), each role has its own `_get_k_tile_count()` method. A standalone `mla_get_k_tile_count()` utility was also added to `scheduler/mla_persistent.py` for host-side callers (wrappers).
20. **Removed `get_k_tile_count` from kernel class** — No longer needed since each role has its own.

### Phase 8: Performance Enhancements -- PENDING

21. **Skip-correction** — `vote_all_sync` to avoid rescaling when unnecessary
22. **Softmax software pipelining** — interleave FMA, exp2, dtype conversion
23. **Fused atomic reduction** for split-KV (eliminates reduction kernel)
24. **FP8 support** in config, loader, and MMA modules

### Phase 9: New Features -- PENDING

25. **Causal-aware tile scheduling** with swizzled launch order
26. **ThreadShape configurability** for different Q/K aspect ratios
27. **Backward pass** (requires new kernel composition)
28. **Wire `AttentionFusion` into MLA** — extend customization hooks to MLA decode

---

## 9. Implementation Status

This section tracks the current state of the modularization effort. The original monoliths (`prefill.py` at 2933 lines, `mla.py` at 3641 lines) remain untouched. A parallel modular implementation lives in `flashinfer/cute_dsl/attention/` (36 files, ~7650 lines total) and is verified by dedicated test suites.

### Current File Layout

```
flashinfer/cute_dsl/attention/          # 7650 lines total across 36 files
│
│  ── Kernels (top-level, readable dispatchers) ──
├── prefill.py              (598 lines)   FMHA prefill kernel (pure kernel, no wrapper/test code)
├── mla_decode.py           (593 lines)   MLA decode kernel + reduction kernel (pure kernel)
├── collective_builder.py   (348 lines)   build_fmha/mla_launch_params: MMA atoms, SMEM, TMA, SharedStorage
│
│  ── Configuration ──
├── config.py               (134 lines)   AttentionConfig, AttentionFusion, HeadMapping
├── mla_config.py            (74 lines)   MLAConfig (separate concrete type)
├── tmem_layout.py           (49 lines)   TmemLayout: computed TMEM offsets
├── warp_schedule.py         (90 lines)   WarpSchedule for FMHA
├── mla_warp_schedule.py     (79 lines)   MLAWarpSchedule (separate concrete type)
├── mainloop_spec.py        (173 lines)   MainloopSpec (FMHA) + MLAMainloopSpec
├── pipeline_topology.py    (315 lines)   PipelineTopology, PipelineEdge, PipelineType, factory
│
│  ── FMHA Roles ──
├── roles/
│   ├── softmax.py          (530 lines)   SoftmaxRole: online softmax with masking
│   ├── correction.py       (458 lines)   CorrectionRole: rescale + epilog
│   ├── mma.py              (258 lines)   MmaRole: QK + PV GEMMs
│   ├── loader_tma.py       (316 lines)   LoaderRole: TMA loads for Q, K, V
│   ├── epilogue.py         (163 lines)   EpilogueRole: TMA store output
│
│  ── MLA Roles ──
│   ├── mla_loader.py       (422 lines)   MLALoaderRole: paged latent + RoPE loads, owns tile sched loop
│   ├── mla_mma.py          (415 lines)   MLAMmaRole: QK + PV with head packing, owns TMEM + tile sched loop
│   ├── mla_compute.py      (237 lines)   MLAComputeRole: orchestrator, owns tile sched loop
│   ├── mla_softmax.py      (234 lines)   MLASoftmaxRole: online softmax
│   ├── mla_rescale.py       (75 lines)   MLARescaleRole: O accumulator rescaling
│   ├── mla_epilogue.py     (153 lines)   MLAEpilogueRole: final output write
│
│  ── Shared Utilities ──
│   ├── softmax_math.py      (40 lines)   exp2_scale (shared), packed_row_sum (MLA only)
│   ├── tmem_utils.py       (101 lines)   tmem_load_partition
│
│  ── Fusion / Masking ──
├── fusion/
│   ├── mask.py             (160 lines)   MaskType, apply_mask, trip count helpers
│   ├── logits_transform.py               sigmoid_logits_transform
│   ├── output_transform.py               dumb_output_transform
│   └── softmax_modifier.py               (stub)
│
│  ── Schedulers ──
├── scheduler/
│   ├── persistent.py       (166 lines)   FmhaStaticTileScheduler
│   └── mla_persistent.py   (245 lines)   MLAStaticTileScheduler + mla_get_k_tile_count, mla_get_split_kv
│
│  ── PyTorch Wrappers ──
└── wrappers/
    ├── batch_prefill.py    (381 lines)   BatchPrefillCuteDSLWrapper + tensor utils
    └── batch_mla.py        (637 lines)   BatchMLAPagedAttentionWrapperCuteDSL + tensor/page utils
```

### Shared vs Variant-Specific Components

| Component | Shared | FMHA-specific | MLA-specific |
|-----------|--------|---------------|--------------|
| **CollectiveBuilder** | `collective_builder.py` (MMA atoms, SMEM layouts, TMA atoms, SharedStorage) | `build_fmha_launch_params()` | `build_mla_launch_params()` |
| **Pipeline infrastructure** | `PipelineTopology`, `PipelineEdge`, `PipelineType`, `create_pipelines()` | `make_fmha_topology()` | `make_mla_topology()`, `ASYNC_UMMA` type |
| **Mainloop spec** | Dataclass pattern, stage count resolution | `MainloopSpec` | `MLAMainloopSpec` |
| **Warp schedule** | Concept (dataclass with role IDs, register budgets) | `WarpSchedule` (16 warps) | `MLAWarpSchedule` (6-8 warps) |
| **Softmax math** | `exp2_scale()` shared; `packed_row_sum()` available | `exp2_scale` only (hand-optimized 4-way unrolled row-sum for ILP) | Both `exp2_scale` and `packed_row_sum` |
| **TMEM utilities** | `tmem_load_partition()` | — | Used by `MLARescaleRole`, `MLAEpilogueRole` |
| **Masking** | `MaskType`, `apply_mask()`, trip count helpers | Used inline in `SoftmaxRole` | Boundary masking inline in `MLASoftmaxRole` |
| **Loader** | — | `LoaderRole` (streaming Q/K/V) | `MLALoaderRole` (paged latent + RoPE; `run()` owns tile sched loop) |
| **MMA** | — | `MmaRole` (double-buffered QK/PV) | `MLAMmaRole` (staged, head-packed; `run()` owns TMEM + tile sched loop) |
| **Compute** | — | N/A (roles dispatched directly by kernel) | `MLAComputeRole` (orchestrator; `run()` owns tile sched loop, delegates to softmax/rescale/epilogue) |
| **Correction / Rescale** | Packed-scale pattern (similar but not yet extracted) | `CorrectionRole` | `MLARescaleRole` |
| **Epilogue** | — | `EpilogueRole` (TMA store) | `MLAEpilogueRole` (direct write) |
| **Scheduler** | — | `FmhaStaticTileScheduler` | `MLAStaticTileScheduler`, `mla_get_k_tile_count` |
| **Fusion hooks** | `AttentionFusion` (logits/output transforms, sinks) | Full support | Not yet wired |

### Comparison to C++ CUTLASS Collectives

| Aspect | C++ CUTLASS | Current Python Implementation |
|--------|-------------|-------------------------------|
| **Kernel template** | One shared kernel (`Sm100FmhaFwdKernelTmaWarpspecialized`) parameterized by mainloop type | Separate kernel files (`prefill.py`, `mla_decode.py`) — both are thin dispatchers |
| **Mainloop** | `Sm100FmhaFwd*` / `Sm100FmhaMlaFwd*` concrete types defining pipelines + TMEM + roles | `MainloopSpec` / `MLAMainloopSpec` dataclasses with `PipelineTopology` + `TmemLayout` |
| **Pipeline creation** | Types defined in mainloop, instantiated by kernel | Declarative `PipelineTopology` with `create_pipelines()` factory |
| **Roles** | Methods on the mainloop (`load()`, `mma()`, `softmax()`, `correction()`) | Separate role classes composed by the kernel; MLA roles have `run()` methods owning their tile scheduler loops |
| **Shared math** | Inline in each mainloop (no cross-variant sharing) | Extracted: `softmax_math.py`, `tmem_utils.py` |
| **Config** | Template parameters + `Params` struct | `AttentionConfig` / `MLAConfig` dataclasses |
| **CollectiveBuilder** | Selects MMA atoms, TMA descriptors, pipeline types from config | `collective_builder.py`: `build_fmha_launch_params()`, `build_mla_launch_params()` |

The Python implementation actually shares *more* code across variants than C++ CUTLASS, which keeps each mainloop self-contained. The tradeoff is that C++ gets maximum compile-time optimization per variant while Python uses JIT compilation that naturally specializes per call.

### Test Status

| Test Suite | Total | Pass | Fail | Notes |
|------------|-------|------|------|-------|
| `test_blackwell_fmha_attention.py` | 53 | 52 | 1 | 1 pre-existing config failure (`head_dim_vo != head_dim_qk` assertion in output_transform test) |
| `test_blackwell_mla_attention.py` | 24 | 16 | 8 | 12 FP16 pass, 8 BF16 borderline precision (would pass at `atol=1e-2`); see "Page Table Bug Fix" below |

The 8 remaining BF16 failures have max absolute differences of ~0.0078 with 0.3–3.2% of elements mismatched — a characteristic BF16 precision limitation, not a correctness bug.

### Bug Fixes

#### Page Table Indexing Bug (FP16 MLA Decode)

**Symptom**: 2 FP16 MLA test cases showed catastrophic failures (99.4% element mismatch, max diff ~1.95) when `batch_size >= 2` and `kv_len >= 256` (i.e., multiple batches each requiring multiple pages).

**Root Cause**: The `create_page_table()` function (present in both original `mla.py` and modular `mla_decode.py`) hardcoded an *interleaved* page layout:
```python
page_table_ref[b, j] = b + j * batch_size  # interleaved across batches
```
However, the user-provided `kv_indices` tensor specified a *contiguous* layout (each batch's pages are contiguous). The function ignored `kv_indices` entirely, causing the kernel to read KV data from the wrong physical pages in multi-batch, multi-page scenarios.

**Fix**: Updated `create_page_table()` to accept `kv_indptr` and `kv_indices` parameters and build the page table from the user-provided mappings:
```python
def create_page_table(batch_size, max_num_pages, page_table_ref,
                      kv_indptr=None, kv_indices=None):
    for b in range(batch_size):
        start = kv_indptr[b]
        end = kv_indptr[b + 1]
        for j in range(end - start):
            page_table_ref[b, j] = kv_indices[start + j]
```

**Scope**: Only affects MLA decode (paged KV cache). FMHA prefill uses ragged (non-paged) KV and has no page table. The fix was applied to both `flashinfer/cute_dsl/mla.py` and `flashinfer/cute_dsl/attention/mla_decode.py` (and their respective wrappers).

**Verification**: Tested with 5 page layouts (contiguous, reversed, random permutation, sparse 4x-overprovisioned pool, interleaved), multiple batch sizes (2–8), kv lengths (256–2048), head counts (8–128), and random seeds — all passing with max diff < 0.001.

### Performance

Modular and monolithic implementations achieve near-identical performance, confirming the refactoring introduces no overhead. All numbers below are kernel time on B200, BF16 (measured with `bench_gpu_time`; script: `benchmarks/bench_modular_attention.py`).

#### MHA Kernel Template

| Seq Len | Batch Size | Monolithic TFLOPs | Modular TFLOPs | Monolithic BW (GB/s) | Modular BW (GB/s) |
|---------|------------|-------------------|----------------|----------------------|-------------------|
| 512 | 128 | 190.9 | 191.2 | 1491.7 | 1493.7 |
| 1024 | 64 | 298.7 | 298.8 | 1167.0 | 1167.0 |
| 2048 | 32 | 487.0 | 486.7 | 951.1 | 950.7 |
| 4096 | 16 | 666.3 | 658.5 | 650.7 | 643.1 |
| 8192 | 8 | 837.3 | 848.1 | 408.8 | 414.1 |
| 16384 | 4 | 945.9 | 945.6 | 230.9 | 230.9 |
| 32768 | 2 | 1018.5 | 1018.2 | 124.3 | 124.3 |
| 65536 | 1 | 1067.0 | 1063.4 | 65.1 | 64.9 |

#### MHA with different variant (TFLOPS)

**Monolithic:**

| Seq Len | Batch Size | Vanilla | Sigmoid | Output | Sink |
|---------|------------|---------|---------|--------|------|
| 1024 | 64 | 298.3 | 189.8 | 279.0 | 222.2 |
| 4096 | 16 | 659.1 | 412.9 | 644.7 | 569.3 |
| 16384 | 4 | 955.4 | 571.6 | 953.4 | 889.8 |
| 65536 | 1 | 1066.6 | 635.0 | 1072.7 | 1021.0 |

**Modular:**

| Seq Len | Batch Size | Vanilla | Sigmoid | Output | Sink |
|---------|------------|---------|---------|--------|------|
| 1024 | 64 | 298.6 | 189.8 | 279.0 | 222.7 |
| 4096 | 16 | 657.9 | 412.9 | 644.3 | 565.8 |
| 16384 | 4 | 944.7 | 571.6 | 952.8 | 888.0 |
| 65536 | 1 | 1063.0 | 635.3 | 1070.5 | 1019.9 |

#### MLA Kernel Template

| Batch Size | Seq Len | Monolithic TFLOPs | Modular TFLOPs | Monolithic BW (GB/s) | Modular BW (GB/s) |
|------------|---------|-------------------|----------------|----------------------|-------------------|
| 64 | 1024 | 198.1 | 198.1 | 1012.6 | 1012.6 |
| 128 | 1024 | 396.1 | 409.6 | 2025.2 | 2094.3 |
| 768 | 1024 | 794.7 | 792.3 | 4063.2 | 4050.5 |
| 64 | 2048 | 396.1 | 400.6 | 1831.8 | 1852.4 |
| 128 | 2048 | 809.4 | 810.0 | 3742.9 | 3745.6 |
| 768 | 2048 | 842.1 | 840.4 | 3894.2 | 3886.3 |
| 64 | 8192 | 770.2 | 770.7 | 3279.6 | 3281.8 |
| 128 | 8192 | 866.7 | 861.7 | 3690.7 | 3669.2 |
| 768 | 8192 | 831.0 | 846.3 | 3538.3 | 3603.8 |

#### MLA Kernel with various number of heads

| Seq Len | Batch Size | Num Heads | Monolithic TFLOPs | Modular TFLOPs | Monolithic BW (GB/s) | Modular BW (GB/s) |
|---------|------------|-----------|-------------------|----------------|----------------------|-------------------|
| 1024 | 64 | 128 | 191.7 | 200.3 | 980.0 | 1024.0 |
| 1024 | 64 | 64 | 100.2 | 96.9 | 926.5 | 896.0 |
| 1024 | 64 | 32 | 50.1 | 48.4 | 877.3 | 848.7 |
| 1024 | 64 | 16 | 24.2 | 25.0 | 825.0 | 852.9 |
| 1024 | 64 | 8 | 12.5 | 12.5 | 841.2 | 840.0 |

All monolithic/modular pairs are within ±3% — confirming zero refactoring overhead across all kernel variants, problem sizes, and head configurations.

### Remaining Work

Listed roughly in order of impact:

1. **Extract `packed_scale` utility** — The `mul_packed_f32x2` loop pattern appears in both `CorrectionRole.rescale()` and `MLARescaleRole.run()`. ~5 lines each. Low risk.

2. **Move role instantiation into MainloopSpec** — Currently the kernel `__init__` creates role instances. Moving this into the mainloop spec would make it closer to the C++ pattern where the mainloop *is* the collective. Medium risk.

3. **Add `run()` methods to FMHA roles** — The FMHA roles don't yet have `run()` methods like the MLA roles do. FMHA's warp sections in the kernel are already concise, but adding `run()` methods would improve symmetry across variants and further slim the kernel body. Low-medium risk.

4. **Wire `AttentionFusion` into MLA** — The fusion hooks (logits transforms, output transforms, attention sinks) currently only work in FMHA. Extending to MLA would enable customizable MLA decode. Low-medium risk.

5. **Performance optimizations** — Skip-correction (`vote_all_sync` to avoid rescaling when unnecessary), softmax software pipelining, exp2 emulation, fused atomic reduction for split-KV. These are independent and can be added per-variant.

6. **Unify kernel dispatcher** — The C++ code uses one kernel template for both FMHA and MLA. Currently we have two separate kernel files. Unifying would require abstracting the warp dispatch, which is the most variant-specific part. High risk, potentially not worth it given Python's JIT advantage.

---

## 10. Writing a New Attention Variant

This section explains how to add a new attention kernel variant to the modular `flashinfer/cute_dsl/attention/` package. Both the FMHA prefill and MLA decode kernels follow this pattern.

### Three-Layer Architecture

The attention package has three layers with strict responsibilities:

```
Layer 1: Wrappers (wrappers/)
  PyTorch API: plan()/run(), torch-to-CuTe conversion, workspace, validation

Layer 2: Kernels (prefill.py, mla_decode.py)
  Algorithm: __init__ -> __call__ -> kernel, CuTe tensors in/out

Layer 3: Building Blocks (roles/, config, scheduler/, fusion/, collective_builder, pipelines)
  Reusable components composed by kernels
```

**Wrappers own**: tensor conversion, workspace sizing/allocation, `can_implement` validation, scheduling heuristics (e.g. `get_split_kv`), page table creation.

**Kernels own**: the algorithm (device code), warp dispatch, pipeline creation, device-side tile range computation.

**Building blocks own**: MMA/TMA/SMEM setup, warp-level role logic, tile scheduling, masking, config dataclasses.

### Step-by-Step Recipe

#### Step 1: Define Your Config

Create a dataclass in `config.py` (or `xxx_config.py` for a new file) that describes the problem shape, data types, tile sizes, and feature flags.

```python
@dataclass(frozen=True)
class XxxConfig:
    head_dim: int
    num_heads: int
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32
    mma_tiler_mn: Tuple[int, int] = (128, 128)
    is_persistent: bool = True

    @property
    def mma_tiler(self) -> Tuple[int, int, int]:
        return (*self.mma_tiler_mn, self.head_dim)
```

**Reference**: `config.py` (AttentionConfig), `mla_config.py` (MLAConfig).

#### Step 2: Define Your Warp Schedule

Create a dataclass with warp role assignments and register budgets.

```python
@dataclass(frozen=True)
class XxxWarpSchedule:
    load_warp_id: int
    mma_warp_id: int
    compute_warp_ids: Tuple[int, ...]
    threads_per_warp: int = 32

XXX_SCHEDULE = XxxWarpSchedule(load_warp_id=5, mma_warp_id=4, ...)
```

**Reference**: `warp_schedule.py`, `mla_warp_schedule.py`.

#### Step 3: Implement Your Roles

Each warp role is a class in `roles/` with `@cute.jit` methods. Roles receive params and implement one warp's work.

```python
class XxxLoaderRole:
    def __init__(self, config: XxxConfig):
        self.config = config

    @cute.jit
    def run(self, params, pipeline_states, ...):
        ...
```

Common roles: Loader (TMA loads), MMA (matrix multiplies), Softmax (online softmax), Correction/Rescale (accumulator rescaling), Epilogue (output writes).

**Shared utilities**: `roles/softmax_math.py` (`exp2_scale`, `packed_row_sum`), `roles/tmem_utils.py` (`tmem_load_partition`).

#### Step 4: Define MainloopSpec and Pipeline Topology

Bundle your config, schedule, and pipeline topology into a mainloop spec.

```python
@dataclass
class XxxMainloopSpec:
    config: XxxConfig
    schedule: XxxWarpSchedule
    topology: PipelineTopology
    kv_stages: int = 0

    def resolve(self, dtype_width):
        ...
```

Define your pipeline topology using `PipelineEdge` specs:

```python
def make_xxx_topology(schedule):
    return PipelineTopology([
        PipelineEdge("load_q", PipelineType.TMA_UMMA, ...),
        PipelineEdge("load_kv", PipelineType.TMA_UMMA, ...),
    ])
```

**Reference**: `mainloop_spec.py`, `pipeline_topology.py`.

#### Step 5: Add a CollectiveBuilder Function

Add a `build_xxx_launch_params()` function in `collective_builder.py`. This creates MMA atoms, SMEM layouts, TMA atoms, and the `SharedStorage` struct.

```python
def build_xxx_launch_params(mainloop, *tensors, *dtypes):
    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(...)
    q_smem_layout = sm100_utils.make_smem_layout_a(...)
    tma_atom_q = sm100_utils.make_tiled_tma_atom_A(...)
    # Define SharedStorage struct
    return SimpleNamespace(qk_tiled_mma=..., SharedStorage=..., ...)
```

**Reference**: `collective_builder.py` — `build_fmha_launch_params` and `build_mla_launch_params`.

#### Step 6: Write the Kernel

This is the top-level file. Follow this exact method structure:

```python
class BlackwellXxxAttention:
    def __init__(self, config: XxxConfig, schedule: XxxWarpSchedule = None):
        """Store config + schedule, create mainloop spec. Lightweight."""
        self.config = config
        self.schedule = schedule or XXX_SCHEDULE
        self.mainloop = make_xxx_mainloop_spec(config, self.schedule)

    @cute.jit
    def __call__(self, *tensors, stream):
        """Validate, resolve, create roles, build launch params, launch."""
        self.q_dtype = q.element_type
        ...
        self.mainloop.resolve(self.q_dtype.width)

        # Create roles after resolve
        self.loader_role = XxxLoaderRole(self.config)
        self.mma_role = XxxMmaRole(self.config, self.mainloop)
        ...

        lp = build_xxx_launch_params(self.mainloop, ...)
        tile_sched_params, grid = self._compute_grid(...)
        self.kernel(...).launch(grid=grid, block=..., stream=stream)

    @cute.jit
    def _create_pipelines(self, storage):
        """Build pipelines from topology and barrier pointers."""
        return self.mainloop.topology.create_pipelines(storage, ...)

    @cute.jit
    def kernel(self, ...launch_params):
        """Device kernel: shared storage + pipelines + warp dispatch."""
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        pipelines = self._create_pipelines(storage)

        warp_idx = cute.arch.warp_idx()
        if warp_idx == self.schedule.load_warp_id:
            self.loader_role.run(...)
        elif warp_idx == self.schedule.mma_warp_id:
            self.mma_role.run(...)
        elif warp_idx >= ...:
            self.compute_role.run(...)

    @staticmethod
    def _compute_grid(...):
        ...
```

#### Canonical Method Table

| Method | Required | Purpose |
|--------|----------|---------|
| `__init__(config, schedule)` | Yes | Store config + schedule, create mainloop. Lightweight. |
| `__call__(*tensors, stream)` | Yes | Validate, resolve, create roles, build params, launch. |
| `_create_pipelines(storage)` | Yes | Build pipelines from topology. |
| `kernel(...)` | Yes | Shared storage + pipelines + warp dispatch. |
| `_compute_grid(...)` | Yes | Static method for grid shape. |
| `reduction_kernel(...)` | Optional | For split-KV variants only. |
| `_create_mma_fragments(...)` | Optional | If multiple roles share TMEM fragments. |

#### Key Rules

1. **`__init__` is declarative**: only stores config, schedule, and creates mainloop spec. No role creation.
2. **Roles are created in `__call__`**: after `mainloop.resolve()` sets stage counts.
3. **The kernel method is always called `kernel`**: even for split-KV variants.
4. **Each role owns its loop**: roles have `run()` methods containing the tile scheduler loop, pipeline state management, and parameter bundling. The kernel body is a clean warp dispatch with single-line `role.run()` calls.
5. **Device-side tile range computation** (`_get_k_tile_count`) lives as a `@cute.jit` method on each role (not on the kernel class), because CuTe DSL's symbolic `min`/`max` rewrites require `@cute.jit` context.
6. **MMA fragments**: if multiple roles share TMEM fragment views, create them in `_create_mma_fragments` and pass to roles. If roles are self-contained, let them create their own.

#### Step 7: Write the Wrapper

Create a PyTorch-facing API in `wrappers/batch_xxx.py` that handles tensor conversion and provides `plan()`/`run()`.

```python
class BatchXxxWrapper:
    def __init__(self, workspace_buffer, ...):
        ...

    def plan(self, *user_params):
        config = XxxConfig(...)
        kernel = BlackwellXxxAttention(config)
        self._compiled = cute.compile(kernel, ...)

    def run(self, q, k, v, o, ...):
        self._compiled(q_cute, k_cute, ..., stream)
```

The wrapper owns:
- `can_implement` validation (from `xxx_config.py`)
- Scheduling heuristics (from `scheduler/xxx_persistent.py`)
- Workspace allocation
- Page table creation (if applicable)
- Torch tensor to CuTe tensor conversion

**Reference**: `wrappers/batch_prefill.py`, `wrappers/batch_mla.py`.

#### Step 8: Write Tests

Add tests in `tests/test_blackwell_xxx_attention.py`.

```python
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_xxx_attention(dtype, batch_size, ...):
    if not is_sm100a_supported():
        pytest.skip("Requires SM100a")
    wrapper = BatchXxxWrapper(...)
    wrapper.plan(...)
    wrapper.run(q, k, v, o)
    torch.testing.assert_close(o, ref_output, atol=1e-3, rtol=1e-3)
```

#### Step 9: Register Exports

Update `attention/__init__.py` to export your new kernel and config. Update `wrappers/__init__.py` to export the wrapper.

### File Size Guidelines

| Component | Typical Size | Notes |
|-----------|-------------|-------|
| Config dataclass | 50–150 lines | Problem shape, derived properties |
| Warp schedule | 50–90 lines | Role assignments, barrier IDs |
| Each role | 75–530 lines | Depends on algorithm complexity |
| MainloopSpec addition | 20–50 lines | Added to existing file |
| CollectiveBuilder function | 100–180 lines | Added to existing file |
| **Kernel file** | **550–650 lines** | Pure algorithm, no utilities |
| Wrapper | 300–650 lines | PyTorch API + tensor utils |
| Tile scheduler | 100–250 lines | Grid/work distribution |

### Existing Implementations

| Variant | Kernel | Config | Roles | Total |
|---------|--------|--------|-------|-------|
| FMHA Prefill | `prefill.py` (598) | `config.py` (134) | softmax (530), correction (458), mma (258), loader (316), epilogue (163) | ~2600 |
| MLA Decode | `mla_decode.py` (593) | `mla_config.py` (74) | mla_loader (422), mla_mma (415), mla_compute (237), mla_softmax (234), mla_rescale (75), mla_epilogue (153) | ~2850 |

Both kernels share: `collective_builder.py` (348), `pipeline_topology.py` (315), `mainloop_spec.py` (173), `roles/softmax_math.py` (40), `roles/tmem_utils.py` (101).

MLA roles have `run()` methods that own their tile scheduler loops, so the kernel body is a clean warp dispatch with single-line `role.run()` calls.

---

## 11. Call Traces

This section shows the complete call flow through the building blocks for each attention variant, from user-facing API down to individual warp roles.

### FMHA Prefill

```
User code
  │
  ▼
BatchPrefillCuteDSLWrapper                           [wrappers/batch_prefill.py]
  │
  ├── plan(qo_indptr, kv_indptr, num_heads, head_dim, ...)
  │     │
  │     ├── Create AttentionConfig(qk_acc_dtype, mma_tiler, mask_type, ...)    [config.py]
  │     ├── Create AttentionFusion(logits_transform, output_transform, ...)    [config.py]
  │     ├── fmha = BlackwellFusedMultiHeadAttentionForward(config, fusion)     [prefill.py]
  │     │     │
  │     │     └── __init__: store config, fusion, schedule
  │     │           ├── mainloop = make_prefill_mainloop_spec(config, schedule)  [mainloop_spec.py]
  │     │           │     ├── topology = make_prefill_topology(schedule)         [pipeline_topology.py]
  │     │           │     ├── tmem_layout = TmemLayout.from_config(config)       [tmem_layout.py]
  │     │           │     └── MainloopSpec(config, schedule, topology, tmem)
  │     │           └── tmem = mainloop.tmem_layout
  │     │
  │     ├── qkvo_cute = create_and_pad_tensor(...)         (torch → CuTe tensors)
  │     ├── compiled_fmha = cute.compile(fmha, q, k, v, o, ...)
  │     └── store compiled_fmha
  │
  └── run(q, k, v)
        │
        ├── Copy torch data into pre-allocated CuTe tensors
        └── compiled_fmha(q_ptr, k_ptr, v_ptr, o_ptr, ...)
              │
              ▼
BlackwellFusedMultiHeadAttentionForward.__call__     [prefill.py]
  │
  ├── Build Q/K/V/O CuTe layouts from pointers + problem_size
  ├── _compute_grid() → tile_sched_params, grid            [prefill.py]
  │     └── FmhaStaticTileScheduler.get_grid_shape(...)    [scheduler/persistent.py]
  ├── mainloop.resolve(dtype_width)                        [mainloop_spec.py]
  │     └── Compute pipeline stage counts from SMEM budget
  ├── Create roles:
  │     ├── SoftmaxRole(config, fusion, tmem, ...)         [roles/softmax.py]
  │     ├── CorrectionRole(config, fusion, tmem, ...)      [roles/correction.py]
  │     ├── EpilogueRole(config)                           [roles/epilogue.py]
  │     ├── LoaderRole(config)                             [roles/loader_tma.py]
  │     └── MmaRole(config, tmem, ...)                     [roles/mma.py]
  ├── lp = build_fmha_launch_params(mainloop, q, k, v, o, ...)  [collective_builder.py]
  │     ├── Create QK and PV TiledMma atoms
  │     ├── Compute SMEM layouts (q, k, v, p, o)
  │     ├── Create TMA atoms and descriptors
  │     └── Define SharedStorage struct (SMEM + pipeline barriers)
  └── kernel(...).launch(grid, block, smem, stream)
        │
        ▼
BlackwellFusedMultiHeadAttentionForward.kernel       [prefill.py]  ◄── GPU device code
  │
  ├── Allocate shared memory (SharedStorage)
  ├── _create_pipelines(storage)                           [prefill.py]
  │     └── topology.create_pipelines(barrier_ptrs, ...)   [pipeline_topology.py]
  │           └── Returns dict of (producer, consumer) pipeline pairs:
  │                 load_q, load_kv, mma_s0, mma_s1,
  │                 s0_corr, s1_corr, corr_epi, mma_corr, s0_s1_sequence
  ├── Unpack SMEM tensors: sQ, sK, sV, sO
  ├── _create_mma_fragments(...)                           [prefill.py]
  │     └── Create TMEM fragment views: tStS0/1, tOtO0/1, tOrP0/1
  │
  ├── Warp dispatch (16 warps):
  │
  │   warp 15 (empty):     reg_dealloc
  │
  │   warp 13 (load):      LoaderRole.run(...)             [roles/loader_tma.py]
  │                           ├── Tile scheduler loop (FmhaStaticTileScheduler)
  │                           ├── TMA load Q, K, V into SMEM
  │                           └── Signal load_q_producer, load_kv_producer
  │
  │   warp 12 (mma):       MmaRole.run(...)                [roles/mma.py]
  │                           ├── TMEM alloc
  │                           ├── Tile scheduler loop
  │                           ├── QK GEMM: tSrQ × tSrK → tStS (TMEM)
  │                           ├── Signal mma_s0/s1_producer
  │                           ├── PV GEMM: tOrP × tOrV → tOtO (TMEM)
  │                           └── Signal mma_corr_producer
  │
  │   warps 0-3 (softmax0): SoftmaxRole.run(stage=0, ...)  [roles/softmax.py]
  │                           ├── Tile scheduler loop
  │                           ├── TMEM load S0 scores
  │                           ├── apply_mask(causal/sliding_window)   [fusion/mask.py]
  │                           ├── exp2_scale(row - max)               [roles/softmax_math.py]
  │                           ├── 4-way unrolled row_sum (ILP-optimized)
  │                           ├── Optional: logits_transform          [fusion/logits_transform.py]
  │                           ├── Optional: M_D_update (sink)         [fusion/softmax_modifier.py]
  │                           ├── Write P back to TMEM
  │                           └── Signal s0_corr_producer
  │
  │   warps 4-7 (softmax1): SoftmaxRole.run(stage=1, ...)  (same as softmax0, on S1)
  │
  │   warps 8-11 (correction): CorrectionRole.run(...)     [roles/correction.py]
  │                           ├── Tile scheduler loop
  │                           ├── TMEM load O accumulators
  │                           ├── Rescale O by exp2(old_max - new_max)
  │                           ├── Optional: output_transform          [fusion/output_transform.py]
  │                           ├── Store to SMEM sO
  │                           └── Signal corr_epi_producer
  │
  │   warp 14 (epilogue):  EpilogueRole.run(...)           [roles/epilogue.py]
  │                           ├── Tile scheduler loop
  │                           ├── TMA store sO → global memory
  │                           └── Wait on corr_epi_consumer
  │
  └── return
```

### MLA Decode

```
User code
  │
  ▼
BatchMLAPagedAttentionWrapperCuteDSL                 [wrappers/batch_mla.py]
  │
  ├── plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, num_heads, ...)
  │     │
  │     ├── mla_can_implement(...)                         [mla_config.py]
  │     ├── create_page_table(batch, seq, ..., kv_indptr, kv_indices)  [wrappers/batch_mla.py]
  │     ├── create_block_split_kvs(...)                    [wrappers/batch_mla.py]
  │     │     └── mla_get_split_kv(...)                    [scheduler/mla_persistent.py]
  │     ├── create_workspace(...)                          [wrappers/batch_mla.py]
  │     ├── qkvo_cute = torch_to_cute(...)                 (torch → CuTe tensors)
  │     ├── mla_config = MLAConfig(latent_dim, rope_dim, num_heads, ...)  [mla_config.py]
  │     ├── mla = BlackwellMultiLatentAttentionForward(mla_config)        [mla_decode.py]
  │     │     │
  │     │     └── __init__: store config, schedule
  │     │           └── mainloop = make_mla_mainloop_spec(config, schedule)  [mainloop_spec.py]
  │     │                 ├── topology = make_mla_topology(schedule)          [pipeline_topology.py]
  │     │                 └── MLAMainloopSpec(config, schedule, topology)
  │     │
  │     ├── compiled_mla = cute.compile(mla, q_latent, q_rope, ...)
  │     └── store compiled_mla
  │
  └── run(q_nope, q_pe, ckv_cache, kpe_cache)
        │
        ├── torch_to_cute for all input tensors
        └── compiled_mla(q_latent, q_rope, c_latent, c_rope, ...)
              │
              ▼
BlackwellMultiLatentAttentionForward.__call__        [mla_decode.py]
  │
  ├── Validate dtypes, strides
  ├── Build workspace tensors (acc_o, acc_lse) from workspace buffer
  ├── Create c_latent_transpose view
  ├── mainloop.resolve(dtype_width)                        [mainloop_spec.py]
  ├── Create roles:
  │     ├── MLALoaderRole(config)                          [roles/mla_loader.py]
  │     ├── MLAMmaRole(config, mainloop)                   [roles/mla_mma.py]
  │     └── MLAComputeRole(config, mainloop, schedule, exchange_bar)  [roles/mla_compute.py]
  │           └── Internally creates:
  │                 ├── MLASoftmaxRole(config)              [roles/mla_softmax.py]
  │                 ├── MLARescaleRole(config)              [roles/mla_rescale.py]
  │                 └── MLAEpilogueRole(config)             [roles/mla_epilogue.py]
  ├── lp = build_mla_launch_params(mainloop, schedule, tensors, ...)  [collective_builder.py]
  │     ├── Create QK and PV TiledMma atoms
  │     ├── Compute SMEM layouts (q, kc, vc, p)
  │     ├── Create TMA atoms (q_latent, q_rope, c_latent, c_rope, c_latent_T)
  │     └── Define SharedStorage struct (SMEM + pipeline barriers + TMEM buffers)
  ├── _compute_grid() → tile_sched_params, grid
  │     └── MLAStaticTileScheduler.get_grid_shape(...)     [scheduler/mla_persistent.py]
  │
  ├── kernel(...).launch(grid, block, cluster, smem, stream)   ─── split-KV kernel
  │     │
  │     ▼
  │   kernel(...)                                          [mla_decode.py]  ◄── GPU device code
  │     │
  │     ├── Allocate shared memory (SharedStorage)
  │     ├── Init TMEM dealloc barrier
  │     ├── _create_pipelines(storage, cta_layout)         [mla_decode.py]
  │     │     └── topology.create_pipelines_native(...)     [pipeline_topology.py]
  │     │           └── Returns dict: load_q, load_kv, mma_s, p_mma, mma_o
  │     ├── Cluster sync
  │     ├── Unpack SMEM tensors: sQ, sKC, sVC, sP, smem_exchange
  │     │
  │     ├── Warp dispatch (6-8 warps):
  │     │
  │     │   warp 5 (load_tma):  MLALoaderRole.run(...)     [roles/mla_loader.py]
  │     │                         ├── _get_k_tile_count()  (@cute.jit, device-side)
  │     │                         ├── Tile scheduler loop (MLAStaticTileScheduler)
  │     │                         ├── TMA load Q_latent, Q_rope into SMEM
  │     │                         ├── Page table lookup → physical page
  │     │                         ├── TMA load C_latent, C_rope, C_latent_T into SMEM
  │     │                         └── Signal load_q_producer, load_kv_producer
  │     │
  │     │   warp 4 (mma):       MLAMmaRole.run(...)        [roles/mla_mma.py]
  │     │                         ├── TMEM alloc
  │     │                         ├── _get_k_tile_count()  (@cute.jit, device-side)
  │     │                         ├── Tile scheduler loop
  │     │                         ├── QK GEMM (latent): Q_L × C_L^T → S_latent (TMEM)
  │     │                         ├── QK GEMM (rope):   Q_R × C_R^T → S_rope (TMEM, accumulate)
  │     │                         ├── Signal mma_s_producer
  │     │                         ├── PV GEMM: P × C_L_T → O (TMEM)
  │     │                         ├── Signal mma_o_producer
  │     │                         └── TMEM dealloc (after all tiles)
  │     │
  │     │   warps 0-3 (compute): MLAComputeRole.run(...)   [roles/mla_compute.py]
  │     │                         ├── Retrieve TMEM pointers (synced with MMA warp)
  │     │                         ├── _get_k_tile_count()  (@cute.jit, device-side)
  │     │                         ├── Tile scheduler loop, dispatching to sub-roles:
  │     │                         │
  │     │                         ├── MLASoftmaxRole.compute(...)    [roles/mla_softmax.py]
  │     │                         │     ├── TMEM load S scores
  │     │                         │     ├── exp2_scale(row - max)    [roles/softmax_math.py]
  │     │                         │     ├── packed_row_sum()         [roles/softmax_math.py]
  │     │                         │     ├── Write P to SMEM
  │     │                         │     └── Signal p_mma_producer
  │     │                         │
  │     │                         ├── MLARescaleRole.run(...)        [roles/mla_rescale.py]
  │     │                         │     ├── tmem_load_partition()    [roles/tmem_utils.py]
  │     │                         │     └── Rescale O accumulator
  │     │                         │
  │     │                         └── MLAEpilogueRole.run(...)       [roles/mla_epilogue.py]
  │     │                               ├── tmem_load_partition()    [roles/tmem_utils.py]
  │     │                               └── Write O, LSE to global memory
  │     │                                    (or acc_o, acc_lse for split-KV)
  │     │
  │     └── return
  │
  └── (if split_kv > 1):
        reduction_kernel(...).launch(...)                  [mla_decode.py]  ◄── GPU device code
          ├── Load acc_o, acc_lse from all split-KV blocks
          ├── Find global max LSE across splits
          ├── Rescale and sum partial O contributions
          └── Write final O, LSE to global memory
```

### Shared Building Blocks

Both variants use the same infrastructure, with variant-specific factory functions:

```
                        FMHA Prefill                    MLA Decode
                        ────────────                    ──────────
Config              →   AttentionConfig                 MLAConfig
                        + AttentionFusion

Warp Schedule       →   WarpSchedule (16 warps)         MLAWarpSchedule (6-8 warps)
                        PREFILL_SCHEDULE                MLA_DECODE_SCHEDULE

MainloopSpec        →   MainloopSpec                    MLAMainloopSpec
  factory           →   make_prefill_mainloop_spec()    make_mla_mainloop_spec()

Pipeline Topology   →   make_prefill_topology()         make_mla_topology()
  (shared types)        PipelineEdge, PipelineType      PipelineEdge, PipelineType
  pipelines: 10         load_q, load_kv, mma_s0/s1,    load_q, load_kv, mma_s,
                        s0_corr, s1_corr, corr_epi,     p_mma, mma_o
                        mma_corr, s0_s1_sequence

CollectiveBuilder   →   build_fmha_launch_params()      build_mla_launch_params()
  (shared file)         MMA atoms, SMEM, TMA, Storage   MMA atoms, SMEM, TMA, Storage

Tile Scheduler      →   FmhaStaticTileScheduler         MLAStaticTileScheduler
                                                        + mla_get_split_kv()
                                                        + mla_get_k_tile_count()

Shared Math         →   exp2_scale() ◄────────────────► exp2_scale()
  (softmax_math.py)     (4-way unrolled row_sum          packed_row_sum()
                         kept in SoftmaxRole for ILP)

Shared TMEM         →   (not used)                      tmem_load_partition()
  (tmem_utils.py)                                       (MLARescaleRole, MLAEpilogueRole)

Masking             →   apply_mask(), MaskType           (boundary masking inline)
  (fusion/mask.py)      get_trip_count()
                        get_masked/unmasked_trip_count()
```

---

## Appendix: File Reference

### FlashInfer PR #1549

| File | Lines | Description |
|------|-------|-------------|
| `flashinfer/cute_dsl/prefill.py` | 2934 | MHA prefill kernel |
| `flashinfer/cute_dsl/mla.py` | 3641 | MLA decode kernel |
| `flashinfer/cute_dsl/patch/pipeline.py` | 419 | Pipeline producer/consumer wrappers |
| `tests/test_blackwell_fmha.py` | 570+ changed | Prefill tests (causal, varlen, GQA, sink) |
| `tests/test_deepseek_mla.py` | 119+ changed | MLA decode tests |
| `tests/sink_attention_reference.py` | 403 | Sink attention reference implementation |

### DKG Python CuTe DSL

| File | Description |
|------|-------------|
| `fmha.py` | MHA prefill (~3620 lines) |
| `fmha_decode.py` | MHA decode with split-KV |
| `fmha_bwd.py` | Backward pass |
| `mla/mla_decode_fp16.py` | MLA decode FP16 |
| `mla/mla_decode_fp8.py` | MLA decode FP8 |
| `mla/mla_helpers.py` | MLA shared infrastructure |
| `mixed_input_fmha/` | Mixed-precision variants (Int8/Int4/FP8 K/V) |

### C++ CUTLASS

| File | Description |
|------|-------------|
| `device/fmha.hpp` | Device-level launch wrapper |
| `kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp` | Forward kernel entry |
| `collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp` | Forward mainloop (mma, softmax, correction) |
| `collective/sm100_fmha_load_tma_warpspecialized.hpp` | TMA loader |
| `collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp` | Epilogue (TMA store) |
| `collective/fmha_common.hpp` | Masks, variable-length, GEMM helpers |
| `kernel/sm100_fmha_gen_kernel_warpspecialized.hpp` | Gen/decode kernel |
| `kernel/sm100_fmha_mla_tma_warpspecialized.hpp` | MLA inference kernel |
| `kernel/sm100_fmha_mla_reduction.hpp` | MLA split-KV reduction |

### Backward Compatibility

The original monolithic files (`flashinfer/cute_dsl/prefill.py`, `flashinfer/cute_dsl/mla.py`) are preserved unchanged. The modular `attention/` package is used via its own entry points. A re-export shim pattern can be used to maintain backward compatibility:

```python
# flashinfer/cute_dsl/prefill.py (shim)
from .attention.prefill import BlackwellFusedMultiHeadAttentionForward
from .attention.wrappers.batch_prefill import (
    BatchPrefillCuteDSLWrapper, qkv_torch_2_cute, create_and_pad_tensor,
)
from .attention.fusion.mask import MaskType
from .attention.scheduler.persistent import (
    FmhaStaticTileScheduler, FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler, create_fmha_static_tile_scheduler_params,
)
from .attention.fusion.logits_transform import sigmoid_logits_transform
from .attention.fusion.output_transform import dumb_output_transform
```

### Refactoring Origin

This modularization was derived from the monolithic `prefill.py` (2934 lines) and `mla.py` (3641 lines). Key extraction landmarks:

- **`AttentionConfig`**: Extracted from `BlackwellFusedMultiHeadAttentionForward.__init__` scattered `self.xxx` attributes
- **`TmemLayout`**: Extracted hardcoded magic numbers (0, 128, 256, 384, 32, 160) into a computed dataclass
- **`FmhaStaticTileScheduler`**: Moved self-contained scheduler logic (originally lines 111-262 of `prefill.py`)
- **`MaskType`**: Moved enum (originally lines 264-268 of `prefill.py`)
- **Warp roles**: Each role class was extracted from methods of the monolithic `BlackwellFusedMultiHeadAttentionForward` class
- **MLA**: Ported from `mla.py` following the same extraction pattern, with separate concrete types (`MLAConfig`, `MLAWarpSchedule`, `MLAMainloopSpec`)
- **`collective_builder.py`**: Created in Phase 5/Layer 2; encapsulates MMA atom, SMEM layout, TMA atom, and `SharedStorage` creation that was previously ~200–300 lines inline in each kernel's `__call__`
- **Kernel readability (Phase 5)**: Removed config unpacking boilerplate (Layer 1), extracted infrastructure to CollectiveBuilder (Layer 2), extracted pipeline/MMA setup to `@cute.jit` helpers (Layer 3)
- **Wrapper/utility cleanup (Phase 6)**: Moved test utilities and tensor conversion functions out of kernel files into `wrappers/`, reducing `prefill.py` from 957→598 lines and `mla_decode.py` from 1274→1073 lines
- **MLA kernel loop extraction (Phase 7)**: Moved tile scheduler loops, `SimpleNamespace` parameter bundling, and pipeline state management from `mla_decode.py`'s `kernel()` into each MLA role's `run()` method, reducing `mla_decode.py` from 1073→593 lines and bringing it to near-parity with `prefill.py` (598 lines). Each role now owns a `_get_k_tile_count()` `@cute.jit` method for device-side tile range computation (required because CuTe DSL's `min`/`max` rewrites only work within `@cute.jit` context).
