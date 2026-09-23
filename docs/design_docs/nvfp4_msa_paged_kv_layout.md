# NVFP4 paged-KV MSA cache layout contract (SM100 / SM103)

This note fixes the physical layout of the NVFP4 KV cache consumed by the
MiniMax Sparse Attention (MSA) decode and prefill routes in
`flashinfer.msa_ops`, and states how the vLLM producer, the `flashinfer.msa_ops`
reader, and the MiniMax `nv_dev` kernels relate to it. It is the reference for
`page_layout()` and the four strided views the route verifies before serving a
call.

## 1. Fixed geometry

| Symbol | Value | Meaning |
|---|---|---|
| `D` | 128 | head dimension |
| `P` | 128 | tokens per page (= sparse block size) |
| `G` | 16 | head-dim values per E4M3 block scale |
| `DATA_DIM` | `D / 2 = 64` | packed E2M1 bytes per token |
| `SCALE_DIM` | `D / G = 8` | E4M3 scale bytes per token |
| `TOPK` | 16 | MiniMax-M3 production selection width |
| `Hkv` | `max(1, 4 / TP)` | KV heads held by one tensor-parallel rank |

`D`, `P`, and `TOPK` are compile-time constants of the kernel bodies; `Hkv` is a
runtime argument, and every byte offset below is a function of it, not a
constant. A rank's page shrinks with its KV-head count.

## 2. Element encoding

* **Data**: E2M1, two values per byte, even element in the low nibble
  (`byte = code[2i] | code[2i+1] << 4`). Code bits are `sign<<3 | mag`, with
  magnitudes `0, 0.5, 1, 1.5, 2, 3, 4, 6`.
* **Block scale**: one positive E4M3 byte per 16 consecutive head-dim values of
  one token. In vLLM's `nvfp4` mode, `sf = e4m3(amax16 / (6 * global_scale))`
  and `q = e2m1(x / (float(sf) * global_scale))`. The `nvfp4_4over6` mode differs
  only in the append-time scale search; the reader ABI is identical.
* **Global scales**: explicit, per side, positive floats
  (`k_global_scale`, `v_global_scale`). Dequant is
  `x = e2m1 * float(sf) * global_scale`. A reader folds `k_global_scale` into the
  softmax scale and applies `v_global_scale` in the output epilogue; neither
  touches the stored bytes.

## 3. Canonical page map (bytes)

One page holds one side after the other, and inside a side the packed data of
all `Hkv` heads precedes the block scales of all `Hkv` heads:

```
offset                                   region              shape (row-major)
0                                        K data              [Hkv, P, DATA_DIM]
Hkv * P * DATA_DIM                       K scale (linear)    [Hkv, P, SCALE_DIM]
Hkv * P * (DATA_DIM + SCALE_DIM)         V data              [Hkv, P, DATA_DIM]
Hkv * P * (2*DATA_DIM + SCALE_DIM)       V scale (swizzled)  [Hkv, P, SCALE_DIM]
page_bytes = 2 * Hkv * P * (DATA_DIM + SCALE_DIM)            (= 73,728 for Hkv = 4)
```

This is exactly what `flashinfer.msa_ops` `page_layout(num_kv_heads)` returns.

### 3.1 V-scale swizzle

K scales are token-major linear: `k_scale[h, t, s]`. V scales use the vLLM SM100
consumer swizzle so that the block-scale writer and the MSA readers share one
layout. For logical `(t, s)` with `t in [0,128)`, `s in [0,8)`:

```
swz_t = (t // 4) * 4 + s // 2
swz_s = (s % 2) * 4 + t % 4
v_scale_swizzled[h, swz_t, swz_s] = v_scale_linear[h, t, s]
```

Equivalently: view `[T/4, 4, 4, S/4]`, permute `(0, 2, 3, 1)`, reshape `[T, S]`.
The route's `_v_scale_unswizzle_index` inverts exactly this map.

## 4. Public views of one allocation

All SM100/SM103 readers take four strided `uint8` (or E4M3-reinterpreted) views
of the same allocation, never a repacked copy:

| view | shape | strides (bytes) | storage offset |
|---|---|---|---|
| `k` | `[pages, Hkv, P, 64]` | `(page_bytes, P*64, 64, 1)` | 0 |
| `k_scale` | `[pages, Hkv, P, 8]` | `(page_bytes, P*8, 8, 1)` | `Hkv*P*64` |
| `v` | `[pages, Hkv, P, 64]` | `(page_bytes, P*64, 64, 1)` | `Hkv*P*72` |
| `v_scale` | `[pages, Hkv, P, 8]` | `(page_bytes, P*8, 8, 1)` | `Hkv*P*72 + Hkv*P*64` |

Whole-tensor contiguity is not required; readers need unit inner stride and
16-byte-aligned base pointers only. The decode route additionally verifies the
byte offsets between the four base pointers before serving a call: shape, dtype,
and stride alone cannot distinguish four views of one planar page from four
unrelated allocations, and the byte offsets are what pin the inputs to the page
map the cache writer used.

### 4.1 How each producer/consumer spells the same bytes

* **vLLM** allocates `[num_blocks, 2, Hkv, P, 72]` and derives the four views
  above from that tensor's strides (`data_dim = full_dim * 8 // 9`). The
  `72`-wide last axis is only an allocation shape: inside one side the bytes are
  `[data of all heads | scales of all heads]`, exactly Section 3.
* **`flashinfer.msa_ops`** describes the page as a flat byte pool via
  `page_layout()` and verifies the four byte offsets (Section 4).

These are byte-identical; the `[pages, 2, Hkv, P, 72]` form is an allocation
alias of Section 3.

### 4.2 MiniMax `nv_dev` adapter

The MiniMax `q8kv4` decode kernels take four separate contiguous tensors with
**linear** V scales and a token-major selection, so they do not read the vLLM
cache zero-copy:

| aspect | canonical (Sections 3-4) | MiniMax `q8kv4` | adapter |
|---|---|---|---|
| K/V data | strided views, page stride `page_bytes` | contiguous, page stride `Hkv*P*64` | repack unless the kernel accepts a page stride |
| V scale | (4,4) swizzled | linear | unswizzle, or a swizzle-aware reader |
| top-k | `[Hkv, total_q, TOPK]` head-major, ascending, `-1` tail | `[total_q, Hkv, TOPK]` token-major, local page last | `permute(1,0,2)`, contiguous |
| Q dtype | BF16 | E4M3 only | quantize Q, or use the FP8-compute path |
| global scales | explicit floats | none | fold `k` into `sm_scale`, apply `v` to the output |

## 5. Selection and position contract

* `q2k_indices`: `int32 [Hkv, total_q, TOPK]`, logical block ids into the
  request's `page_table` row, ascending, `-1` tail-padded, one row per
  `(kv_head, query token)`. Each query token keeps its own selection; adjacent
  tokens' rows are never unioned.
* `page_table`: `int32 [batch, max_blocks]` logical to physical page; unused
  slots may hold `-1` and are never dereferenced.
* `seqused_k`: `int32 [batch]`, total KV tokens including the current
  decode/MTP query tokens. Query token `i` of a request with uniform `seqlen_q`
  sits at position `seqused_k - seqlen_q + i` (right-aligned).
* Causality masks every key position `> q_position`; the final page is read only
  up to `seqused_k`. Readers must mask page tails by `seqused_k` and must not
  rely on zero-filled tail tokens: prefix caching and page reuse leave stale
  bytes there.
* Fully masked rows return exact zeros (and `-inf` LSE where exposed).

## 6. Validation checklist for any reader claiming this contract

1. Random pool bytes everywhere, including tail tokens past `seqused_k`.
2. Permuted page tables and pages shared between requests.
3. Contexts shorter than `TOPK` pages (`-1` tail) and page tails (`kv % 128 != 0`).
4. Non-unit `k_global_scale` / `v_global_scale`.
5. Compact and non-contiguous views (strided K/V, E4M3 vs `uint8` scale views).
6. The exact V-scale swizzle vector (Section 3.1).
7. TP1/TP2/TP4/TP8 head geometries; `seqlen_q` in `[1, 8]`.
8. An FP32 reference computed from the pool bytes (not from the unquantized
   source), scored per output row with a scale-free relative-L2 bound: an
   absolute tolerance is blind here because a softmax-weighted mean of V has
   magnitude `~|V|/sqrt(N)`. The `flashinfer.msa_ops` NVFP4 decode tests use
   unit-RMS inputs, an FP32-from-bytes reference, and a relative-Frobenius
   bound with a cosine floor for exactly this reason.

## 7. Generated Cake decode program

`flashinfer.msa_ops.prepare_msa_nvfp4_sparse_decode` (experimental, Cake
backend under `flashinfer/experimental/msa_nvfp4_decode/`) is a second reader
of exactly this contract on compute capability 10.0/10.3: it consumes the four
strided views of section 4 in place, folds `k_global_scale` into the softmax
scale, applies `v_global_scale` in its epilogue, and takes the head-major
top-k selection of section 5. It adds no layout, encoding or selection rule of
its own; `tests/experimental/test_cake_msa_nvfp4_decode.py` checks it against
the FP32 oracle of the existing route and against that route on the same
pages.
