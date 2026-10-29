# Experimental NVFP4 paged-KV MSA decode (Cake backend)

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection; `flashinfer.msa_ops.msa_sparse_decode_attention` keeps
serving the NVFP4 paged-KV route it has today.

`flashinfer.msa_ops.prepare_msa_nvfp4_sparse_decode(...)` serves the MiniMax
Sparse Attention decode step over the planar NVFP4 page pool fixed by
[`docs/design_docs/nvfp4_msa_paged_kv_layout.md`](../../../docs/design_docs/nvfp4_msa_paged_kv_layout.md)
(E2M1 data, E4M3 block scales, linear K scales, swizzled V scales, 128-token
pages, top-k 16) on compute capability 10.0, 10.3 and 10.7 with three kinds of
generated programs:

- **Persistent program** (`route == "persistent"`). Each work item is one
  `(query token, KV head)` pair with its sixteen selected pages. A producer
  warp streams the packed K/V pages and their block scales with two TMA
  requests per page side; two transform warpgroups dequantize them to BF16 in
  shared memory; the K tiles stay resident in tensor memory and both MMAs run
  in the swapped `S^T` / `O^T` orientation so the sixteen query heads of a KV
  head form the MMA N dimension. Softmax runs in the exponent domain with a
  lazily moved origin; the epilogue is lane-owned per head. Small batches
  split the page pairs of every item across 2, 4 or 8 CTAs
  (`cake_backend.split_factor`: one CTA per item whenever the items fill at
  least half of the resident CTAs, otherwise the largest power of two that
  still fits one wave; items with fewer than four page pairs never split).
  The CTAs of a split item launch as one cluster and merge their FP32
  partials through distributed shared memory; no workspace is involved.
  On parts that register a last-round-split program (`tail`), a batch of more
  items than resident CTAs runs every full round unsplit and only the
  remainder items of the last round as cluster units (`cake_backend.tail_plan`).
- **Short-item program** (`route == "short"`). Batches whose requests all span
  at most four selected pages (`page_table` with at most four columns, e.g.
  the 257-token tail of a decode step) run one eight-CTA cluster of
  register-MMA CTAs per work item, two CTAs per selected page, the FP32
  partials merged through distributed shared memory.
- **Request-packed union program** (`route == "union"`, selected by
  `use_cache_hit=True`, compute capability 10.0 and 10.3). Each work unit is
  one `(request, KV head)` pair: the `seqlen_q` query tokens of the request
  share one CTA, the producer merges their top-k selections into one sorted
  union list on device, and every page of that union streams once (TMA, NVFP4
  transform, K in tensor memory) while per-token masks select the MMA
  columns, so a page selected by several tokens is loaded once and a token
  never attends a page it did not select. The program is traced for a
  query-group width of 2 (`seqlen_q <= 2`) or 4 (`seqlen_q` 3 or 4; three
  tokens run with an empty phantom token). Small unit counts split each unit
  across a cluster of 2, 4 or 8 CTAs that merge their partials through
  distributed shared memory (`cake_backend.union_split_factor`: one CTA per
  unit whenever the units alone fill more than half of the SMs, otherwise the
  largest power of two at most 8 whose `units x S` CTAs fit one wave and
  leave every CTA at least one page pair -- 16 units on 148 SMs split
  eight ways, 32 units four ways, 64 units two ways); the cluster launch is
  one cluster per unit, the unsplit launch is persistent over
  `min(units, SMs)` CTAs (`cake_backend.union_plan`).
- `seqlen_q` in `[1, 32]` query tokens per request (speculative / MTP verify)
  attend causally; decode tokens are right-aligned at `seq_len - seqlen_q + i`.
- Preparation binds the tensors and decides the program and the split factor;
  the runner launches with no allocation and no host synchronization, reads
  every buffer on device, and can be captured into a CUDA Graph.

### `use_cache_hit`

`prepare_msa_nvfp4_sparse_decode(..., use_cache_hit=...)` accepts exactly
`False`, `True` or `"auto"` (the default); `1`, `"true"` and similar
spellings raise `ValueError`. Every value computes the same attention: each
token attends only its own sixteen selected pages. The switch selects the
program that serves the `seqlen_q` tokens of a request.

| Value | Program | Notes |
| --- | --- | --- |
| `False` | persistent or short-item program (per `(token, KV head)` work item) | the routes above, unchanged |
| `True` | request-packed union program | requires a uniform `seqlen_q` in `[2, 4]`; `ValueError` for a larger `seqlen_q`; at `seqlen_q == 1` the switch is a no-op and the `False` programs run |
| `"auto"` | the `False` programs | currently equal to `False` on every shape; a measured device-side policy that is never slower than `False` is future work |

`True` under CUDA-graph capture requires the selected union program to have
been built before the capture starts (prepare the same shape once eagerly
first); preparing a not-yet-built union program inside a capture raises
`ValueError`. A part that registers no union program (compute capability 10.7
in this checkout) raises `NotImplementedError` for `True` inside the domain.
The union program is a separate kernel from the per-token programs, so its
output is not bitwise identical to theirs; both are held to the same
tolerance against the FP32 oracle (`atol = rtol = 1e-2`, per-row relative L2
error at most 0.004).

| Tensor | Shape | dtype |
| --- | --- | --- |
| `q` | `[batch * seqlen_q, num_q_heads, 128]` | bfloat16 |
| `k`, `v` | `[num_pages, num_kv_heads, 128, 64]` strided views of the page pool | uint8 |
| `k_scale`, `v_scale` | `[num_pages, num_kv_heads, 128, 8]` strided views of the page pool | uint8 or float8_e4m3fn |
| `q2k_indices` | `[num_kv_heads, batch * seqlen_q, 16]` ascending, `-1` padded | int32 |
| `page_table` | `[batch, max_pages]` | int32 |
| `seqused_k` | `[batch]` KV tokens per request | int32 |
| `out` | `[batch * seqlen_q, num_q_heads, 128]` | bfloat16 |
| `lse` | `[batch * seqlen_q, num_q_heads]` natural-log softmax normalizer | float32 |

`k_global_scale` is folded into the softmax scale and `v_global_scale` is
applied in the epilogue, exactly as the layout contract states; neither touches
the stored bytes. Query head `h` attends to KV head `h // (num_q_heads //
num_kv_heads)` with at most sixteen query heads per KV head.

```python
from flashinfer.msa_ops import prepare_msa_nvfp4_sparse_decode

# q, k, v, k_scale, v_scale, q2k_indices, page_table, seqused_k come from the
# serving stack (vLLM planar NVFP4 pages + msa_topk_select).
decode = prepare_msa_nvfp4_sparse_decode(
    q, k, v, q2k_indices,
    k_scale=k_scale, v_scale=v_scale, page_table=page_table, seqused_k=seqused_k,
    k_global_scale=0.75, v_global_scale=0.85,
)
out = decode()          # launches on the current stream, returns out
q.copy_(next_q); q2k_indices.copy_(next_selection)
out = decode()          # same runner (or a captured graph), new step

# Speculative verify: four drafted tokens per request served by the union program.
verify = prepare_msa_nvfp4_sparse_decode(
    q4, k, v, q2k4, seqlen_q=4, use_cache_hit=True,
    k_scale=k_scale, v_scale=v_scale, page_table=page_table, seqused_k=seqused_k,
    k_global_scale=0.75, v_global_scale=0.85,
)
assert verify.route == "union"
```

## Generated sources

`csrc/cake_msa_nvfp4_decode/` holds one device source per program and one host
launcher per route, selected by the record's compile flags and shared by every
architecture the program compiles for, plus the family's device preamble
(`*_device_common.cuh`). The device helpers that the preamble includes live in
the repository's shared `csrc/cake_device_helpers/`; the host launchers include
the shared host preamble
`csrc/blackwell_msa/cake_blackwell_msa_host_common.cuh`.
`cake_jit.py` lists the programs once (`PROGRAMS`: architectures, route,
split factor, last-round split or union query-group width and cluster flag,
sources) together with each route's argument order (`ARG_PLANS`), and
compiles a program with the exact flag set of the device it runs on;
`cake_backend.py` validates the inputs against the kernel's contract, selects
the program and the split factor, and binds the launch.
The tests in `tests/experimental/test_cake_msa_nvfp4_decode.py` compare the
programs with the FP32 oracle of the existing NVFP4 route and with that route
itself, check that a token's union output does not depend on its siblings'
selections, and replay captured graphs; `tests/test_helpers/cake_msa_nvfp4_inputs.py`
builds the inputs (independent or overlap-controlled selections, shuffled,
identity or shared-prefix page tables); `benchmarks/bench_cake_msa_nvfp4_decode.py`
times the programs against `msa_sparse_decode_attention` on the same tensors
with CUPTI (`--use-cache-hit false true auto` selects the arms).
