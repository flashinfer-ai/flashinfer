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
pages, top-k 16) on compute capability 10.0 and 10.3 with one generated
persistent kernel:

- Each work item is one `(query token, KV head)` pair with its sixteen selected
  pages. A producer warp streams the packed K/V pages and their block scales
  with two TMA requests per page side; two transform warpgroups dequantize
  them to BF16 in shared memory; the K tiles stay resident in tensor memory and
  both MMAs run in the swapped `S^T` / `O^T` orientation so the sixteen query
  heads of a KV head form the MMA N dimension. Softmax runs in the exponent
  domain with a lazily moved origin; the epilogue is lane-owned per head.
- `seqlen_q` in `[1, 32]` query tokens per request (speculative / MTP verify)
  attend causally; decode tokens are right-aligned at
  `seq_len - seqlen_q + i`.
- Small batches split the page pairs of every item across 2, 4 or 8 CTAs
  (`cake_backend.split_factor`: one CTA per item whenever the items fill at
  least half of the resident CTAs, otherwise the largest power of two that
  still fits one wave; items with fewer than four page pairs never split).
  The FP32 partials and the self-resetting completion counters live in a
  caller-owned workspace.
- Preparation binds the tensors and decides the split factor; the runner
  launches with no allocation and no host synchronization, reads every buffer
  on device, and can be captured into a CUDA Graph.

| Tensor | Shape | dtype |
| --- | --- | --- |
| `q` | `[batch * seqlen_q, num_q_heads, 128]` | bfloat16 |
| `k`, `v` | `[num_pages, num_kv_heads, 128, 64]` strided views of the page pool | uint8 |
| `k_scale`, `v_scale` | `[num_pages, num_kv_heads, 128, 8]` strided views of the page pool | uint8 or float8_e4m3fn |
| `q2k_indices` | `[num_kv_heads, batch * seqlen_q, 16]` ascending, `-1` padded | int32 |
| `page_table` | `[batch, max_pages]` | int32 |
| `seqused_k` | `[batch]` KV tokens per request | int32 |
| `workspace_buffer` | `>= msa_nvfp4_decode_workspace_size(batch, num_kv_heads, device, seqlen_q=...)` bytes | uint8 |
| `out` | `[batch * seqlen_q, num_q_heads, 128]` | bfloat16 |
| `lse` | `[batch * seqlen_q, num_q_heads]` natural-log softmax normalizer | float32 |

`k_global_scale` is folded into the softmax scale and `v_global_scale` is
applied in the epilogue, exactly as the layout contract states; neither touches
the stored bytes. Query head `h` attends to KV head `h // (num_q_heads //
num_kv_heads)` with at most sixteen query heads per KV head.

```python
import torch
from flashinfer.msa_ops import prepare_msa_nvfp4_sparse_decode
from flashinfer.experimental.msa_nvfp4_decode.cake_backend import (
    msa_nvfp4_decode_workspace_size,
)

# q, k, v, k_scale, v_scale, q2k_indices, page_table, seqused_k come from the
# serving stack (vLLM planar NVFP4 pages + msa_topk_select).
workspace = torch.empty(
    msa_nvfp4_decode_workspace_size(batch, num_kv_heads, q.device),
    dtype=torch.uint8, device=q.device,
)
decode = prepare_msa_nvfp4_sparse_decode(
    q, k, v, q2k_indices,
    k_scale=k_scale, v_scale=v_scale, page_table=page_table, seqused_k=seqused_k,
    k_global_scale=0.75, v_global_scale=0.85, workspace_buffer=workspace,
)
out = decode()          # launches on the current stream, returns out
q.copy_(next_q); q2k_indices.copy_(next_selection)
out = decode()          # same runner (or a captured graph), new step
```

`cake_jit.py` registers the generated translation units per architecture and
split factor (`MODULES`, populated by the generated-program export) and builds
them with FlashInfer's JIT; `cake_backend.py` validates the inputs against the
kernel's contract, mirrors the split rule and carves the workspace. The tests
in `tests/experimental/test_cake_msa_nvfp4_decode.py` compare the program with
the FP32 oracle of the existing NVFP4 route and with that route itself, and
replay a captured graph; `benchmarks/bench_cake_msa_nvfp4_decode.py` times it
against `msa_sparse_decode_attention` on the same tensors with CUPTI.
