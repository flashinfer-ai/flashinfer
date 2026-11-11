# Experimental DSA indexer top-k (LiteTopK)

`flashinfer.dsa_indexer.dsa_indexer_topk` is an experimental API. Calling it
opts in and emits FlashInfer's experimental warning once; it is not selected
automatically. The API may change or be removed without deprecation.

This implementation ports the FP8 LiteTopK pipeline from
[vLLM #65920](https://github.com/vllm-project/vllm/pull/65920), commit
`833266447780a74713c574e67f3b3bf9caa868eb`, into FlashInfer's JIT and TVM-FFI.
For each query, it computes the indexer score
`sum_h weights[q,h] * relu(dot(q[q,h], kv[key])) * kv_scales[key]` and selects
keys in that row's visible range:

1. Score a strided sample and predict a threshold using distances to a row reference.
2. Score every visible key and compact candidates that meet the threshold.
3. Select top-k from the FP32 candidates.
4. Recover rows whose candidate count overflowed or fell short by rescoring and
   selecting on the GPU using a bounded scratch pool.

The common path does not materialize `[num_q, seq_kv]` logits. Recovery is part
of the same API call and needs no host readback. Queries run in windows of at
most 4096 rows; candidate storage is `8 * min(num_q, 4096) * cand_cap` bytes.
The workspace also includes sample codes, counters, row state, and a recovery
pool of at most 64 MiB. Query/key tensors and output storage are additional.

## Usage

```python
import torch
from flashinfer.dsa_indexer import dsa_indexer_topk, dsa_indexer_topk_workspace

rows, heads, keys, top_k = 16, 32, 65536, 2048
q = torch.randn(rows, heads, 128, device="cuda").to(torch.float8_e4m3fn)
kv = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
kv_scales = torch.ones(keys, device="cuda")
weights = torch.randn(rows, heads, device="cuda") / heads
cu_end = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
indices = dsa_indexer_topk(q, kv, kv_scales, weights, cu_end, top_k)

# Optional storage reuse; warm the operation before CUDA graph capture.
cu_start = torch.zeros(rows, device="cuda", dtype=torch.int32)
workspace = dsa_indexer_topk_workspace(q.device, rows, heads, keys)
out = torch.empty(rows, top_k, device="cuda", dtype=torch.int32)
dsa_indexer_topk(
    q, kv, kv_scales, weights, cu_end, top_k,
    cu_start=cu_start, out=out, workspace=workspace,
)
```

For packed requests, pass `cu_start[num_q]` and `cu_end[num_q]` as the
inclusive start and exclusive end of each query's visible key range. They are
clamped to `[0, seq_kv]` on the GPU. Returned indices are **relative to the
clamped start**, unordered, and padded with `-1` when fewer than `top_k` keys
are visible. An end before its start denotes an empty row. Convert valid
indices to absolute KV positions by adding the clamped start; preserve `-1`
entries when doing so.

The workspace helper returns six tensors: block counters, sample codes,
candidate scores, candidate indices, row state, and recovery scores. The
counters start at zero and are restored to zero by each call. A workspace can
be reused for smaller query/key counts with the same head count and candidate
capacity. **Do not share it between concurrently executing calls or graphs.**
Keep workspace and captured input/output tensors alive for graph replay. Change
input contents in place between replays; changing addresses requires recapture.

## Contract and limitations

- SM100-family GPUs, CUDA toolkit 12.9 or newer, and installed `deep_gemm` SM100
  headers are required. CUDA code is JIT-only and uses C++20. With nvcc older
  than 13.5, compilation disables the new NVVM rematerialization pass to avoid
  producer register spills.
  The port was validated with DeepGEMM headers at
  `82ca617e5879d0efb0aba5b91b1c81852588af6f` (the source vLLM revision's pin).
- `q` is FP8 e4m3 `[num_q, H, 128]`, `H` is 32 or 64; `kv` is FP8 e4m3
  `[seq_kv, 128]`; `kv_scales` is FP32 `[seq_kv]`; `weights` is FP32
  `[num_q, H]` and incorporates any query or attention scaling.
- `top_k` is 512, 1024, or 2048. `1 <= seq_kv <= 2**20`; an empty query batch
  returns an empty output. `cand_cap` defaults to 12288 and must be a multiple
  of four in `[4096, 12288]`. Reducing it can increase recovery frequency.
- Inputs must be finite, with finite computed scores. All tensors are contiguous
  and on the same CUDA device. `q`, `kv`, scales, weights, output, sample codes,
  and candidate slabs require 16-byte aligned addresses. Shapes and alignment
  are checked without reading device tensor contents back to the host.
- Selection is exact for the kernel's FP32 scores, up to ties. The scoring
  reduction uses a linear-term decomposition and is not bitwise identical to
  a direct FP32 or DeepGEMM reduction. Boundary ties can yield different indices
  and output order across runs.
- Poor samples or extreme score distributions can trigger expensive recovery.
  A score span exceeding 65504 or a very narrow tail can make recovery frequent;
  correctness does not depend on the sample predicting a good threshold.

## Migration from the earlier PR API

The earlier prefix-based API returned `(indices, status)`. This revision
removes `prefix_logits` and `status`: sampling happens inside the kernel, and
GPU recovery completes invalid candidate rows before returning the indices
tensor. No dense prefix scoring or KV reordering is needed. Calls now use
`dsa_indexer_topk(q, kv, kv_scales, weights, cu_end, top_k=2048)` with optional
`cu_start`, `out`, `workspace`, and `cand_cap` keyword arguments. The former
49152-candidate default is replaced by 12288 FP32 score/index pairs per row.

## Validation and benchmarking

```bash
pytest tests/experimental/test_dsa_indexer.py
python benchmarks/bench_dsa_indexer_topk.py --num-q 8192 --heads 32 --top-k 2048 --seq-kv 262144 1048576
python benchmarks/bench_dsa_indexer_topk.py --num-q 8192 --heads 64 --top-k 512 --seq-kv 262144
```

Tests compare selected scores with an independent FP32 reference, including
ragged ranges, short/empty rows, ties, adversarial sampling that forces recovery,
partial row windows, reusable workspace, and CUDA graph replay with changed
inputs. They also check bounds, uniqueness, padding, and invalid-input handling.

The benchmark compares the whole call against DeepGEMM FP8 MQA logits plus
`flashinfer.top_k_ragged_transform` on synthetic causal-prefill inputs. It
supports the older `fp8_mqa_logits` API and the newer `fp8_fp4_mqa_logits` API
with packed unit MX scales. Ragged selection ignores uninitialized logits
outside each query's visible range. It prints the GPU,
problem sizes, absolute latency, and memory. Timing uses cold L2 and the span
from the first GPU activity's start to the last activity's end via CUPTI,
falling back to CUDA events. Compilation is excluded; host launch gaps can
affect the measured span. Fused timing reuses workspace/output and includes producer,
selector, and recovery. Its memory column includes retained workspace/output
and extra peak allocations; dense memory includes logits and top-k scratch.
No measurements from the earlier prefix-based implementation are reused here.

## Attribution

The sample/filter/select design originates in
[LiteTopK](https://arxiv.org/abs/2607.11976). The kernel lineage and Apache-2.0
notices from vLLM and [SGLang #47469](https://github.com/sgl-project/sglang/pull/47469)
are retained in the CUDA sources. The tensor-core scoring pipeline uses
DeepGEMM's MIT-licensed SM100 helpers; its license is retained in
`csrc/LICENSE.deepseek-deepgemm`.
