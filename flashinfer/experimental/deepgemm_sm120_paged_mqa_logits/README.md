# SM120a paged FP8 MQA lightning-indexer logits

`from flashinfer.sm120_paged_mqa_logits import prepare_sm120_paged_mqa_logits`
prepares the DeepSeek-V3.2 / V4.1-Flash decode-side "lightning indexer"

```
logits[b * next_n + t, k] = sum_h relu(Q[b, t, h, :] . KV[k, :]) * scale[k] * w[b * next_n + t, h]
```

over a block-table KV cache on **SM120a** (RTX PRO 6000 Blackwell / RTX 5090,
GB202). Each prepared call returns a plan; `plan.run()` submits on the current
PyTorch stream and returns the logical `[B * next_n, max_context_len]` view of
the FP32 output.

A call is a real **two-kernel sequence**: a single-warp scheduler writes
`schedule_meta[num_sms + 1, 2]` (per-CTA `(q_atom_idx, kv_split_idx)` walk
boundaries over 128-row KV segments) and a persistent 384-thread register-MMA
program (`mma.sync.m16n8k32 e4m3·e4m3→f32` + `ldmatrix` + TMA + mbarriers; no
tcgen05/TMEM/WGMMA, which `ptxas` rejects on this architecture) walks the range
it finds there. Both kernels are delivered as **one** generated prepared-sequence
binding, so `plan.run()` is a single FFI submission with the per-stage
programmatic-dependent-launch attributes intact. `get_paged_mqa_logits_metadata`
submits the scheduler alone through its own standalone program and returns the
real schedule.

## Shipped programs and routes

Routes are selected from host-known scalars only — head count, page size and
`next_n` — and named `sm120:fp8:h<H>:p<page_kv>:n<next_n>`:

| heads | page_kv | next_n | model |
|---|---|---|---|
| 32 | 128, 64 | 1, 2, 4 | DeepSeek-V4.1-Flash decode |
| 64 | 64 | 1, 2, 4 | DeepSeek-V3.2 decode |

`route_available(num_heads, page_kv, next_n)` answers on the host; there is no
fallback for an unsupported configuration. The scheduler program is shared by
every route (route `sm120:metadata`).

The launch grid of the logits program **is** the CTA budget: the programs take
no compile-line definitions, so one delivered text per program serves every SM
count and every `sm_count` override.

## Operands

`prepare_sm120_paged_mqa_logits(q, kv_cache, weights, context_lens,
block_table, max_context_len, *, page_kv=None, schedule_meta=None, output=None,
sm_count=None)` binds

- `q` E4M3 `[B, next_n, H, 128]`, contiguous;
- `kv_cache` uint8 `[pages, page_kv, 1, 132]` — each page holds `page_kv`
  FP8 rows of 128 bytes followed by `page_kv` FP32 scales, read in place (the
  vLLM `indexer_k_store` layout). The token rows must be dense (`stride(3) ==
  1`, `stride(1) == 132`); the page stride `stride(0)` may exceed
  `page_kv * 132` — the strided per-layer view of a block-outermost engine
  layout (every layer's page in one block) and an alignment-padded page are
  read without a copy. Such an allocation may also be passed as its 2-D
  `[pages, block_stride_bytes]` view together with `page_kv=`. The block stride
  must be a multiple of 16 bytes; the TMA descriptor takes it from the tensor
  and the kernel never reads past the `page_kv * 132` bytes of a page;
- `weights` FP32 `[B * next_n, H]`, contiguous;
- `context_lens` int32 `[B, next_n]`, contiguous — the schedule is sized from
  each request's **last** token, every token masks with its own length;
- `block_table` int32 `[B, S]` with `stride(1) == 1` (any row stride, e.g. an
  engine's `[::next_n]` view);
- `1 <= max_context_len <= S * page_kv`.

The output is FP32 `[B * next_n, paged_logits_stride(max_context_len)]` with
`paged_logits_stride(n) = align(align(n, 128), 256)` (DeepGEMM's 1024-byte row
rule); consume `plan.logical_output` (`output[:, :max_context_len]`). Store
semantics are DeepGEMM's `clean_logits=False`: inside each request's computed
extent (`ceil(context_lens[b, -1] / 128) * 128` columns) the cells at or past a
token's own length carry unspecified values and the consumer masks by length;
columns beyond that extent are left untouched.

`output` and `schedule_meta` are allocated when not supplied and stay bound to
the plan. Operand values may change in place between runs, including CUDA Graph
replay; the schedule is rebuilt by every `run()`, so a caller-owned
`schedule_meta` never goes stale. Prepare a new plan when shapes or tensor
identities change, and do not use one plan's output concurrently on different
streams.

## DeepGEMM-signature entries

```python
from flashinfer.sm120_paged_mqa_logits import (
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)

meta = get_paged_mqa_logits_metadata(context_lens_2d, page_kv, num_sms)  # int32 [num_sms + 1, 2], real launch
logits = fp8_paged_mqa_logits(
    q, kv_cache, weights, context_lens_2d, block_table, meta, max_context_len
)  # FP32 [B * next_n, max_context_len]
```

These mirror the `deep_gemm` calls a DeepSeek-V3.2 serving engine makes, so a
caller switches backends without re-shaping its tensors. `schedule_meta` fixes
the CTA budget through its first dimension (`None` selects the device's SM
count) and is rebuilt in the same sequence before the logits program reads it,
so the one-shot entry is self-contained. `clean_logits=True` is rejected for
two-dimensional context lengths exactly as DeepGEMM does, and `indices`
(variable-length request selection) must be `None` — that scheduling branch is
not ported.

## Architecture targets

The generated sources are one architecture-neutral closure compiled as one
exact-architecture module per admitted target (`sm120a_nvcc_flags`). Admitted
capabilities follow FlashInfer's compilation context — `FLASHINFER_CUDA_ARCH_LIST`
when set, otherwise the visible devices — and `supported_capabilities()`
reports the intersection with the shipped set; nothing probes `nvcc` or a device
capability to pick targets. FlashInfer normalises capability 12.0 to the `f`
suffix on CUDA >= 12.9 and to `a` on 12.8, so either entry admits the family.
`sm_121a` (DGX Spark / GB10) has the same instruction surface but needs its own
cubin and its own measured denominator; it is not shipped yet.

## Layout

Generated CUDA and bindings live in
`csrc/experimental/deepgemm_sm120_paged_mqa_logits/generated` (one device source
per program, one binding for the scheduler, one prepared-sequence binding per
route). The runtime and the program/route catalog are in this directory; the
public API is `flashinfer/sm120_paged_mqa_logits.py`.

```bash
pytest tests/experimental/test_sm120_paged_mqa_logits.py -q
python benchmarks/bench_sm120_paged_mqa_logits.py
```

The test covers every shipped `(H, page_kv, next_n)` program against a
dequantized PyTorch reference and an independent Python mirror of the schedule,
the `clean_logits=False` write extent, the padded block-stride cache view, CUDA
Graph replay with changed contents, one-shot/plan equivalence and the
architecture-target gate. It skips on devices without catalogued programs.

## Catalog schema `sm120_paged_mqa.v1`

`sm120_paged_mqa_catalog.json` carries `arches`, `policy`, `programs` and
`routes`. Policy keys: `head_dim`, `fused_row_bytes` (132), `heads`, `page_kv`,
`next_n`, `split_kv` (128 for every exported program, which is why the metadata
entry keeps DeepGEMM's head-count-free signature), `next_n_atoms` (the Q-atom
rule: one atom per request for every shipped `next_n` — 1 and 2 pair the
tokens as DeepGEMM does, 4 scores the whole request from one atom where
DeepGEMM runs two 2-token atoms), `max_batch` (the scheduler's shared-memory request
ceiling), `logits_stride_alignment` (`[128, 256]`), `clean_logits` (`"raw"`),
`metadata_program`, `metadata_route`, `threads` and per-program `programs`
records (tile size, group count, KV stages, atoms, shared-memory bytes). A
program record carries `kind` (`"module"` or `"sequence"`), `role`, `sources`,
`standalone`, `compile_flags`, `ffi_entry`, `arg_plan` and the per-architecture
`closure_sha256`. Every route record has `stages` (`[[stage, program], ...]`),
`sequence` (the prepared-sequence binding, `null` for the metadata route),
`kernel_launches`, `num_heads`, `page_kv`, `next_n` and `clean_logits`.
