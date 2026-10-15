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

Routes are selected from host-known scalars only — head count, page size,
`next_n` and the batch `B = context_lens.shape[0]` — and named
`sm120:fp8:h<H>:p<page_kv>:n<next_n>:a<atom>`, where `a<atom>` is the Q-atom
variant (tokens per Q atom) the catalog's `atom_rule` selects from
`(H, next_n, B)`:

| heads | page_kv | next_n | model |
|---|---|---|---|
| 32 | 128, 64 | 1, 2, 3, 4, 5, 6 | DeepSeek-V4.1-Flash decode |
| 64 | 64 | 1, 2, 3, 4, 5, 6 | DeepSeek-V3.2 decode |

Q-atom variant per batch class (`atom_rule.batch_class_min = [1, 2, 9, 128]`:
`B = 1`, `2 <= B <= 8`, `9 <= B < 128`, `B >= 128`; a request is scored from
`ceil(next_n / atom)` atoms, each atom streaming the KV once):

| next_n | heads | `B = 1` | `2 <= B <= 8` | `9 <= B < 128` | `B >= 128` |
|---|---|---|---|---|---|
| 1 | 32, 64 | `a1` (1 atom) | `a1` | `a1` | `a1` |
| 2 | 32, 64 | `a2` (1 atom) | `a2` | `a2` | `a2` |
| 3 | 32, 64 | `a3` (1 atom) | `a2` (2 atoms) | `a3` (1 atom) | `a3` |
| 4 | 32, 64 | `a2` (2 atoms) | `a2` (2 atoms) | `a4` (1 atom) | `a4` |
| 5, 6 | 32 | `a2` (3 atoms) | `a2` (3 atoms) | `a5` / `a6` (1 atom) | `a5` / `a6` |
| 5, 6 | 64 | `a2` (3 atoms) | `a2` (3 atoms) | `a2` (3 atoms) | `a3` (2 atoms) |

Small batches keep DeepGEMM's 2-token atomization (work-item balance across
the SMs), except a single 3-token request, whose one native atom streams the
KV once with nothing left to balance; the 64-head 5-/6-token atom exceeds the
256-row TMA Q box, so those geometries run two 3-token atoms from batch 128 up.
The table is data in the catalog, derived by the exporter from the kernel's
own rule.

`route_available(num_heads, page_kv, next_n[, batch])` answers on the host
(without `batch`: whether any variant of the geometry ships); there is no
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

meta = get_paged_mqa_logits_metadata(
    context_lens_2d, page_kv, num_sms, num_heads=q.shape[2]
)  # int32 [num_sms + 1, 2], real launch
logits = fp8_paged_mqa_logits(
    q, kv_cache, weights, context_lens_2d, block_table, meta, max_context_len
)  # FP32 [B * next_n, max_context_len]
```

These mirror the `deep_gemm` calls a DeepSeek-V3.2 serving engine makes, so a
caller switches backends without re-shaping its tensors. `schedule_meta` fixes
the CTA budget through its first dimension (`None` selects the device's SM
count) and is rebuilt in the same sequence before the logits program reads it,
so the one-shot entry is self-contained. `num_heads` is optional on the
metadata entry: the schedule's item count is the Q-atom count of
`(num_heads, next_n, B)`, and the head count is required only where the two
exported head counts select different schedules — `next_n` 5 and 6 at
`B > 8` — in which case the `ValueError` names it; for `next_n` 1–4 and for
every `B <= 8` the DeepGEMM-shaped call without `num_heads` is exact.
`clean_logits=True` is rejected for two-dimensional context lengths exactly as
DeepGEMM does, and `indices` (variable-length request selection) must be
`None` — that scheduling branch is not ported.

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

## Measured against DeepGEMM (2026-10-15, Cake `32a558c2`)

Cake / DeepGEMM kernel time (`bench_gpu_time`, CUPTI, cold L2, interleaved pairs; DeepGEMM `yyihuang/DeepGEMM@79a4ef3b` on the same device and clocks), geometric mean over the 12 (batch, context) rows of each program ((batch, average context) = (1, 16K), (8, 4K), (8, 16K), (8, 64K), (32, 16K), (32, 64K), (64, 4K), (64, 16K), (128, 16K), (128, 64K), (256, 4K), (256, 16K); 216 rows per device). `faster` counts rows with ratio < 1 and a bootstrap 95 % CI upper bound <= 1; `tie` counts rows within 0.5 % whose CI contains 1 (all of them on the DeepGEMM-identical schedules). Logits are bitwise-identical to DeepGEMM on every row of both devices.

| program | RTX 5090 Cake/DG | RTX 5090 faster / tie / slower | RTX PRO 6000 Cake/DG | RTX PRO 6000 faster / tie / slower |
|---|---:|---:|---:|---:|
| `h32p128n1` | 0.995 | 9 / 3 / 0 | 0.997 | 10 / 0 / 2 |
| `h32p128n2` | 0.991 | 11 / 0 / 1 | 0.995 | 11 / 1 / 0 |
| `h32p128n3` | 0.726 | 12 / 0 / 0 | 0.775 | 12 / 0 / 0 |
| `h32p128n4` | 0.745 | 11 / 0 / 1 | 0.781 | 12 / 0 / 0 |
| `h32p128n5` | 0.649 | 12 / 0 / 0 | 0.712 | 12 / 0 / 0 |
| `h32p128n6` | 0.688 | 12 / 0 / 0 | 0.713 | 12 / 0 / 0 |
| `h32p64n1` | 0.996 | 9 / 2 / 1 | 0.998 | 11 / 1 / 0 |
| `h32p64n2` | 0.991 | 10 / 0 / 2 | 0.993 | 12 / 0 / 0 |
| `h32p64n3` | 0.729 | 12 / 0 / 0 | 0.774 | 12 / 0 / 0 |
| `h32p64n4` | 0.753 | 11 / 0 / 1 | 0.783 | 12 / 0 / 0 |
| `h32p64n5` | 0.646 | 12 / 0 / 0 | 0.697 | 12 / 0 / 0 |
| `h32p64n6` | 0.686 | 12 / 0 / 0 | 0.707 | 12 / 0 / 0 |
| `h64p64n1` | 0.988 | 11 / 1 / 0 | 0.994 | 11 / 0 / 1 |
| `h64p64n2` | 0.993 | 10 / 1 / 1 | 0.995 | 7 / 3 / 2 |
| `h64p64n3` | 0.763 | 12 / 0 / 0 | 0.763 | 12 / 0 / 0 |
| `h64p64n4` | 0.921 | 11 / 0 / 1 | 0.801 | 10 / 1 / 1 |
| `h64p64n5` | 0.888 | 12 / 0 / 0 | 0.889 | 12 / 0 / 0 |
| `h64p64n6` | 0.949 | 10 / 0 / 2 | 0.919 | 8 / 0 / 4 |
| **all 18 programs** | 0.828 | 199 / 7 / 10 | 0.841 | 200 / 6 / 10 |

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

The test covers every shipped `(heads, page_kv, next_n, atom)` program — both
variants of a geometry through a small-batch and a large-batch case — against
a dequantized PyTorch reference and an independent Python mirror of the
schedule, the `clean_logits=False` write extent, the padded block-stride cache
view, CUDA Graph replay with changed contents, one-shot/plan equivalence, the
head-count rule of the metadata entry and the architecture-target gate. It
skips on devices without catalogued programs.

## Catalog schema `sm120_paged_mqa.v1`

`sm120_paged_mqa_catalog.json` carries `arches`, `policy`, `programs` and
`routes`. Policy keys: `head_dim`, `fused_row_bytes` (132), `heads`, `page_kv`,
`next_n`, `split_kv` (128 for every exported program, so the metadata entry
needs no page size), `atom_rule` (the Q-atom selection rule as data:
`batch_class_min` — the first batch of every batch class, the last class open
up to `max_batch` —, `selection["h<H>:n<N>"]` — the variant `a<atom>` per
class — and `num_atoms["n<N>:a<A>"]` — the schedule's `num_next_n_atoms` per
variant; the runtime evaluates it with `atom_variant` / `next_n_atoms` and
never needs the producer), `max_batch` (the scheduler's shared-memory request
ceiling), `logits_stride_alignment` (`[128, 256]`), `clean_logits` (`"raw"`),
`metadata_program`, `metadata_route`, `threads` and per-program `programs`
records keyed `h<H>:p<P>:n<N>:a<A>` (`atom_variant`, `atom_tokens`,
`next_n_atom`, `num_atoms`, `last_atom_tokens`, `num_q_stages`, `hoist_b`,
tile size, group count, KV stages, shared-memory bytes). A program record
carries `kind` (`"module"` or `"sequence"`), `role`, `sources`, `standalone`,
`compile_flags`, `ffi_entry`, `arg_plan` and the per-architecture
`closure_sha256`. Every route record has `stages` (`[[stage, program], ...]`),
`sequence` (the prepared-sequence binding, `null` for the metadata route),
`kernel_launches`, `num_heads`, `page_kv`, `next_n`, `next_n_atom`,
`atom_variant` and `clean_logits`.
