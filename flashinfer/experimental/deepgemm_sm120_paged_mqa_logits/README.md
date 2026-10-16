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
`next_n`, the batch `B = context_lens.shape[0]` and, optionally, a caller-known
upper bound of the batch's context lengths (`max_context_hint`, below) — and
named `sm120:fp8:h<H>:p<page_kv>:n<next_n>:a<atom>`, where `a<atom>` is the
Q-atom variant (tokens per Q atom) the catalog's `atom_rule` selects from
`(H, next_n, B[, max_context_hint])`:

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

**Long-context hint for batches 2 to 8.** The batch-only rule keeps the
reference atoms at `2 <= B <= 8` because the context lengths live on the
device when the program is picked. At long contexts the native atom measured
faster there (CAKE-1373 r2 round R3, batch 8 at 16K and 64K KV tokens; see
the measured table below). A caller that knows an
upper bound of the batch's context
lengths on the host (for example the framework's `seq_lens_cpu` maximum)
passes it as `max_context_hint`. When the hint is at least
`atom_rule.long_context_min_kv` (16384 KV tokens) the catalog's
`selection_long` table applies. It differs from the table above in the
`2 <= B <= 8` column only:

| next_n | heads | `2 <= B <= 8`, hint `>= 16384` | atoms |
|---|---|---|---|
| 1 | 32, 64 | `a1` | 1 |
| 2 | 32, 64 | `a2` | 1 |
| 3 | 32, 64 | `a3` | 1 |
| 4 | 32, 64 | `a4` | 1 |
| 5, 6 | 32 | `a5` / `a6` | 1 |
| 5, 6 | 64 | `a2` (unchanged; the native atom exceeds the TMA Q box) | 3 |

`B = 1` and every batch from 9 up are unchanged by the hint. `None` (the
default) is the batch-only selection of the table above. The hint does not
add a program or a route: every variant it selects is one the batch rule
already ships.

`route_available(num_heads, page_kv, next_n[, batch][, max_context_hint=...])`
answers on the host (without `batch`: whether any variant of the geometry
ships); there is no fallback for an unsupported configuration. The scheduler
program is shared by every route (route `sm120:metadata`).

The launch grid of the logits program **is** the CTA budget: the programs take
no compile-line definitions, so one delivered text per program serves every SM
count and every `sm_count` override.

## Operands

`prepare_sm120_paged_mqa_logits(q, kv_cache, weights, context_lens,
block_table, max_context_len, *, page_kv=None, schedule_meta=None, output=None,
sm_count=None, max_context_hint=None)` binds

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
- `1 <= max_context_len <= S * page_kv`;
- `max_context_hint` (optional, `>= 1`): an upper bound, in KV tokens, of this
  batch's context lengths. It selects the Q-atom program for `2 <= B <= 8`
  only (see "Long-context hint" above) and is stored as `plan.max_context_hint`.
  `max_context_len` is the logits row width; it is not used for the selection.

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

# With a host-known upper bound of the batch's context lengths: the SAME hint
# on both calls of one decode step.
hint = int(seq_lens_cpu.max())
meta = get_paged_mqa_logits_metadata(
    context_lens_2d, page_kv, num_sms, num_heads=q.shape[2], max_context_hint=hint
)
logits = fp8_paged_mqa_logits(
    q, kv_cache, weights, context_lens_2d, block_table, meta, max_context_len,
    max_context_hint=hint,
)
```

Full signatures: `get_paged_mqa_logits_metadata(context_lens, page_kv,
num_sms, indices=None, *, num_heads=None, max_context_hint=None, out=None)`,
`fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table,
schedule_meta, max_context_len, clean_logits=False, indices=None, *,
page_kv=None, max_context_hint=None)` and `prepare_sm120_paged_mqa_logits`
as in "Operands".

These mirror the `deep_gemm` calls a DeepSeek-V3.2 serving engine makes, so a
caller switches backends without re-shaping its tensors. `schedule_meta` fixes
the CTA budget through its first dimension (`None` selects the device's SM
count) and is rebuilt in the same sequence before the logits program reads it,
so the one-shot entry is self-contained. `num_heads` is optional on the
metadata entry: the schedule's item count is the Q-atom count of
`(num_heads, next_n, B)`, and the head count is required only where the two
exported head counts select different schedules — `next_n` 5 and 6 at
`B > 8` — in which case the `ValueError` names it; for `next_n` 1–4 and for
every `B <= 8` without a long-context hint the DeepGEMM-shaped call without
`num_heads` is exact (the hint changes this for `next_n` 5 and 6; see below).
`clean_logits=True` is rejected for two-dimensional context lengths exactly as
DeepGEMM does, and `indices` (variable-length request selection) must be
`None` — that scheduling branch is not ported.

`max_context_hint` is accepted by all three entries and has the same meaning
on each: an upper bound, in KV tokens, of this batch's context lengths. It is
used only to pick the Q-atom program for `2 <= B <= 8`; `None` keeps the
batch-only selection. The metadata call and the logits call of one decode step
must receive the same hint, exactly as they must receive the same `num_heads`.
The schedule buffer is DeepGEMM-identical and carries no atom count, so a
mismatch is not detectable from the buffer; the logits entries therefore
rebuild the schedule from their own `(context_lens, num_heads,
max_context_hint)` before the program reads it (see above), which overwrites
a buffer built under another hint instead of consuming it. Keep both calls
consistent anyway: the metadata entry's buffer is what a DeepGEMM-style
caller inspects. With a hint of at least 16384 at
`2 <= B <= 8`, `num_heads` becomes required on the metadata entry for
`next_n` 5 and 6 as well (32 heads select `a5` / `a6`, 64 heads keep `a2`);
`next_n` 1 to 4 still resolve without it. The positional `max_context_len`
(the logits row width) is never used for the selection.

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

## Measured against DeepGEMM (2026-10-15, Cake `02a3e0a3`, with `max_context_hint`)

Cake / DeepGEMM kernel time (`bench_gpu_time`, CUPTI, cold L2, interleaved pairs; DeepGEMM `yyihuang/DeepGEMM@79a4ef3b` on the same device and clocks), geometric mean over the 12 (batch, context) rows of each program ((batch, average context) = (1, 16K), (8, 4K), (8, 16K), (8, 64K), (32, 16K), (32, 64K), (64, 4K), (64, 16K), (128, 16K), (128, 64K), (256, 4K), (256, 16K); 216 rows per device). `faster` counts rows with ratio < 1 and a bootstrap 95 % CI upper bound <= 1; `tie` counts rows within 0.5 % whose CI contains 1 (on the DeepGEMM-identical schedules, plus RTX 5090 `h64p64n4` at batch 8 × 16K under the hint). Logits are bitwise-identical to DeepGEMM on every row of both devices.

| program | RTX 5090 Cake/DG | RTX 5090 faster / tie / slower | RTX PRO 6000 Cake/DG | RTX PRO 6000 faster / tie / slower |
|---|---:|---:|---:|---:|
| `h32p128n1` | 0.996 | 10 / 1 / 1 | 0.997 | 10 / 0 / 2 |
| `h32p128n2` | 0.991 | 11 / 1 / 0 | 0.995 | 11 / 1 / 0 |
| `h32p128n3` | 0.703 | 12 / 0 / 0 | 0.769 | 12 / 0 / 0 |
| `h32p128n4` | 0.724 | 11 / 0 / 1 | 0.773 | 12 / 0 / 0 |
| `h32p128n5` | 0.614 | 12 / 0 / 0 | 0.702 | 12 / 0 / 0 |
| `h32p128n6` | 0.675 | 12 / 0 / 0 | 0.699 | 12 / 0 / 0 |
| `h32p64n1` | 0.997 | 9 / 0 / 3 | 0.998 | 11 / 1 / 0 |
| `h32p64n2` | 0.992 | 9 / 1 / 2 | 0.994 | 12 / 0 / 0 |
| `h32p64n3` | 0.711 | 12 / 0 / 0 | 0.770 | 12 / 0 / 0 |
| `h32p64n4` | 0.740 | 11 / 0 / 1 | 0.779 | 12 / 0 / 0 |
| `h32p64n5` | 0.626 | 12 / 0 / 0 | 0.680 | 12 / 0 / 0 |
| `h32p64n6` | 0.664 | 12 / 0 / 0 | 0.691 | 12 / 0 / 0 |
| `h64p64n1` | 0.988 | 12 / 0 / 0 | 0.994 | 11 / 0 / 1 |
| `h64p64n2` | 0.992 | 11 / 0 / 1 | 0.996 | 9 / 0 / 3 |
| `h64p64n3` | 0.739 | 12 / 0 / 0 | 0.757 | 12 / 0 / 0 |
| `h64p64n4` | 0.922 | 9 / 2 / 1 | 0.795 | 11 / 0 / 1 |
| `h64p64n5` | 0.885 | 12 / 0 / 0 | 0.888 | 12 / 0 / 0 |
| `h64p64n6` | 0.948 | 10 / 0 / 2 | 0.920 | 8 / 1 / 3 |
| **all 18 programs** | 0.815 | 199 / 5 / 12 | 0.835 | 203 / 3 / 10 |

Every row was measured with `max_context_hint` set to the batch's maximum context length (the production caller's bound), RTX 5090 on a `2u1g-b650-19xx` node and RTX PRO 6000 on `2u2g-emr-0475`. The hint changes the program selection only on the batch-8 rows at 16K and 64K context: the 32-head programs and `h64p64n3` / `h64p64n4` take the native atoms there, which is where the r2 delivery (batch-only selection, reference atoms) had lost most against DeepGEMM — for example RTX 5090 `h32p128n3` at batch 8 × 64K moves from 0.99 to 0.72 and `h32p128n5` from 0.98 to 0.59, RTX PRO 6000 `h32p64n5` at batch 8 × 64K from 0.97 to 0.75. Without the hint those rows keep the r2 numbers. The `slower` rows are DeepGEMM-identical schedules within noise (+0.05 % to +2.2 %) on both devices; the one hinted row that reads a tie is RTX 5090 `h64p64n4` at batch 8 × 16K (1.0000, CI [0.9989, 1.0011]).

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
head-count rule of the metadata entry, the long-context hint on all three
entries (including the schedule difference between the hinted and the
batch-only selection, and that the logits entries rebuild the schedule under
their own hint) and
the architecture-target gate. It skips on devices without catalogued programs.

## Catalog schema `sm120_paged_mqa.v1`

`sm120_paged_mqa_catalog.json` carries `arches`, `policy`, `programs` and
`routes`. Policy keys: `head_dim`, `fused_row_bytes` (132), `heads`, `page_kv`,
`next_n`, `split_kv` (128 for every exported program, so the metadata entry
needs no page size), `atom_rule` (the Q-atom selection rule as data:
`batch_class_min` — the first batch of every batch class, the last class open
up to `max_batch` —, `selection["h<H>:n<N>"]` — the variant `a<atom>` per
class — and `num_atoms["n<N>:a<A>"]` — the schedule's `num_next_n_atoms` per
variant; the runtime evaluates it with `atom_variant` / `next_n_atoms` and
never needs the producer; `atom_rule` also carries the two long-context keys
`long_context_min_kv`, the smallest `max_context_hint` in KV tokens that
selects from the second table (16384), and `selection_long["h<H>:n<N>"]`, the
same shape as `selection`, used when the hint is at least that bound and
differing from `selection` in batch class 1 only), `max_batch` (the scheduler's shared-memory request
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
