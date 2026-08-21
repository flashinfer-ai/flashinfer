# Batch MLA backend architecture

This document describes the backend boundary for
`BatchMLAPagedAttentionWrapper` and its FA2, FA3, and CUTLASS backends. Public
access is through `flashinfer.mla`; the `_batch_mla` package and its concrete
backend classes are private implementation details.

## Summary

The architecture separates four concerns at the public wrapper boundary:

- The wrapper owns the public lifecycle and compatibility policy.
- Contract and planning modules own metadata meaning, tensor representation,
  and plan/run compatibility.
- Each backend module owns a complete concrete planning and launch
  implementation.
- A normal planned `run()` performs only runtime validation, zero-copy input
  lowering, output preparation, and dispatch to the already planned backend.

The plan/run flow is:

```text
flashinfer.mla.BatchMLAPagedAttentionWrapper
  -> normalize canonical or deprecated plan inputs
  -> build one backend-neutral plan request
  -> selected backend.plan_from_wrapper()
  -> publish the completed backend and run contract
  -> normalize and validate run inputs
  -> selected backend.run_from_wrapper()
  -> kernel
```

There is no generic repository-wide backend interface in this design. The
contracts are specific to Batch MLA.

## Motivation

Batch MLA planning must preserve a mature public API while handling several
metadata representations, multiple tensor layouts, backend-specific
restrictions, persistent plan state, and CUDA Graph pointer lifetime. When
those concerns are interleaved, a backend change can alter public behavior, and
planning work can accidentally move into the latency-sensitive run path.

The boundary makes ownership explicit:

- `BatchMLAPagedAttentionWrapper` accepts canonical and deprecated inputs,
  chooses the configured backend, and publishes only a successful plan.
- `MLAPlanMetadata` and `_MLAPlanMetadataResolver` validate and translate
  request-local metadata.
- `MLAInputContract` records the runtime options and layouts fixed by a plan.
- FA2, FA3, and CUTLASS own their backend-specific validation, persistent state,
  module loading, output preparation, and launch assembly.

This separation keeps backend behavior independently inspectable while leaving
functional APIs and selection machinery outside this interface.

## Design properties

- Public entry points live in the supported `flashinfer.mla` namespace.
- Canonical planning metadata and planned runtime requirements are explicit.
- Backend modules contain backend-specific constraints and launch mechanics.
- CSR or dense metadata is derived only when the selected backend needs it.
- Packed/split conversion has explicit zero-copy and rejection behavior.
- Plan state is published transactionally where the execution model permits it.
- Backend selection, module loading, metadata conversion, and persistent
  allocation stay outside the normal planned run path.
- Historical compatibility behavior is isolated and deprecated.

## Interface boundaries

This architecture is specific to the planned Batch MLA wrapper. It does not
define:

- A common base class for every attention backend in FlashInfer.
- A functional `batch_mla_paged_attention` API or functional runner lifecycle.
- A backend registry, candidate loop, typed fallback, selection trace, or
  autotuning policy.
- Additional Batch MLA backends beyond FA2, FA3, and CUTLASS.
- Public exposure of `_batch_mla` or its concrete backend classes.
- Sparse DSV4 orchestration inside the dense Batch MLA package.

Planning does not require live query or KV-cache tensors, and a normal planned
run does not implicitly copy an independently split paged KV cache.

## Layering and ownership

### Public facade

`flashinfer.mla` is the public facade. Its `_core` module is the dense/sparse
routing boundary and imports the canonical wrapper and metadata objects from
the private package. Callers use:

```python
flashinfer.mla.BatchMLAPagedAttentionWrapper
flashinfer.mla.MLAPlanMetadata
```

There is no supported `flashinfer.mla.batch_mla` namespace. Private module
paths and concrete backend class names are not stable APIs.

### Wrapper controller

`_batch_mla/_wrapper.py` owns:

- The public `BatchMLAPagedAttentionWrapper`, `plan()`, and `run()` signatures.
- Interpretation of `backend="auto"`, `"fa2"`, `"fa3"`, or `"cutlass"`.
- Canonical and deprecated argument normalization.
- User-facing compatibility warnings.
- Construction of the backend-neutral `_MLAPlanArguments` request.
- Validation of planned runtime options through `MLAInputContract`.
- Zero-copy lowering of structural query and KV-cache values.
- Transactional publication of a completed backend.

It does not own generated-module planning or backend kernel launch assembly.

### Contract and planning modules

`_batch_mla/_contracts.py` owns:

- `MLAPlanMetadata`, the canonical public metadata value.
- `MLAInputContract`, the immutable plan/run compatibility value.
- The exact packed, split, and trusted-redundant runtime tuple grammar.
- Structural validation and zero-copy packed/split view resolution.

`_batch_mla/_planning.py` owns:

- CSR and dense metadata validation.
- Logical equivalence checks for dual metadata.
- Lazy conversion between CSR and dense forms.
- Device staging of dense launch metadata.
- `_MLAPlanArguments`, the immutable request passed to a backend.

`_batch_mla/_backends/_capabilities.py` holds declarative capability facts and
pure rejection helpers. Capabilities explain why a selected backend cannot
satisfy a plan; they are not a registry or fallback mechanism.

### Backend vertical slices

Each concrete backend owns one complete planned implementation:

- Capability and backend-specific plan validation.
- Selection of its native metadata representation.
- Module or launcher acquisition.
- Persistent workspace and plan state.
- Metadata staging required by its execution model.
- Output and LSE preparation.
- Kernel argument assembly and launch.

FA2 and FA3 share generated-backend mechanics in `_fa_common.py` because they
use the same planning, workspace, staging, and run contracts. The
concrete `fa2_backend.py` and `fa3_backend.py` modules own the named
backend classes and generated-module specialization. CUTLASS has a separate
implementation because its metadata, layout, hardware, and output contracts
differ.

## Package structure

The private package is operation-first:

```text
flashinfer/mla/
|-- __init__.py
|-- _core.py
`-- _batch_mla/
    |-- __init__.py
    |-- _contracts.py
    |-- _planning.py
    |-- _wrapper.py
    `-- _backends/
        |-- __init__.py
        |-- _capabilities.py
        |-- _fa_common.py
        |-- fa2_backend.py
        |-- fa3_backend.py
        `-- cutlass_backend.py
```

Concrete modules use the `_backend.py` suffix. Shared leaves remain private and
name the mechanism they share rather than introducing a general backend layer.

## Plan lifecycle

`plan()` has six phases:

1. Normalize one canonical metadata value or one deprecated flat metadata form.
2. Normalize the structural and output contract for subsequent `run()` calls.
3. Build `_MLAPlanArguments`, which owns a request-local lazy metadata resolver.
4. Enforce CUDA Graph replanning restrictions.
5. Ask the configured concrete backend to build a complete backend instance.
6. Publish the backend and `MLAInputContract` only after planning succeeds.

The wrapper's configured backend is fixed at construction. An explicit request
evaluates only that backend. `backend="auto"` calls the
`determine_mla_backend()` architecture helper once and selects FA3 on
supported SM90a devices and FA2 otherwise. It does not consider CUTLASS, iterate
candidates, or fall back after a planning error. On Blackwell, the wrapper warns
that this legacy architecture default is not Blackwell-native and points callers
to the available alternatives.

Backend capability rejection, module-loading failures, allocation failures, and
planning errors surface to the caller. They do not silently select another
backend.

## Planning contract

### Metadata forms

`MLAPlanMetadata` can contain one or both canonical forms:

- CSR: `qo_indptr`, `kv_indptr`, `kv_indices`, and `kv_len_arr`. This is native
  to FA2 and FA3.
- Dense: `cum_seq_lens_q`, `block_tables`, `seq_lens`, and optional
  `max_q_len`. This is native to CUTLASS.

`MLAPlanMetadata.csr()`, `.dense()`, and `.dual()` construct these values. A
form must be complete. If both forms are present, the resolver verifies that
they describe the same query boundaries, KV lengths, page counts, and live page
mapping.

Metadata tensors must be contiguous `torch.int32` tensors on CPU or the wrapper
device. One request may mix those two locations. Metadata on another
accelerator is rejected. The metadata value retains caller tensor references;
validation and conversion happen only when the configured backend asks the
resolver for CSR or dense data.

The resolver preserves a supplied native representation. If conversion is
needed, the derived representation is cached only for that plan request. FA2
and FA3 request CSR metadata and stage launch metadata to the wrapper device.
CUTLASS requests device-resident dense metadata whose table width is aligned to
`128 / page_size`.

Canonical metadata receives strict shape and value validation. Deprecated flat
CSR input is normalized through the same resolver but deliberately isolates the
historical noncanonical batch-shape tolerance needed by existing callers. Calls
through the canonical metadata interface do not receive that tolerance.

### Structural and output facts

Planning does not require representative query or KV-cache tensors. Callers
instead declare the facts needed to build a backend plan:

- Number of query heads and compressed/positional head dimensions.
- Page size, causality, and softmax scale.
- Query, KV-cache, and output dtypes.
- Planned query and KV-cache layouts.
- LSE, output scaling, KV scaling, profiler, and skip-softmax options.

Canonical `metadata=` calls default `query_layout` and `kv_cache_layout` to
`"packed"`. Deprecated flat metadata calls retain the historical `"split"`
defaults. Both can override those defaults explicitly.

The wrapper lowers these facts into `_MLAPlanArguments`. The selected backend's
capability declaration rejects unsupported LSE, KV layout, output scaling,
scale, and skip-softmax combinations before backend-specific planning proceeds.
The backend then validates constraints such as dtype, shape, softmax scale, and
hardware support.

After planning succeeds, `MLAInputContract` records the query/KV layouts, split
widths, LSE mode, output dtype, output scaling mode, and KV scaling mode.
`run()` must satisfy this contract; changing it normally requires replanning.

## Runtime tensor contract

The preferred runtime interface supplies `query=` and `kv_cache=`. Each value
uses this exact grammar:

- A tensor is packed across the last dimension.
- `(left, right)` is a complete split pair.
- `(packed, (left, right))` and `((left, right), packed)` are trusted redundant
  forms containing both representations.

The planned layout determines which representation is validated and consumed:

| Planned need | Caller provides | Behavior |
| --- | --- | --- |
| Packed | Packed tensor | Validate and pass through. |
| Packed | Adjacent split views | Reinterpret as one zero-copy packed view. |
| Packed | Independent split pair | Reject; replan for split input. |
| Split | Split pair | Validate and pass through. |
| Split | Packed tensor | Slice into zero-copy component views. |
| Either | Trusted redundant value | Validate and use only the planned member. |

Split tensors must agree in rank, leading shape, dtype, device, and planned
last-dimension widths. A split pair is adjacent only when both tensors are
in-bounds views of the same contiguous storage with compatible strides. The
wrapper can then create a packed view without allocating or copying.

The redundant form is explicitly trusted: the wrapper does not prove that its
packed and split members contain equivalent values. The caller owns that
invariant, and the member not selected by the plan is not inspected.

For a normal planned run, independently allocated split values never satisfy a
packed layout through concatenation. In particular, the wrapper does not copy a
full paged KV cache on every decode step.

### Planned run options

Before structural lowering, `MLAInputContract.validate_run_options()` checks:

- Whether no LSE, base-2 LSE, or natural-log LSE was planned.
- The dtype of a caller-provided output buffer.
- The presence or absence of CUTLASS per-tensor output scaling.
- The complete presence or absence of generated-FA KV scales.

Caller-owned `out` and `lse` buffers are used directly; when returned, they are
returned by identity. Backend code performs the remaining tensor-shape, dtype,
device, and backend-specific option checks.

## Backend contracts

| Backend | Native metadata | Kernel input | LSE | Output scale | CUDA Graph replan |
| --- | --- | --- | --- | --- | --- |
| FA2 | CSR | Split | None, base 2, or base e | None | Supported with reserved metadata buffers |
| FA3 | CSR | Split | None, base 2, or base e | None | Supported with reserved metadata buffers |
| CUTLASS | Dense | Packed | None | None or per-tensor FP8 | Rejected |

### FA2 and FA3

The generated-FA backends accept FP16 or BF16 queries and require the output
dtype to match the query dtype. KV-cache data may be FP16, BF16, or FP8 E4M3.
FP8 KV-cache plans require an SM90 device, a BF16 query, compressed
width 512, positional width 0 or 64, and the per-tensor KV scale contract.
At run time, that contract requires `kpe_scale` plus exactly one scalar
`ckv_scale` or contiguous FP32 `ckv_scale_arr`. Non-FP8 plans use the default
no-scale contract.

FA2 and FA3 support causal and non-causal planning, all three LSE modes, and
backend profiler planning. Their kernel-facing query and cache representation is
split; a packed planned input reaches that representation through zero-copy
slices.

### CUTLASS

CUTLASS is available only when explicitly selected as the wrapper backend. It
requires:

- Compute capability major version 10 or 11.
- Non-causal attention with exactly 128 query heads.
- Compressed and positional widths 512 and 64.
- Matching FP16 or BF16 query and KV-cache dtypes.
- A page size no greater than 128 that divides 128.
- The fixed MLA softmax scale `1 / sqrt(128 + 64)`.
- Packed query and KV-cache representations.
- No LSE, profiler, KV scale, or skip-softmax contract.

Unscaled output has the query dtype. Per-tensor output scaling requires a
caller-provided FP8 E4M3 or E5M2 output buffer and a finite positive `o_scale`.
A planned call normally reuses its planned `kv_len` and `page_table`; callers
may override both together when batch size and page size remain compatible with
the plan.

## Plan state and CUDA Graph safety

A new backend instance owns the resources it creates while planning. The
wrapper does not replace `_planned_backend` or `_input_contract` until
`plan_from_wrapper()` returns successfully. A failed non-graph replan therefore
leaves the previously published plan usable.

Generated-FA CUDA Graph planning also stages metadata transactionally. Before
copying, it verifies each reserved buffer's presence, rank where required,
`int32` dtype, wrapper device, contiguity, capacity, pairwise non-overlap, and
source-to-target alias safety. If a copy fails, snapshots restore every target
buffer before the error is re-raised.

The reserved `qo_indptr`, `kv_indptr`, and `kv_len_arr` shapes must match the
new metadata exactly. The reserved `kv_indices` buffer may be larger than its
active prefix. Generated-FA graph replanning can therefore change the live
page-index count within capacity, but cannot change the reserved batch shapes.

After a successful generated-FA graph replan, the wrapper retains the previous
backend object. This keeps its plan-owned workspaces alive for graphs that
reference their addresses. Retention is a lifetime guarantee, not a
promise that the Python wrapper can intercept or validate direct external graph
replay.

CUTLASS graph-mode replanning is rejected because its dense metadata pointers
do not have an equivalent reserved-buffer protocol. An initial CUTLASS plan may
be used, but callers must construct another wrapper to change that plan in
CUDA Graph mode.

## Planned run hot path

After a backend is published, a normal `run()` is launch-oriented. It may:

- Normalize structural and deprecated runtime arguments.
- Validate runtime options against the planned contract.
- Create zero-copy packed or split views.
- Validate backend-specific runtime tensors and options.
- Prepare caller-requested output or LSE storage.
- Delegate to the selected backend's `run_from_wrapper()` method.

A normal planned `run()` does not:

- Select or fall back to another backend.
- Perform support probing or autotuning.
- Load or compile a module.
- Rebuild persistent plan metadata.
- Allocate persistent workspaces.
- Concatenate an independent split cache for a normal planned request.

Compatibility paths described below are intentionally excluded from this
normal hot-path guarantee.

## Compatibility and deprecation boundaries

The following adapters preserve existing callers. Each deprecated form has a
canonical replacement:

- Positional `plan()` and `run()` arguments warn once per wrapper instance.
- Flat CSR or dense plan metadata warns once per process; use
  `metadata=MLAPlanMetadata.csr(...)` or `.dense(...)` instead.
- Separate `q_nope` / `q_pe` and `ckv_cache` / `kpe_cache` run parameters warn
  once per wrapper; use structural `query=` and `kv_cache=` values instead.
- Deprecated flat CSR FA2/FA3 plans preserve their historical
  dynamic LSE behavior and warn when a run relies on it. Canonical plans require
  plan/run LSE agreement.
- An explicitly requested CUTLASS wrapper may still call `run()` without a
  prior `plan()` when both `kv_len` and `page_table` are present. This adapter
  builds a dense plan from runtime facts and may concatenate independently split
  query or KV-cache tensors. It warns and is the only planned-wrapper path that
  intentionally permits those copies.

Trace and Trace Apply integration preserve the public wrapper identity and
plan-owned metadata capture. The MLA trace template normalizes structural
`query` and `kv_cache` values to its stable split schema using the planned split
widths. Compatibility adapters remain registered for both the historical
`_core` module path and the canonical private implementation path.
