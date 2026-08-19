# Attention Backend Architecture: Batch MLA

**Scope**: Batch MLA public orchestration, planning contracts, backend ownership,
and the relationship between planned and functional execution paths.

## Summary

The Batch MLA implementation is the reference architecture for FlashInfer
attention APIs that expose several kernel backends through both a planned
wrapper and a functional entry point.

The core principles are:

- Public controllers own user-facing behavior and backend policy.
- Operation-specific contracts own metadata normalization and representation
  rules.
- Backend modules own complete backend vertical slices.
- Planned wrappers and functional dispatchers retain distinct orchestration
  lifecycles.
- Shared backend mechanics are extracted only when their contracts are truly
  identical.
- Public access remains through `flashinfer.mla`; the private package layout is
  not a new public API.

This is a reference convention, not a requirement that every attention API
implement one universal backend interface.

## Motivation

Before this refactor, Batch MLA behavior accumulated in a large core module.
Public API compatibility, metadata conversion, backend selection, autotuning,
module loading, persistent plan state, backend-specific validation, and launch
assembly were closely interleaved.

That structure made several kinds of change unnecessarily risky:

- Adding a backend required changing central control flow and understanding
  unrelated backends.
- Backend-specific options could be accepted, rejected, or ignored
  inconsistently.
- Functional and stateful APIs duplicated mechanics despite having different
  lifecycles.
- It was unclear which layer owned candidate ordering, backend viability,
  metadata conversion, or persistent state.
- Planning work could leak into the latency-sensitive run path.
- Refactoring internals risked changing public imports, signatures, fallback
  behavior, or CUDA Graph pointer stability.
- A global backend abstraction was tempting even though attention operations
  do not yet share one stable lifecycle or request model.

The resulting architecture creates explicit ownership boundaries while
preserving operation-specific behavior and the supported public namespace.

## Goals

- Make every backend a coherent vertical slice with a narrow private contract.
- Separate backend ordering from backend viability.
- Keep explicit backend requests strict.
- Permit automatic fallback only for a typed unsupported-plan result.
- Make wrapper planning transactional where the execution model permits it.
- Keep selection, autotuning, module loading, and persistent allocation out of
  the planned wrapper's run hot path.
- Make tensor representation and copy behavior explicit.
- Share mechanics between stateful and functional surfaces only below their
  lifecycle boundary.
- Establish an operation-first private package convention that other attention
  APIs can adopt incrementally.
- Preserve compatibility where it remains intentional and identify deprecated
  paths explicitly.

## Non-goals

- A repository-wide backend base class.
- One candidate set or autotuning lifecycle shared by every attention API.
- Public exposure of concrete backend classes or the `_batch_mla` package.
- Moving sparse DSV4 orchestration into the dense Batch MLA package.
- Implicit full-cache copies in the normal planned wrapper run path.
- Requiring live query or KV-cache tensors during planning. Serving systems may
  plan before those tensors exist.

## Durable Layering and Ownership

Batch MLA has two related but distinct flows.

Planned wrapper flow:

```text
BatchMLAPagedAttentionWrapper
  -> normalized plan request and metadata
  -> explicit or automatic planned-backend selection
  -> backend-owned plan_from_wrapper()
  -> committed backend state
  -> backend-owned run_from_wrapper()
  -> kernel
```

Functional flow:

```text
batch_mla_paged_attention
  -> immutable functional request
  -> explicit selector or functional auto policy
  -> backend-owned TunableRunner
  -> backend preparation and launch
  -> kernel
```

The functional controller does not construct or plan a temporary wrapper.
The two surfaces share only lower-level mechanics with identical contracts.

### Public controllers

The wrapper and functional controllers own:

- Public signatures and compatibility behavior.
- Request-wide validation.
- Backend request interpretation.
- Candidate ordering and architecture routing.
- Automatic-selection and autotuning policy.
- User-facing warnings, logs, and selection diagnostics.
- Publication of a successfully planned wrapper backend.

They do not own concrete kernel launch mechanics.

### Planning and contract modules

The operation-specific planning and contract modules own:

- `MLAPlanMetadata` and `MLAInputContract`.
- CSR and dense metadata validation and canonicalization.
- Lazy, request-local conversion between metadata forms.
- Packed and split tensor representation rules.
- The immutable `_MLAPlanArguments` request.
- Cross-backend capability declarations and validation.

They do not own backend ordering, concrete module loading, launch assembly, or
functional autotuning.

### Backend vertical slices

Each concrete backend module owns one operation/backend implementation,
including:

- Capability and applicability validation.
- Backend-specific lowering.
- Module, artifact, compiler, or launcher acquisition.
- Backend-owned persistent plan state.
- Kernel launch argument assembly.
- The planned wrapper implementation.
- The functional `TunableRunner`, when that backend supports the functional
  API.

A concrete backend must not call back into a public controller to make a
selection decision. Shared leaves stay narrow, policy-free, and close to their
consumers.

## Package Structure

The operation-first private structure is:

```text
flashinfer/mla/
|-- __init__.py
|-- _core.py
|-- _sparse_mla_sm120.py
`-- _batch_mla/
    |-- __init__.py
    |-- _auto_policy.py
    |-- _contracts.py
    |-- _functional.py
    |-- _planning.py
    |-- _wrapper.py
    `-- _backends/
        |-- __init__.py
        |-- _capabilities.py
        |-- _fa_common.py
        |-- _cute_dsl_common.py
        |-- _cute_dsl_functional_common.py
        |-- fa2_backend.py
        |-- fa3_backend.py
        |-- cutlass_backend.py
        |-- trtllm_gen_backend.py
        |-- cute_dsl_monolithic_backend.py
        |-- cute_dsl_modular_backend.py
        `-- xqa_backend.py
```

Concrete modules use the `_backend.py` suffix. Shared leaves use
leading-underscore names that describe the mechanism they share.

A complex future operation should receive its own private package when it has
several backend, planning, or shared-mechanism modules. A simple API should
remain in its parent core until that boundary provides a demonstrated benefit.

## Root and Batch MLA Surface Ownership

The root `mla/_core.py` remains the public facade and the dense/sparse routing
boundary. It owns:

- Public orchestration that spans dense and sparse paths.
- Delegation into the private dense Batch MLA package.
- Sparse and DSV4 validation and routing.
- Compatibility aliases whose public identities must remain stable.

The dense functional controller, `_batch_mla/_functional.py`, owns:

- `batch_mla_paged_attention`.
- Construction of `_FunctionalMLARequest`.
- Explicit functional backend selection.
- Functional auto policy and `AutoTuner` integration.
- The mapping from concrete selectors to backend-owned runners.

The planned wrapper controller, `_batch_mla/_wrapper.py`, owns:

- `BatchMLAPagedAttentionWrapper`.
- Public `plan()` and `run()` signatures.
- Metadata and run-input normalization.
- Explicit, CuTe-family, and global automatic selection policy.
- Typed automatic fallback.
- Transactional publication of selected backend state.
- Backend-neutral validation of the planned run contract.

Sparse orchestration remains in the root core. The hardware-specific SM120
sparse implementation remains top-level and is not part of the dense backend
package.

## Planning Contract

### Metadata forms

`plan()` accepts one `MLAPlanMetadata` containing either or both canonical
forms:

- CSR metadata: `qo_indptr`, `kv_indptr`, `kv_indices`, and `kv_len_arr`.
  This form is native to FA2 and FA3.
- Dense metadata: `cum_seq_lens_q`, `block_tables`, `seq_lens`, and optional
  `max_q_len`. This form is native to CUTLASS, TRTLLM-GEN, CuTe DSL, and XQA.

`MLAPlanMetadata.csr()`, `.dense()`, and `.dual()` construct these forms. If
both forms are supplied, they must describe the same requests and page mapping.

Each metadata tensor may be on CPU or the wrapper's device, and one form may
mix those locations. The resolver co-locates values only when a logical check
spans devices, while backend adapters receive device-resident launch metadata.
Generated FA planning can therefore consume host indptr and length tensors
without first copying them from CUDA; dense and run-time metadata is staged on
the wrapper device when its backend requires it. Metadata on another
accelerator device is rejected.

The planner preserves the supplied representation and derives the selected
backend's alternate form only when needed. Derived metadata is scoped to the
planning request and is never reconstructed during a normal planned `run()`.

Flat CSR and dense metadata arguments remain deprecated compatibility forms.
They are normalized into the same internal metadata object.

### Structural planning values

Planning may happen before live query and KV-cache tensors exist. Therefore,
the caller supplies the structural values needed for backend selection and
plan construction:

- Number of query heads.
- Compressed and positional head dimensions.
- Page size.
- Causality and softmax scale.
- Query and KV-cache dtypes.
- Declared query and KV-cache layouts.
- Output, LSE, scaling, sink, profiler, and skip-softmax behavior where
  applicable.

`query_layout` and `kv_cache_layout` each use two public values:

- `"packed"`: the two logical components can be provided as one last-dimension
  packed tensor. Adjacent split views of the same storage are also accepted
  because they can be reinterpreted zero-copy.
- `"split"`: independent component tensors are supported. A packed tensor can
  still be sliced into split zero-copy views.

The preferred metadata-object form defaults both layouts to `"packed"`.
Deprecated flat metadata retains its historical split default.

These declarations describe the representation that later calls must be able
to provide. They are not representative tensors and do not key planning on
tensor shapes, devices, or identities beyond the explicit structural values.

## Runtime Tensor Contract

The preferred wrapper runtime interface supplies `query` and `kv_cache`.
Each value may be:

- A packed tensor.
- A complete `(left, right)` split pair.
- A trusted redundant form containing both packed and split references.

The selected backend resolves the native representation. Conversion rules are
intentionally asymmetric:

| Planned/native need | Caller provides | Behavior |
| --- | --- | --- |
| Packed | Packed tensor | Pass through. |
| Packed | Adjacent split views | Reinterpret as one zero-copy packed view. |
| Packed | Independent split pair | Reject and require replanning for split input. |
| Split | Split pair | Pass through. |
| Split | Packed tensor | Slice into zero-copy component views. |

The normal planned wrapper does not concatenate independently allocated split
KV caches for a packed-native backend. Copying a full paged cache on every
decode step would violate the hot-path contract.

The functional API has a different lifecycle. A one-shot functional call may
materialize an independent split pair with `torch.cat()` when it selects a
packed-native runner. It emits a process-wide warning because this introduces a
per-call allocation and copy. This behavior preserves functional compatibility
without weakening the planned wrapper's no-copy contract.

The deprecated explicitly requested CUTLASS run-without-plan path is the other
intentional exception. It constructs and pins a CUTLASS plan from runtime
metadata, and independently allocated split inputs are concatenated on the GPU
on each call. The path warns and exists only for historical compatibility.

The legacy `q_nope`, `q_pe`, `ckv_cache`, and `kpe_cache` wrapper keywords and
positional wrapper arguments remain deprecated adapters to the structural
`query` and `kv_cache` interface.

## Backend Selection and Fallback

Selection policy belongs to the operation controller, not to a global registry
or to backend modules.

### Explicit requests

An explicit request evaluates only the requested backend. Unsupported
configurations, invalid options, compilation failures, loading failures, and
planning failures surface to the caller. Explicit requests never silently try
another backend.

The `"cute-dsl"` family selector is a narrow exception in naming, not failure
semantics: it selects between the monolithic and modular CuTe DSL
implementations according to their declared capabilities. Either concrete name
can be requested when the caller requires one implementation.

### Automatic wrapper selection

The wrapper owns a complete architecture-preferred candidate order. Every
candidate performs its own capability and plan-time viability validation.
Automatic selection continues only when a candidate raises the narrow internal
unsupported-plan exception. Runtime, allocation, compilation, and programming
errors are not fallback signals.

Outside an autotuning context, the first viable candidate is selected
deterministically. When the existing `AutoTuner` context requests tuning or
cache lookup, `_auto_policy.py` profiles or resolves viable candidates using
synthetic inputs derived from normalized planning facts. Synthetic allocation
and profiling happen during planning, never during the committed run path.

The wrapper publishes an `MLAAutoSelectionTrace` containing candidates,
rejections, selection mode, and the resolved backend for diagnostics.

### Functional selection

Functional auto intentionally remains separate from wrapper auto. It retains
its own immutable request, candidate set, and `AutoTuner` lifecycle. Explicit
functional backends may exceed the functional auto candidate set. The two
policies should not be unified without product and performance evidence.

## Plan State and Transactionality

Concrete backend objects own completed plan state, compiled modules, launchers,
metadata, and persistent launch resources. The wrapper retains only
backend-neutral input compatibility state, selection diagnostics, the selected
backend name, and the selected implementation.

The wrapper publishes a backend only after `plan_from_wrapper()` succeeds. A
rejected candidate or failed explicit plan cannot partially replace a previous
successful plan.

With CUDA Graph mode enabled, dense backends reject replanning because their
metadata pointers cannot be replaced safely. If automatic planning first
selects FA2 or FA3, later graph-mode replans remain pinned to that concrete
backend instead of falling through to another implementation. Direct external
graph replay remains the caller's responsibility because the Python wrapper
cannot intercept it.

## Planned Run Hot-Path Contract

After a wrapper commits a backend, `run()` remains launch-oriented.

Allowed work includes:

- Immediate validation of runtime-only options.
- Lightweight argument normalization.
- Zero-copy packed/split view resolution.
- Output preparation required by the selected backend.
- Lowering that inherently depends on current runtime tensors.
- Delegation to the selected backend's `run_from_wrapper()` method.

The normal planned run path must not perform:

- Backend selection or candidate fallback.
- Support probing or autotuning.
- JIT or module loading.
- Persistent metadata conversion or workspace allocation.
- Rejected-candidate diagnostics.
- Reconstruction of plan state.
- Implicit materialization of an independently split paged KV cache.

The one-shot functional dispatcher is evaluated against its own lifecycle and
is not required to satisfy the planned wrapper's run-path constraints.

## Public API and Compatibility

The supported access path remains `flashinfer.mla.<symbol>`. There is no public
`flashinfer.mla.batch_mla` module.

Compatibility rules include:

- `mla/__init__.py` continues to expose the supported root namespace.
- Public objects implemented in `_batch_mla` are assigned compatibility aliases
  from the root core.
- Private implementation modules and classes are not stable APIs.
- Flat metadata and positional wrapper arguments remain deprecated.
- Legacy split runtime keywords remain deprecated adapters.
- Meaningful backend restrictions fail explicitly.
- Unknown keywords are not forwarded through arbitrary `**kwargs` plumbing.
- Direct TRTLLM-GEN and XQA functional entry points remain deprecated facades
  over `batch_mla_paged_attention`.
- Removed private concern-first module paths do not receive compatibility
  shims.
- Sparse DSV4 behavior remains upstream-owned in the root core.

## Convention for Future Attention APIs

When applying this architecture to another operation:

1. **Classify the lifecycle.** Decide whether the API is a persistent planned
   wrapper, a one-shot functional dispatcher, an autotuned dispatcher, or a
   combination. Do not impose `plan()` and `run()` on a functional API merely
   for consistency.
2. **Define controller ownership.** Document public validation, explicit
   backend behavior, automatic ordering or tuning, architecture routing,
   warnings, and compatibility.
3. **Create an operation package only when warranted.** Use an operation-first
   package when several backend, planning, or shared-mechanism modules need a
   boundary.
4. **Give each backend a vertical slice.** Keep capability checks, resource
   loading, persistent state, and launch assembly together. Backends should not
   depend on one another.
5. **Separate selection from viability.** The controller orders candidates;
   each backend validates exact support. Automatic fallback catches only the
   typed unsupported-plan result.
6. **Define publication and failure behavior.** Specify request-local versus
   backend-owned state, when wrapper state commits, and what replanning means
   for CUDA Graphs.
7. **Protect the run path.** List the work that belongs in planning and verify
   it does not reappear during normal execution.
8. **Keep adapters lifecycle-specific.** Planned wrappers use request-local
   plan arguments and persistent backend state; functional calls use immutable
   per-call requests and runners.
9. **Extract shared leaves narrowly.** Share only stable identical mechanics,
   keep them close to consumers, and prevent them from absorbing policy.
10. **Preserve the supported namespace.** Re-export public symbols through the
    established package and do not expose the private operation package.

## Current Implementation

Batch MLA currently provides separate wrapper and functional controllers,
request-local planning, explicit capability contracts, automatic wrapper
selection with optional autotuning, and seven concrete backends:

- FA2.
- FA3.
- CUTLASS.
- TRTLLM-GEN.
- CuTe DSL monolithic.
- CuTe DSL modular.
- XQA.

It also provides:

- Backend-owned wrapper adaptation and functional runners.
- `MLAPlanMetadata` with lazy CSR/dense normalization.
- Explicit packed/split planning and raw runtime tensor contracts.
- Typed unsupported fallback and transactional publication.
- Architecture-preferred wrapper ordering.
- Separate wrapper and functional benchmark lifecycles.
- Explicit public exports and deprecated compatibility facades.
- Production validation across Hopper, datacenter Blackwell, and consumer
  Blackwell backend families.

## Residual Risks and Follow-up Work

- Wrapper and functional auto intentionally have different candidate sets and
  lifecycles.
- Shape-informed wrapper tuning and deterministic architecture ordering coexist
  in one policy module and should remain behaviorally distinguishable.
- Deprecated flat metadata, positional inputs, legacy split keywords, direct
  functional facades, and CUTLASS run-without-plan need future removal
  decisions.
- The wrapper's public parameter superset may become difficult to understand as
  backend-specific features expand. Typed backend options or experimental
  direct APIs may eventually be warranted.
- Private backend contracts and runner classes remain intentionally unstable.
- Sparse orchestration remains in the root core as a staged boundary.
- Reusable abstractions for other attention operations should be extracted from
  demonstrated repetition rather than designed ahead of evidence.
- Adoption should remain incremental; this architecture does not impose
  structural uniformity across FlashInfer.
