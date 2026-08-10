# Recurrent KDA H12 Phase-A evidence harness

This harness freezes the six H12 recurrent-KDA prefill cases used to review the
non-aligned-head public path added by [#4351]. It is an evidence generator, not
a checked-in performance claim. A complete promotion result still requires the
whole denominator on both SM100a and SM103a.

The harness branch is based on upstream `main` at
`2ab910c58fdd2392914ea05e2a8714946ac0eef6` and requires the #4351 route commit
`38bf507f9c9eba6b4544bee016d2bdf9c4fed02b` as an ancestor. At execution time it
also records the current FlashInfer commit, relevant source hashes, and the
resolved paths and hashes of the imported `flashinfer.kda` modules. An import
from a different checkout or any tracked/nonignored-untracked FlashInfer source
change is rejected. Evidence JSON must be written outside both verified source
checkouts, so generating a receipt never creates an exception to the clean-tree
gate.

## Frozen denominator

All cases use H=12, K=V=128, BF16 inputs and state, a provided initial state,
in-kernel Q/K L2 normalization, gate calculation, and beta sigmoid, with
`lower_bound=-5.0`.

| Case | Layout and sequence lengths | Seed |
| --- | --- | ---: |
| `h12_packed_512x32` | packed `(512,) * 32` | 12000 |
| `h12_packed_128x8` | packed `(128,) * 8` | 12001 |
| `h12_fixed_512` | fixed `512` | 12002 |
| `h12_fixed_8192` | fixed `8192` | 12003 |
| `h12_packed_mixed` | packed `(1300, 547, 2048, 963, 271, 3063)` | 12004 |
| `h12_packed_1024x8` | packed `(1024,) * 8` | 12005 |

The machine-readable source of truth is
[`presets/recurrent_kda_prefill_h12_phase_a.json`](presets/recurrent_kda_prefill_h12_phase_a.json).
It is strict: case identity, order, seeds, common parameters, and
`per_case_only` aggregation must match exactly. Cross-shape geometric means are
not produced.

## Correctness contract

For every case, the harness checks both the public output and the complete
final state at BF16 `atol=rtol=1e-2` against:

1. an independent direct token-by-token recurrence with a BF16 state store
   after every token;
2. the pinned MoonshotAI/FlashKDA implementation; and
3. FLA's Triton implementation.

The FlashKDA checkout must be exactly
`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`, with its CUTLASS submodule at
`5c149f52a436782210263fb2f19b354443a61c6a`. Tracked and nonignored untracked
changes in either checkout are rejected. The Python package and extension must
resolve inside that clean checkout and match a build manifest produced by
`build_flash_kda_phase_a.py` in the same GPU allocation. The helper force-builds
the extension and receipts the exact command, Python/C++/NVCC/CUDA/PyTorch
toolchain, the effective `FLASH_KDA_CUDA_ARCHS=auto` and `NVCC_THREADS` build
settings, Slurm allocation, GPU architecture, source/CUTLASS pins, imported
artifact paths, and package/extension SHA-256 hashes. An arbitrary or stale
`.so` without that binding is rejected. The evidence runner additionally
requires the manifest's Slurm job/cluster/partition/node, GPU UUID/architecture,
and Python/PyTorch/CUDA runtime to equal the current receipt process exactly;
same-architecture output from an earlier allocation is not reusable.

FLA is required and never impersonated. Before importing FLA, the runner sets
`FLA_FLASH_KDA=0` and `FLA_DISABLE_BACKEND_DISPATCH=1`, so the required FLA
comparison uses the default Triton implementation. Its package version, source
paths, source hashes, and Git facts are reported. Missing FLA, prior FLA import,
a non-Git install, or any tracked/nonignored-untracked FLA change fails closed.
Every one of the six cases must include FLA output and full-final-state
correctness plus raw CUPTI timing; the path is never omitted.

## Timing contract

Every candidate sample calls the public `flashinfer.kda.recurrent_kda` API.
CUPTI runtime/driver correlation selects all GPU kernel, memcpy, and memset
activities launched by that call. The reportable public duration is the span
from the first selected activity start through the last selected activity end.
For H12, each sample must contain exactly one `PackBetaForTmaKernel`, followed
by exactly one `kernel_flashkda_bf16_fused_m128`; otherwise the run fails.

The public report preserves raw per-sample values for GPU span, activity sum,
kernel sum, active union, uncovered gap, host submission, and synchronized E2E
(diagnostic only), plus launch/kernel/copy names, counts, correlation facts,
and activity order. Cold-L2 flushing completes before the CUPTI-clock host
bracket. There is no CUDA-event or wall-clock timing fallback.

`prepared_recurrence` is a clearly labeled recurrence-only view derived from
the same public samples. It excludes beta preparation and is never substituted
for the inclusive public metric. The pinned FlashKDA raw path, its explicit
public-state-semantics adapter, and required FLA/Triton path are reported
separately. Measurement blocks alternate forward and reverse path order.

## CUDA Graph and dual-architecture gates

Every GPU evidence run invokes exactly:

```bash
python -m pytest -q \
  tests/kda/test_recurrent_kda_prefill.py::test_frozen_prefill_non_aligned_heads_graph_refreshes_beta
```

This is the H6/H12 changed-beta CUDA Graph regression at source lines 981–1042.
The receipt records the command, node, parameterization, source hash, return
code, stdout, and stderr. A failure writes a non-complete receipt and stops the
evidence run.

A successful single-GPU receipt sets only
`complete_per_arch_denominator=true`. It is not a promotion claim. The reducer
requires exactly one SM100a and one SM103a receipt with matching frozen preset,
FlashInfer commit/source hashes, pinned FlashKDA source/package identity, clean
FLA commit/source hashes, graph-test source, all six ordered cases, all three
output/full-state oracles, and all four CUPTI timing paths. Only then does it
emit `promotion_complete_dual_arch=true`. It computes no cross-shape aggregate.

## Running and reviewing

CPU-only validation does not import torch or initialize CUDA:

```bash
python benchmarks/bench_recurrent_kda_prefill_h12_phase_a.py --validate-only
pytest -q benchmarks/test_kda_h12_evidence.py
```

Prepare the pinned peer checkout, then force-build and receipt it inside each
allocated SM100a/SM103a job. Both manifest and final evidence JSON must live
outside the source checkouts. The build helper and benchmark command below must
run sequentially in the same allocation and on the same selected GPU; submitting
them as separate jobs fails the receipt binding:

```bash
git clone https://github.com/MoonshotAI/FlashKDA.git /absolute/path/to/FlashKDA
git -C /absolute/path/to/FlashKDA checkout 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
git -C /absolute/path/to/FlashKDA submodule update --init cutlass
python benchmarks/build_flash_kda_phase_a.py \
  --flash-kda-source-dir /absolute/path/to/FlashKDA \
  --manifest /outside/checkouts/flash-kda-build-sm100a.json
```

The build-manifest schema can be audited without importing torch or CUDA:

```bash
python benchmarks/build_flash_kda_phase_a.py --validate-only
python benchmarks/build_flash_kda_phase_a.py \
  --validate-only \
  --manifest /outside/checkouts/flash-kda-build-sm100a.json
```

On an allocated SM100a or SM103a GPU worker with `cupti-python>=13`, a built
pinned FlashKDA checkout, and this FlashInfer worktree installed:

```bash
python benchmarks/bench_recurrent_kda_prefill_h12_phase_a.py \
  --flash-kda-source-dir /absolute/path/to/FlashKDA \
  --flash-kda-build-manifest /outside/checkouts/flash-kda-build-sm100a.json \
  --warmup-iters 5 \
  --repeat-iters 20 \
  --blocks 2 \
  --json /absolute/path/to/h12-phase-a.json
```

After both complete runs, freeze the expected FlashInfer and FLA commits at the
CLI boundary and reduce the two receipts:

```bash
python benchmarks/reduce_kda_h12_phase_a.py \
  --sm100a /outside/checkouts/h12-phase-a-sm100a.json \
  --sm103a /outside/checkouts/h12-phase-a-sm103a.json \
  --expected-flashinfer-commit <40-hex-commit> \
  --expected-fla-commit <40-hex-commit> \
  --json /outside/checkouts/h12-phase-a-dual-arch.json
```

A missing case, oracle, timing path, graph receipt, provenance rejection,
activity-route rejection, identity mismatch, or absent architecture is a named
gap rather than evidence that may be filled by averaging another shape or GPU.
Build manifests, per-architecture receipts, and the dual-architecture reduction
are written by same-directory temporary file, `fsync`, and atomic replace so a
preempted job cannot leave a reportable partial JSON file.

[#4351]: https://github.com/flashinfer-ai/flashinfer/pull/4351
