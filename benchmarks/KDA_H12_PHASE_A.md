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

Candidate content is fixed independently of the receipt by a sealed, canonical
GNU SHA-256 manifest. The manifest and its expected raw-file SHA-256 are package
inputs fixed before the allocation runner starts; neither value may be read from
the receipt. Every line is LF-terminated and has the exact form
`<64-lowercase-hex><two spaces><relative-path>`. The path order and denominator
are exactly these 24 entries:

```text
flashinfer/kda.py
flashinfer/kda_prefill.py
flashinfer/jit/__init__.py
flashinfer/jit/flash_kda.py
flashinfer/aot.py
csrc/kda/flashkda_binding_common.cuh
csrc/kda/flashkda_bf16_fused_m64_binding.cu
csrc/kda/flashkda_bf16_fused_m128_binding.cu
csrc/kda/flashkda_bf16_fused_m128.cu
csrc/kda/flashkda_bf16_fused_m128_n16_binding.cu
csrc/kda/flashkda_bf16_fused_m128_n16.cu
csrc/kda/flashkda_bf16_fused_m128_import_manifest.json
tools/import-cake-flashkda-prefill
benchmarks/bench_recurrent_kda_prefill_h12_phase_a.py
benchmarks/build_flash_kda_phase_a.py
benchmarks/kda_h12_evidence.py
benchmarks/presets/recurrent_kda_prefill_h12_phase_a.json
benchmarks/reduce_kda_h12_phase_a.py
benchmarks/test_kda_h12_evidence.py
benchmarks/KDA_H12_PHASE_A.md
tests/kda/test_recurrent_kda_prefill.py
tests/jit/test_flash_kda_jit.py
tests/jit/test_flash_kda_packed_t1_jit.py
docs/api/kda_prefill.rst
```

The runner rejects a manifest with a different raw digest, path, order, count,
or source-file digest. The per-architecture validator and dual-architecture
reducer bind the same external manifest to the expected FlashInfer commit;
matching source hashes copied into two receipts are not an independent source
identity.

## Frozen denominator

All cases use H=12, K=V=128, BF16 inputs and state, a provided initial state,
in-kernel Q/K L2 normalization, gate calculation, and beta sigmoid, with
`lower_bound=-5.0`. Candidate execution must observe exactly the dedicated
`m128_n16` workspace variant; missing or mixed descriptor variants fail closed.

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
final state at BF16 `atol=rtol=1e-2`. The pinned MoonshotAI/FlashKDA
implementation is the external contract authority. `correctness.passed`,
reportable timing, and per-architecture completion require the candidate to
match that pinned output and full final state.

Two additional comparisons are mandatory: an independent direct recurrence
with a sequence-local chunk-16 BF16 state carrier, and FLA's Triton
implementation. The independent recurrence explicitly rounds four H12
residual intermediates through BF16: the state/K prediction, the
V-minus-prediction delta, sigmoid beta, and the post-beta update carrier. All
six cases must still contain valid full output and final-state comparisons at
the same fixed BF16 tolerance. Their agreement is recorded honestly as
`diagnostic_consensus`, but a diagnostic numerical disagreement does not
override the pinned contract or suppress timing. Missing, malformed, partial,
or tolerance-modified diagnostics still invalidate the receipt. The
independent recurrence normalizes Q and K in FP32 with
`x * rsqrt(sum(x * x) + 1e-6)` and does not insert a BF16 carrier after
normalization. Its FP32 contractions run with TF32 disabled and PyTorch's
`highest` float32 matmul precision, and the prior process policy is restored
even if the oracle raises.

The smaller repository H12 smoke tests use a clean-room chunk-16 recurrence.
It applies the same four BF16 residual carrier boundaries, rounds the FP32
state carrier through BF16 between 16-token chunks, and projects each token's
unrounded updated state. That helper is diagnostic only and is explicitly not
a substitute for the pinned six-case contract.

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
comparison and timing use the native `chunk_kda` callable and default Triton
implementation. Its package version, source paths, source hashes, and Git facts
are reported. Missing FLA, prior FLA import, a non-Git install, or any
tracked/nonignored-untracked FLA change fails closed. Every one of the six cases
must include the FLA output/full-final-state diagnostic plus raw CUPTI timing;
the path is never omitted.

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

The pinned FlashKDA `_fwd_raw` metric is the complete public binding call, not
a recurrence-only kernel metric. Its binding materializes
`beta_2d.t().contiguous()` with one ATen direct-copy kernel. Packed layouts then
launch `_flash_kda_build_tile_prefix`; fixed layouts omit that step. Both
layouts launch `_flash_kda_fwd_prepare` followed by
`_flash_kda_fwd_recurrence`. Schema v11 requires exactly that ordered,
nonoverlapping, one-logical-launch-per-activity route, with every activity
matching exactly one expected stage marker. The public-semantics adapter requires
the identical raw route followed by exactly one full-final-state D2D copy with
the expected byte count. This keeps all pinned work inside its CUPTI span and
prevents a recurrence-only interpretation of the peer baseline.
Every CUPTI runtime or driver API invocation has a unique correlation ID. Each
contributing correlation must bind exactly one API record to exactly one GPU
activity, and that API record must start no later than the activity. Duplicate
IDs, orphan correlations, and GPU-before-API evidence fail closed.

The prepared recurrence nested in each public sample must reuse the exact outer
recurrence activity and its single outer runtime/driver launch record. Its
metrics are recomputed from that activity. A separately
timed recurrence with matching names or correlations is not accepted as a view
derived from the public invocation.

The sealed allocation launcher generates a fresh 128-bit run challenge and
persists it in an allocation record outside the receipt before invoking the
runner. GPU evidence requires that challenge on the command line rather than
choosing a receipt-local ID, and both the per-architecture validator and
dual-architecture reducer require the independently recorded launcher challenge
as the expected value.

After the runner atomically finishes writing a per-architecture receipt, the
launcher hashes the receipt's exact raw bytes and persists that SHA-256 outside
the receipt alongside the allocation challenge. The reducer hashes the supplied
raw receipt bytes itself and compares them with the two independently recorded
launcher digests. Its `--expected-sm100a-receipt-sha256` and
`--expected-sm103a-receipt-sha256` values must come from those launcher records;
they must never be copied from a receipt field or recomputed from the receipt
being submitted for reduction.

The receipt byte encoding is canonical and fail-closed: UTF-8 JSON, recursively
sorted object keys, two-space indentation, no NaN/Infinity, and exactly one
terminal LF. Before semantic validation, the raw receipt digest must equal the
digest reconstructed from that encoding. Reordered or duplicate keys, compact
JSON, CRLF/BOM encodings, and missing terminal newlines are therefore rejected
even when a permissive JSON parser would produce the same object.

Each raw sample also preserves the three CUPTI-clock CPU bracket timestamps
(`start_ns`, `submitted_ns`, and `synchronized_ns`) and an exact trace scope
covering the frozen preset, case/shape fingerprint, path, block/order, and
sample. Submission and synchronized-E2E metrics are recomputed from that raw
bracket. The per-architecture validator then reconstructs the canonical
960-sample execution ledger (six cases, two alternating-order blocks, four
paths, and twenty samples), requires nonoverlapping brackets in execution
order, and rejects reused GPU correlations or runtime/driver launch records
anywhere in the process-wide receipt. Each trace also records and requires zero
CUPTI dropped activities. Copied samples, cross-path activity
replay, and timing blocks substituted from another shape therefore fail closed.
Dual-architecture promotion schema v4 additionally requires distinct run IDs and rejects
an identical raw timing ledger even if its trace labels were rewritten, so one
architecture receipt cannot be replayed as the other architecture's evidence.

## CUDA Graph and dual-architecture gates

Every GPU evidence run invokes exactly:

```bash
python -m pytest -q \
  tests/kda/test_recurrent_kda_prefill.py::test_frozen_prefill_non_aligned_heads_graph_refreshes_beta
```

This is the H6/H12 changed-beta CUDA Graph regression at source lines
1090–1164. It compares a captured replay bitwise against an independent eager
launch with separate tensors and a separate workspace, for both output and full
final state, and proves that changing beta changes the replayed result. The
receipt records the exact source line range, command, node, parameterization,
source hash, return code, stdout, and stderr. It preserves the venv interpreter
path actually executed and separately records its strict canonical path for
runtime-provenance comparison. A failure writes a non-complete receipt and
stops the evidence run.

A successful single-GPU receipt sets only
`complete_per_arch_denominator=true`. It is not a promotion claim. The reducer
requires exactly one SM100a and one SM103a receipt with matching frozen preset,
FlashInfer commit/source hashes bound to the external 24-entry manifest, exact
raw receipt digests bound to their independent launcher records, pinned
FlashKDA source/package identity, clean FLA commit/source hashes, graph-test
source, all six ordered cases, the pinned output/full-state contract, both
mandatory structurally valid diagnostic receipts, and all four CUPTI timing
paths. Diagnostic disagreement remains visible in each per-architecture receipt
but is not a promotion gate. Only then does the reducer emit
`promotion_complete_dual_arch=true`. It computes no cross-shape aggregate.

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
# The sealed launcher generates and records this before starting the runner.
evidence_run_id=<32-lowercase-hex-launcher-challenge>
source_manifest=/sealed/package/flashinfer-kda-source-sha256s.txt
source_manifest_sha256=<sealed-64-lowercase-hex-manifest-digest>
python benchmarks/bench_recurrent_kda_prefill_h12_phase_a.py \
  --flash-kda-source-dir /absolute/path/to/FlashKDA \
  --flash-kda-build-manifest /outside/checkouts/flash-kda-build-sm100a.json \
  --expected-source-manifest "${source_manifest}" \
  --expected-source-manifest-sha256 "${source_manifest_sha256}" \
  --warmup-iters 5 \
  --repeat-iters 20 \
  --blocks 2 \
  --evidence-run-id "${evidence_run_id}" \
  --json /absolute/path/to/h12-phase-a.json
```

After both complete runs, use the source manifest and digest fixed by the sealed
package plus the run challenges and raw-receipt digests persisted in the two
independent launcher records. Freeze the expected FlashInfer and FLA commits at
the CLI boundary and reduce the two receipts:

```bash
python benchmarks/reduce_kda_h12_phase_a.py \
  --sm100a /outside/checkouts/h12-phase-a-sm100a.json \
  --sm103a /outside/checkouts/h12-phase-a-sm103a.json \
  --expected-flashinfer-commit <40-hex-commit> \
  --expected-flashinfer-source-manifest \
    /sealed/package/flashinfer-kda-source-sha256s.txt \
  --expected-flashinfer-source-manifest-sha256 \
    <sealed-64-lowercase-hex-manifest-digest> \
  --expected-fla-commit <40-hex-commit> \
  --expected-sm100a-run-id <recorded-sm100a-32-hex-challenge> \
  --expected-sm103a-run-id <recorded-sm103a-32-hex-challenge> \
  --expected-sm100a-receipt-sha256 \
    <launcher-recorded-sm100a-raw-receipt-sha256> \
  --expected-sm103a-receipt-sha256 \
    <launcher-recorded-sm103a-raw-receipt-sha256> \
  --json /outside/checkouts/h12-phase-a-dual-arch.json
```

A missing case, pinned oracle, diagnostic receipt, timing path, graph receipt,
provenance rejection, activity-route rejection, identity mismatch, or absent
architecture is a named gap rather than evidence that may be filled by
averaging another shape or GPU.
Build manifests, per-architecture receipts, and the dual-architecture reduction
are written by same-directory temporary file, `fsync`, and atomic replace so a
preempted job cannot leave a reportable partial JSON file.

[#4351]: https://github.com/flashinfer-ai/flashinfer/pull/4351
