# SM120 NVFP4 configuration selection

The selector returns a complete configuration for dispatch/FC1, FC2/token-back,
and combine. The API and runner default to `numerical_contract="ordinary_topk"`.
To permit intermediate owner-grouped BF16 rounding and allow rank-local combine
(RLC), explicitly pass `numerical_contract="owner_bf16"` or use the runner flag
`--numerical_contract owner_bf16`. Enabling RLC through an override alone does
not grant this numerical permission.

`intermediate` is the combined gate/up width. For example, DSV4 with hidden7168,
intermediate6144, top-k6 and384 experts has FC2 reduction width3072.

## Candidate selection

1. Check the calibrated shape, topology and execution scope.
2. Enumerate the576 frozen complete proposals in `candidate_catalog.py`.
3. Apply explicit overrides, deduplicate and filter kernel legality, numerical
   permission, correctness exclusions and estimated shared-memory capacity.
4. Estimate padded GEMM work, weight requests/reuse, worker waves, queue work,
   dispatch and token-back traffic, local RLC staging, and pipeline tail costs.
5. Rank by fitted full-pipeline cost, then apply the scoped paired route-risk
   policies in `robust_policy.py` and `large_policy.py`.

The default catalog is finite, not the Cartesian product of every legal option.
The81-profile historical subset is retained as data but is not the default.
Kernel constructors and generated resources remain authoritative over analytic
resource estimates. There is no online GPU autotuning or route readback.

The general cost domain is110 SMs, same-NUMA EP4/TP1/DP1, T1..24576,
H2048..7168, combined I4096..8192, top-k4..8 and E128..512, subject to alignment
and other validity checks. These are model scope bounds, not exhaustive GPU
validation of every combination. Nondefault build/JIT execution and uncalibrated
shapes use `legacy_heuristic.py` through the compatibility wrapper. That retained
heuristic still contains historical shape rules and measured decode presets.

Without host expert counts, tile work uses a uniform Poisson estimate and
communication uses estimated owner-hit probabilities. Callers may supply an
already-host tuple of dispatch-aggregated counts, one per global expert, with
sum `T * EP * topk`. Tile work then uses the worst contiguous expert-owner group.
This does not reconstruct source-to-owner traffic; communication remains an
estimate. The unknown-route paired policies are disabled when counts are given.

The paired policies predict log latency changes across balanced, power-law and
zero-owner scenarios. They require an equal-weight mean log-P50 gain greater
than0.005, worst log-P50 regression at most0.005 and worst log-P90 regression at
most0.01. These are policy thresholds, not confidence bounds or measured route
frequencies. Their scope is narrower: H=I4096, top-k6..8, E256..288, owner_bf16,
the qualified EP4 topology, density bounds and specific anchor configurations.

`explain_selection` reports the scope, route assumptions, final configuration,
base-cost alternatives and paired-policy decision. Cost components are fitted
service estimates, not isolated kernel timings or hardware bandwidth counters.
The cached selection is deterministic for the same input and overrides.

## Correctness and cache behavior

The published baseline used cache ABI2735. This cleanup candidate uses ABI2737
to invalidate compiled code after correcting `lds_b32_raw` memory effects. The
helper is marked side-effecting with a compiler memory clobber; this does not
replace GPU synchronization. The following baseline fixes remain in place:

- FC1 readers execute an async.shared proxy fence and warp rendezvous before
  releasing shared stages; all four consumer release arrivals remain required.
- N16 generic output stores round up their loop count, with guarded accesses.
- N16 dual consumers wait on the first producer slot of their shared ready block.
- Green Graph nodes use compiled K1/K2 role identities, avoiding classification
  of unrelated clearing kernels by grid size.
- K1 stage overrides are explicit in compile specialization. Tail reclamation
  requires concurrent Green execution.

Independent FP64 arithmetic accuracy, native QMMA byte equality, actual staging
transport, and baseline replay equivalence are separate verdicts. The native
reference does not replace independent arithmetic validation. Direct peer-store
has no independent pre-store staging oracle and reports transport NOT_CHECKED.
See `NATIVE_QMMA_REFERENCE.md` for reference entry points.

`select_collective_execution_plan` accepts agreed host token counts for the EP
group. An entirely empty group can skip launch. A locally empty rank must still
participate when a peer has work. The caller supplies a common compile capacity,
zeros padded route weights and slices outputs to each rank's local token count.
The low-level kernel API continues to require positive dimensions.

## Validation provenance

The published executable sources and coefficients match (apart from one
trailing-space cleanup in the selector)
the ABI2735 frozen candidate used in the2026-11-01/02 campaigns. Host regression
fixtures contain39 resolved configurations from completed13-point Flash,
GLM and DSV4 sweeps. They preserve measured configurations independently of
runtime selection rather than generating their expected values during tests.

The Flash/GLM combination has bounded same-configuration synchronization,
biased-route and delayed-consumer stress evidence. Fresh .16 sweeps include
full K3,50 warmups and400 max-rank CUDA-event samples per point. DSV4
H7168/I6144/E384 has13 native-reference checked performance points and a T16
four-rank FP64 arithmetic check; it does not inherit the Flash/GLM dynamic
stress qualification. Passing measured cases does not establish all-shape or
all-topology correctness or a universal performance improvement over GitLab.

The historical model-version identifier includes `unaccepted`; it is retained
to identify the exact frozen model. Source publication does not broaden the
recorded validation scope. Raw timing/profiling artifacts and experimental
instrumentation remain outside this package.


## Maintenance and reproducibility

The model version string is a frozen training/provenance identifier, not a
release-status flag. Candidate ordering, coefficient precision, arithmetic
ordering and policy risk gates are part of reproducibility. Formatting or
interface refactors must preserve the complete ranked candidate list, features,
costs, selected configuration and explanation, including compatibility fallback
and explicit overrides. The policies receive an explicit `PolicyContext` rather
than looking up the selector module dynamically. The frozen legacy selector is
still required outside the calibrated domain.
