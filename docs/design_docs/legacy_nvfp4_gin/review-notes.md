# Review draft: legacy NVFP4 NCCL GIN

## Description

The legacy NVFP4 MegaMoE backend previously assumed one NVLink domain and
NVSHMEM-backed peer memory. Add explicit `internode=True, ranks_per_node=2`
for the fixed N2R2 qualification geometry, using one NCCL registered root,
real rail/LSA capability checks and ordered BF16 node reduction. The existing
default-off path retains FI sparse, activation, scale and finalization behavior.

The targeted MR !100 import preserves FI changes instead of replacing the
vendor snapshot. A localized donor token-back generation race is corrected:
a late consumer now derives its target from the stable rail generation rather
than waiting for a possibly already-advanced issue counter. Single-rank
metadata and preparation errors are agreed before device communication.

## Related sources

Donor MR !100 merge `f863d7540614bd546a42867c21adbf51d573061a`;
receiver base `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34`.
See [source audit](migration-audit.md), [hunk ledger](evidence/device-hunk-ledger.json)
and [investigation](investigation.md). The vendor baseline is retained in VENDOR.

## Checks

- [x] Pre-commit hooks run on changed files, including mypy and Ruff.
- [x] Standalone host arena/preflight/lifetime contracts: 16 passed.
- [x] Real N2R2 protocol/Torch/transport-byte oracles: 24 calls/rank exact, including sparse holes and NaN invalid weights.
- [x] Real public negative, two-session, recreation, capture and packed-input tests.
- [x] Off EP1/EP2/EP4 routing/reset/graph and BF16/MXFP8 compatibility tests.
- [x] Final sparse-poison/example and off finalizer qualification jobs complete (3576874 / 3576878, exit 0).

Targeted correctness CI is appropriate, with explicit EP1/EP2/EP4 launch
geometry and a separate two-domain N2R2 lane. Suggested files are the new GIN
tests, `test_nvfp4_cutedsl_kernel_vs_reference.py`, the NVFP4/BF16/MXFP8 layer
files, `test_mega_cuda_graph_multirank.py`, and workspace/native-reducer tests.
The graph file also contains next/GenPhase cases; keep selectors explicit.
No bot request, PR publication, push or merge is part of this delivery.

## Reviewer notes

The supported on geometry is deliberately fixed at four ranks, two ranks per
NVLink domain, capacity 32, hidden 7168, 48 total experts, top-k 4 and public
intermediate_size 1536 (runner gate-up width 3072). Standard SwiGLU, early
routing weights, separate BF16 reduction and default PG are required. All
ranks enter serialized control calls in the same order. Subgroups, graphs,
autotune and custom activation/scale/finalization are rejected on mode.

The final adapter caches session-bound launch arguments, binds the current
stream when launching eager GIN, and retains two collective agreements per
steady-state public call. It uses the lean existing non-GIN entry for ordinary
off cases. All24 off performance gates pass against legacy F_BASE. Internode
host overhead is reduced; raw donor comparisons remain inconclusive due to
baseline dispersion, and native/public full-call scopes differ. No next
baseline is used. Final3578176 passes17 strong/public tests per rank and the
retained4+1 off finalizer/cache/graph checks per rank. See
[performance correction](performance-fix-blackwell.md). Production promotion
and broader geometries need separate review.
