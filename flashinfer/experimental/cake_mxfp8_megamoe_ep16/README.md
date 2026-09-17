# MXFP8 MegaMoE EP16

This directory contains an experimental JIT-only backend for a fixed MXFP8
MegaMoE configuration on exact SM103a devices.

The public entry points are
`flashinfer.moe_ep.CakeMxfp8MegaMoeEp16` and
`flashinfer.moe_ep.preprocess_cake_mxfp8_megamoe_ep16_weights`. Both are
experimental APIs and emit `ExperimentalWarning` when called. Calling them is
the explicit opt-in; no environment variable is required.

## Supported contract

- 16 expert-parallel ranks
- 512 global experts and top-k 8
- hidden size 3072 and intermediate size 5120
- 16, 32, or 64 tokens per rank
- BF16 activations and outputs with MXFP8 expert weights
- immutable routing prepared with the session, with at most 64 routes assigned
  to any expert across all ranks
- exact compute capability 10.3 and NVSHMEM symmetric memory

Session construction materializes the eight fixed-address TMA descriptors with
one setup kernel and synchronizes before returning. Each forward then uses
exactly two kernels: the first publishes and dispatches BF16 rows, executes FC1
and FC2, and returns one contribution per route; the second reduces the eight
route contributions in a fixed order. CUDA Graph capture is not supported.
Calls on one session must be serialized on a single CUDA stream. Recreate the
session after 14,913,080 forwards, before its signed grid-counter epoch would
overflow.

The backend does not participate in automatic routing, autotuning, trace
apply, or AOT packaging. It provides no compatibility guarantee while it is
experimental.

A runnable 16-rank example is provided in
`examples/experimental/cake_mxfp8_megamoe_ep16.py`.
