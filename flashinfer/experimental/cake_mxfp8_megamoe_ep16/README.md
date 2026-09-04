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
- balanced immutable routing prepared with the session
- exact compute capability 10.3 and NVSHMEM symmetric memory

The backend does not participate in automatic routing, autotuning, trace
apply, or AOT packaging. It provides no compatibility guarantee while it is
experimental.

A runnable 16-rank example is provided in
`examples/experimental/cake_mxfp8_megamoe_ep16.py`.
