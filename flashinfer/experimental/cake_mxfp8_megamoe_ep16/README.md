# MXFP8 MegaMoE EP16

This directory contains an experimental JIT-only backend for an MXFP8
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
- 1 through 64 runtime tokens per rank
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

## Tile policy and reusable weights

The existing session factory accepts `tile_n="mixed"` (the default throughput
route), `tile_n=16`, or `tile_n=32` (uniform arithmetic tiles). The independent
`return_protocol` setting accepts `"cta0"`, `"all_cta"`, or `"auto"` (default).
Keep the uniform tile policy fixed across sessions when testing batch
invariance; the mixed route does not provide that guarantee.

For example, `CakeMxfp8MegaMoeEp16(weights, topk_ids, tile_n=32,
return_protocol="cta0")` selects the uniform N32 route. Reuse the same
preprocessed `weights` when constructing sessions for other batch extents.
Each session owns its exact live input/output extent and routing; neither
weights nor the routing tensor may be mutated while the session is in use.
Changing the batch extent requires a new session, not a new compiled kernel.
The JIT cache is keyed by physical tile/return policy and source identity.
All ranks must construct sessions with the same extent and resolved policy.
Routing and prepared weight tensors must have PyTorch version counters (create
them outside `torch.inference_mode()`); inference tensors are rejected during
collective session admission so mutation tracking is not silently disabled.
Forwards may run inside `torch.inference_mode()` with those persistent tensors.
All ranks must issue matching valid forwards; validation does not add a
collective to the two-kernel hot path.

The backend does not participate in automatic routing, autotuning, trace
apply, or AOT packaging. It provides no compatibility guarantee while it is
experimental.

A runnable 16-rank example is provided in
`examples/experimental/cake_mxfp8_megamoe_ep16.py`.
