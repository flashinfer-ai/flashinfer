# MXFP8 MegaMoE EP16

This directory contains experimental JIT-only CUDA and CuTe-DSL backends for a
fixed MXFP8 MegaMoE configuration on exact SM103a devices.

The public entry points are
`flashinfer.moe_ep.CakeMxfp8MegaMoeEp16` and
`flashinfer.moe_ep.preprocess_cake_mxfp8_megamoe_ep16_weights`. Both are
experimental APIs and emit `ExperimentalWarning` when called. Calling them is
the explicit opt-in; no environment variable is required.

## Backend selection

The existing CUDA implementation remains the default. Select CuTe-DSL with the
keyword-only argument `backend="cute_dsl"` when constructing the session:

```python
weights = preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)
session = CakeMxfp8MegaMoeEp16(weights, topk_ids, backend="cute_dsl")
output = session.run(
    hidden_states, topk_ids, topk_weights, out=session.workspace_output
)
```

Omit `backend` or pass `backend="cuda"` for the CUDA implementation. Every rank
in the process group must choose the same backend. This agreement is checked
collectively before symmetric workspace allocation. There is no automatic
backend selection or fallback.

CuTe-DSL requires the appropriate FlashInfer CUDA extra (`cu12` or `cu13`) and
its CUTLASS DSL dependency. The CuTe modules are loaded lazily only when this
backend is selected. They are compiled for `sm_103a` through FlashInfer's CuTe
JIT cache; if `CUTE_DSL_ARCH` is set, it must also select `sm_103a`. Initial
compilation belongs to session setup, not the forward path.

## Supported contract

- 16 expert-parallel ranks
- 512 global experts and top-k 8
- hidden size 3072 and intermediate size 5120
- 16, 32, or 64 tokens per rank
- BF16 activations and outputs with MXFP8 expert weights
- immutable routing prepared with the session, with at most 64 routes assigned
  to any expert across all ranks
- exact compute capability 10.3 and NVSHMEM symmetric memory

CUDA session construction materializes eight fixed-address TMA descriptors with
one setup kernel. CuTe-DSL prepares tensor-map dimensions and strides on the
host and passes descriptors by value through the compiled host entry.

Both implementations synchronize before session construction returns. Each
forward then uses exactly two kernels: the first publishes and dispatches BF16
rows, executes FC1 and FC2, and returns one contribution per route; the second
reduces the eight route contributions in a fixed order. Neither implementation
allocates device storage in the forward path. CUDA Graph capture is not supported.
Calls on one session must be serialized on a single CUDA stream. Recreate the
session after 14,913,080 forwards, before its signed grid-counter epoch would
overflow.

The backend does not participate in automatic routing, autotuning, trace
apply, or AOT packaging. It provides no compatibility guarantee while it is
experimental.

A runnable 16-rank example is provided in
`examples/experimental/cake_mxfp8_megamoe_ep16.py`.

```bash
torchrun --nproc-per-node=16 examples/experimental/cake_mxfp8_megamoe_ep16.py \
    --tokens 16 --backend cute_dsl
```

Use the appropriate multi-node `torchrun` rendezvous options when the 16 ranks
span multiple hosts. The example submits one forward and does not measure
performance.

The experimental correctness test covers both backends at all three supported
token counts, using an analytical sparse-weight reference with mixed-width
tails and the unchanged per-expert capacity limit. Each session runs 32
forwards with paired inputs to check repeated results and bank reuse. These
focused checks are not a general performance or broader-shape guarantee.
