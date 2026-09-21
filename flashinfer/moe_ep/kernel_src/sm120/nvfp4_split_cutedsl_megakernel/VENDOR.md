# SM120 NVFP4 Split Kernel Drop

The raw source under `src/` comes from `bangyus/cutedsl_megamoe`, branch
`hanyueh/sm120-nvfp4-split`, commit
`d330393eb6dbf8398690d9c5e426c336d3d15f0d` (`KERNEL_CACHE_ABI=20`).

This drop includes dispatch rank caching, its caller-stream reset ordering,
128-byte tensor-map descriptors, and the large-EP4 rank-local combine policy.
FlashInfer preserves its outer CUDA Graph child-node adaptation.

The raw package is named `moe_sm120_nvfp4_split`. Its weight and activation
contract is packed E2M1 NVFP4 x NVFP4 with one E4M3 scale per 16 K elements,
FP32 accumulation, and BF16 output. FlashInfer imports only this package's
`shim`; backend code does not import benchmark runners.

The initial integration supports the production same-NUMA `p2p_direct` path.
The standalone cross-NUMA IBGDA transport remains outside this drop until its
transport state is exposed through the framework API.

Multi-rank execution requires NVSHMEM 3.7.0 and matching `nvshmem4py-cu13`
0.3.1 bindings. Single-rank execution can set `MEGA_NO_DIST=1`. The backend
must be warmed up collectively before an outer CUDA Graph capture begins.

Layers with identical geometry share activation, routing, K1/K2 scratch, and
combine buffers through FlashInfer's process-level workspace pool. Each layer
keeps a separate native Green Context graph because the graph captures that
layer's weight pointers.

Dispatch-cache and rank-local scratch reset on the caller stream before the
Green graph's launch-ready event; launch-done orders the next invocation.
Owner-ready flags and their epoch counter are shared by layers using the same
execution bucket. The epoch advances with a device operation, including on
every outer CUDA Graph replay, rather than capturing a host epoch constant.
Peer-written readiness is not independently cleared by the frontend.

The standalone heuristic selects rank-local combine for eligible same-NUMA
EP4/TP1 workloads with dispatch caching and at least 256 expected rows per
expert. It uses the matching owner-partial K3, not ordinary top-k K3. Each
owner partial is rounded to BF16 before final FP32 accumulation, so this path
is deterministic but is not bit-identical to direct top-k-slot reduction.
Set `knobs={"rank_local_combine": False}` to retain direct top-k reduction;
`knobs={"dispatch_rank_cache": False, "rank_local_combine": False}` disables
both optimizations. Compile buckets remain separate from workspace capacity.

FlashInfer adaptation: `runtime/green_context.py` uses the graph-capture-aware
implementation from the SM120 W4A8 integration, allowing the native Green
Context graph to be inserted as a child node during an outer CUDA capture.
