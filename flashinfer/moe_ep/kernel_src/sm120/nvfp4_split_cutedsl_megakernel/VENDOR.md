# SM120 NVFP4 Split Kernel Drop

The raw source under `src/` comes from `bangyus/cutedsl_megamoe`, branch
`hanyueh/sm120-nvfp4-split`, commit
`d0762b1a361c69640885f64e6529e5fec4fa7ef4` (ABI 2812).
This revision includes qualified compact compute policies and the NCCL-only
memory/transport runtime. The private `moe_sm120_nvfp4_split/support/common/` and `support/src/`
directories retain the PR's framework adaptations without exposing generic
`common` or `src` packages that could shadow other backends. `runtime/green_context.py` and its tests retain
the graph-aware FlashInfer adapter rather than the standalone runner wrapper.

This drop includes dispatch rank caching, its caller-stream reset ordering,
128-byte tensor-map descriptors, and the large-EP4 rank-local combine policy.
FlashInfer preserves its outer CUDA Graph child-node adaptation.

The raw package is named `moe_sm120_nvfp4_split`. Its weight and activation
contract is packed E2M1 NVFP4 x NVFP4 with one E4M3 scale per 16 K elements,
FP32 accumulation, and BF16 output. FlashInfer imports only this package's
`shim`; backend code does not import benchmark runners.

The initial integration supports the production same-NUMA `p2p_direct` path.
The cross-NUMA GIN transport is present in the vendored source but is not exposed
by this same-NUMA framework adapter.

Multi-rank execution requires the pinned NCCL 2.32.3 runtime, selected through
`MEGAMOE_NCCL_ROOT` or `nvidia-nccl-cu13==2.32.3`. Keep `NCCL_CUMEM_ENABLE=1` set before initializing the communicator;
frameworks such as SGLang may otherwise override this setting to zero, which is
incompatible with the NCCL device communicator used here. The backend requests
only Torch distributed bootstrap. It allocates one NCCL-registered arena per workspace
and derives direct-P2P pointers through LSA. It does not initialize NVSHMEM.
The runtime rejects processes that already load NVSHMEM, including incompatible
PyTorch builds; use a clean compatible PyTorch build. Single-rank execution can set `MEGA_NO_DIST=1`. The backend
must be warmed up collectively before an outer CUDA Graph capture begins.

Layers with identical geometry share activation, routing, K1/K2 scratch, and
combine buffers through FlashInfer's process-level workspace pool. Each layer
keeps a separate native graph because the graph captures that layer's weight
pointers. Graphs using the same device and K1/K2 SM partition share a
workspace-owned Green Context pair, released only after all frontend graphs.
Layer-dependent input calibration is consumed by each layer's fused stager,
not by workspace allocation, and therefore does not split the workspace pool.
The selected bucket's padding IDs are reset to -1, including when bucket views
shrink and grow; activation and routing-weight padding need not be cleared.

Dispatch-cache and rank-local scratch reset on the caller stream before the
Green graph's launch-ready event; launch-done orders the next invocation.
Owner-ready flags and their epoch counter are shared by layers using the same
execution bucket. The epoch advances with a device operation, including on
every outer CUDA Graph replay, rather than capturing a host epoch constant.
Peer-written readiness is not independently cleared by the frontend.

The default `numerical_contract="ordinary_topk"` preserves ordinary top-k
reduction order and forbids rank-local combine. Explicit
`numerical_contract="owner_bf16"` permits the selector to choose owner partials:
each owner partial is rounded to BF16 before final FP32 accumulation. This
contract is deterministic but not bit-identical to ordinary top-k reduction.
It is included in the workspace pool key. `knobs={"rank_local_combine": False}`
can disable owner partials under either contract. Compile buckets remain
separate from workspace capacity; public intermediate width is post-SwiGLU.

FlashInfer adaptation: `runtime/green_context.py` uses the graph-capture-aware
implementation from the SM120 W4A8 integration, allowing the native Green
Context graph to be inserted as a child node during an outer CUDA capture.
Kernel-node rebinding resolves `CUkernel` from retained compiled libraries and leaves `CUfunction`
unset: a non-null `CUfunction` makes CUDA ignore the requested context.
The wrapper verifies the retained context before instantiation so persistent
K1/K2 grids cannot silently lose their disjoint SM partitions.

The SM120 fused input stager opts into the CUDA FP4 fast-math encode-scale
order (`rcp(sf * rcp(norm))`) to preserve the old CUDA-prequantized path at
E2M1 rounding boundaries. Other MegaMoE backends retain their legacy recipe.
Offset/misaligned-width inputs use the CUDA quantizer fallback. Padding IDs
remain -1; only scale-column alignment padding requires zeroing.

First allocation of a bucket's execution storage drains in-flight device work
before suballocating shared execution storage. This avoids mixing the allocator
with peer-dependent Green Context graphs. Cached execution/storage paths add no
synchronization; cold allocation remains forbidden during outer graph capture.

The adapter retains compiled kernel owners, identifies K1/K2 by library function
identity (not CTA counts), and preserves launch attributes while rebinding.
The cross-rank finalizer and K3 execute on the primary stream after the mixed-context child graph, including
outer CUDA Graph capture. The workspace-owned two-part Green Context pair is
retained for sequential layer reuse and closes after all layer graphs.

The finalizer is deliberately outside the mixed-context child: embedding it
stalled the first N16/dual-N8 decode launch with pooled Green Context resources.
Both post-graph callbacks bind the current stream, so outer capture orders
finalizer, K3 and the next layer's reset on its caller stream.

K1/K2 and their optional drain/finalizer share compiled code per workspace
bucket. Runtime arguments and graph bindings remain per frontend. This avoids
recompiling the persistent drain for every output graph; the shared owners live
until all referring graphs have closed.

The SM120 shim owns its fused quantization stager and imports host weight
helpers from this NVFP4 package. It does not bootstrap the SM100 source tree.
The support `src/inputs_process.py` carries the CUDA-compatible encode-scale
option formerly supplied by the shared stager; the default recipe is unchanged.
Support-module imports also resolve sleep/wait and reference helpers from the
local NVFP4 package, avoiding accidental dependency on an SM100 drop.

When an outer capture allocates a fresh output address, the backend reuses its
warmed workspace graph and copies the live rows to that output. Prepared fixed
outputs retain their direct graph bindings. The public workspace-view option
provides a zero-copy view with main's borrowed-output lifetime.
The fused input stager follows main's CUDA quantizer when a nonzero block's
E4M3 scale rounds to zero: its encoding scale is zero as well.

Shared arena capacity is planned collectively for the prefill bucket and all
reachable decode buckets before allocating workspace storage. Every peer-visible
tensor is a slice of that arena: the kernel's single peer-offset table therefore
remains valid for activation, routing, readiness, scratch and output buffers.
Independent workspace handles own independent NCCL communicators and arenas;
layer pooling retains the arena until all associated frontends are destroyed.
Packed FP4 weight interleave uses byte views so it is bit-preserving on PyTorch
versions without native FP4 `copy_` support. CuTe DLPack conversion uses the
vendored low-precision wrapper where Torch does not export FP4/FP8 DLPack.
