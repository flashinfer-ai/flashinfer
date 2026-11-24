# SM120 NVFP4 Split Kernel Drop

The raw source under `moe_sm120_nvfp4_split/` comes from `bangyus/cutedsl_megamoe`, branch
`hanyueh/sm120-nvfp4-split`, commit
`b2fd4cac1d28d6a9be0cdfaf303ccd072ce21f54`. The FlashInfer
delivery uses ABI `2814542011117` (standalone source ABI `281454201`).
This revision includes continuous decode policies, M128 workers, utility
defaults, DATA TMA eviction hints and the NCCL memory/transport runtime.
It adds the measured anchor table (`tuned_table.json`, DSV4.1-Flash and
GLM5.3-Flash, EP4 in one NUMA node and EP8 across two) and the residency
knobs `k1_ctas_per_sm` / `k2_launch_factor`; kernel cache keys include them.
The FlashInfer frontend scales its K1/K2 launch grids by these knobs.
This drop (`b2fd4ca`) adds EP16 across two hosts: a GPU is identified by
(host, PCI bus) and only GPUs on the same host and NUMA node are P2P peers,
so the other host's GPUs are cross peers served by `p2p_split` like the other
NUMA node's. It adds EP16 anchors, the K1/K2 weight L2 prefetch knob
`weight_prefetch_ahead` (0/4/8 K tiles) and the forwarder split `xnuma_fsplit`
(1/2/4 CTAs per cross source), a fused record+commit GIN put and deferred GIN
quiet in the sender, and launch-phase / sender / scatter diagnostics.
FlashInfer adds `fence.proxy.async.shared::cta` before every K2 consumer stage
release. Without it, a stage released by lane 0 could be refilled by the next
TMA (async proxy) before the warp's last ldmatrix (generic proxy) had read it;
with the K2 weight prefetch on 3-stage pipelines (DSv4.1 EP4 T7/T16, EP8 decode
anchors) the L2-hit refill overwrote the first rows of the stage, giving
intermittent wrong rows. Cache ABI `2814542011117` separates kernels compiled
before this change.
FlashInfer keeps one `xnuma_fsplit` per workspace: its p2p_split buckets share
one epoch and cumulative commit words, and the scatter waits for
`fsplit * epoch` commits, so the value most p2p_split buckets selected is used
for all of them (`MEGAMOE_XNUMA_FSPLIT` overrides it). The sender keeps the
FlashInfer per-tensor input puts and sends only the record content.

On an EP spanning NUMA nodes the frontend can run the `p2p_split` backend
(`transport/xnuma.py`): same-NUMA peers use direct P2P and every cross-NUMA
peer is served through a local shadow segment of the workspace arena that
GIN keeps in sync. The choice is made per compile bucket when the workspace
is allocated: without a `comm_backend` knob a bucket follows the anchor
table (or its rule for untabulated shapes), `knobs={"comm_backend":
"p2p_split"}` selects it for every eligible bucket and `"gin"` disables it.
Only buckets up to `FLASHINFER_MEGAMOE_P2P_SPLIT_MAX_TOKENS` (default 384,
the largest decode bucket) qualify, because the xnuma return buffers grow
with tokens x top-k x hidden x cross-NUMA peers (DSV4.1-Flash: about
0.38 GB per rank at 384 tokens, 4 GB at 4096); larger buckets run `gin`.
A bucket also needs at most 64 local experts, the 8-warp K2 and per-slot
combine (rank-local combine is disabled for it), otherwise it runs `gin`.

FlashInfer adaptation of p2p_split: the shadows cover only the arena prefix
holding the inputs, the output and the p2p_split buckets' combine/shared
buffers (allocated first), not the whole arena. All p2p_split buckets and
layers of a workspace share one device epoch and one control region placed
for the largest such bucket: GIN commits are cumulative signal-adds keyed by
the source word, so every commit word keeps a fixed address. Inputs are sent
as separate per-tensor puts sized to the bucket. Under an outer CUDA graph
the epoch bump and sender are captured before the child graph, and the
return scatter after the K2 finalizer and before K3.
The standalone `tools/sm120_nvfp4_tuning` scripts are not vendored.

FlashInfer fix in the ordinary top-k reduction (K3,
`kernel_combine_reduce.py`): K2 writes no `combine_output` row for a negative
(unrouted) top-k slot, so K3 previously summed whatever that row held from an
earlier call whenever a valid token had such a slot. The frontend now passes
the staged `topk_ids` and K3 skips negative slots (fully unrouted padding rows
and the rank-local combine path were already correct). This is a deviation
from the standalone source, which has the same unmasked reduction.
The FlashInfer-specific EP8 GIN adapter and graph integration are retained
alongside the standalone source; this is not a byte-identical standalone drop.
See [DECODE_DEFAULTS.md](DECODE_DEFAULTS.md) for activation ranges and validation limits. The private `moe_sm120_nvfp4_split/support/common/` and `support/src/`
directories retain the PR's framework adaptations without exposing generic
`common` or `src` packages that could shadow other backends. `runtime/green_context.py` and its tests retain
the graph-aware FlashInfer adapter rather than the standalone runner wrapper.

This drop includes dispatch rank caching, its caller-stream reset ordering,
128-byte tensor-map descriptors, and the large-EP4 rank-local combine policy.
FlashInfer preserves its outer CUDA Graph child-node adaptation.

The raw package is named `moe_sm120_nvfp4_split`. Its weight and activation
contract is packed E2M1 NVFP4 x NVFP4 with one E4M3 scale per 16 K elements,
FP32 accumulation, and BF16 output. FlashInfer imports it only through the
`flashinfer.moe_ep.shims.sm120.nvfp4_split_cutedsl` adapter; backend code does
not import benchmark runners.

The adapter supports same-NUMA `p2p_direct` and cross-NUMA `gin` transport.
At host selection, FlashInfer `gin` maps to the native policy name
`gin_hybrid`, then maps back for execution. Local peers use P2P; remote
peers use NCCL GIN. EP8 retains disjoint K1/TX/RX/K2 Green Contexts.
The CUDA 13.0 driver path splits equal SM units once and assembles the
requested disjoint resources; newer drivers use the structured split API.
Dynamic GIN window registration preserves live device signal-shadow state.
The host bridge and its pinned build manifest are shipped together;
the existing GIN device bitcode is unchanged.

Multi-rank execution requires the pinned NCCL 2.32.3 runtime, selected through
`MEGAMOE_NCCL_ROOT` or `nvidia-nccl-cu13==2.32.3`. Keep `NCCL_CUMEM_ENABLE=1` set before initializing the communicator;
frameworks such as SGLang may otherwise override this setting to zero, which is
incompatible with the NCCL device communicator used here. The backend requests
only Torch distributed bootstrap. It allocates one NCCL-registered arena per workspace
and derives direct-P2P pointers through LSA. It does not initialize NVSHMEM.
By default the runtime rejects processes that load NVSHMEM. The explicit
`MEGAMOE_ALLOW_TORCH_NVSHMEM_LOAD=1` compatibility option permits only
the two recognized Torch-linked library paths, with a warning. MegaMoE
allocation and transport still use NCCL; the process is not NVSHMEM-free.
Other NVSHMEM modules/libraries remain rejected. Single-rank execution can set `MEGA_NO_DIST=1`. The backend
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
