# SM120 W4A8 Split Kernel Drop

The raw source under `src/` comes from `bangyus/cutedsl_megamoe`, branch
`hanyueh/sm120-mxfp4mxfp8-split`, commit
`6aa25a07fe14afcf0c05323139551cd0533c07b0`.

This update includes N64 compact SFB loads, cached-scale GPU-acquire visibility,
and shape-bounded local-combine prefill tuning (`KERNEL_CACHE_ABI=35`).
FlashInfer-specific child-graph capture and vendored import adaptations are
preserved when applying upstream kernel changes.

The local frontend experiment adds 384/512/1024/2048/4096 row buckets between
the existing decode buckets and caller-provided workspace capacity. Buckets
are prepared lazily and remain part of the kernel and frontend graph cache
keys; they are not additional workspace capacities.

The raw package is named `moe_sm120_mxfp4mxfp8_split`; its weight/activation
contract is MXFP4 E2M1 x MXFP8 E4M3 with E8M0 K32 scales.
FlashInfer code imports only this package's `shim`, never raw modules directly.

The first integration supports the production same-NUMA `p2p_direct` path.
The standalone cross-NUMA IBGDA transport remains outside this drop until its
transport state is exposed through the framework API.

The same-NUMA drop includes the production dispatch rank cache and owner-local
combine policies. Their caller-stream reset, graph launch events, and finalizer
form one replay ownership protocol and must be updated together with the raw
kernel source.

Multi-rank execution requires NVSHMEM 3.7.0 and the matching
`nvshmem4py-cu13` 0.3.1 Python bindings. Single-rank execution can set
`MEGA_NO_DIST=1` and does not import NVSHMEM. The backend must be warmed up
collectively before an outer CUDA Graph capture begins.

Layers with identical geometry share activation, routing, K1/K2 scratch, and
combine buffers through FlashInfer's process-level workspace pool. Each layer
keeps a separate native Green Context graph because the graph captures that
layer's weight pointers. Layer execution is sequential, so the graphs safely
reuse the same physical buffers without multiplying the symmetric heap by the
model's MoE layer count.

`runtime/green_context.py` also injects the native Green Context graph as a
child node when an outer CUDA stream capture is active. CUDA does not permit a
plain `cuGraphLaunch` during capture; child-node injection keeps vLLM/FlashInfer
CUDA Graph capture compatible while preserving the K1/K2 SM partitions.
