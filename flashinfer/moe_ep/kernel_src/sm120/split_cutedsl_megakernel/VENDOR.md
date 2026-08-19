# SM120 W4A8 Split Kernel Drop

The raw source under `src/` comes from `bangyus/cutedsl_megamoe`, branch
`hanyueh/sm120-mxfp4mxfp8-split`, commit `d6b00fa`.

The historical raw package name remains `moe_sm120_mxfp8_split`; its actual
weight/activation contract is MXFP4 E2M1 x MXFP8 E4M3 with E8M0 K32 scales.
FlashInfer code imports only this package's `shim`, never raw modules directly.

The first integration supports the production same-NUMA `p2p_direct` path.
The standalone cross-NUMA IBGDA transport remains outside this drop until its
transport state is exposed through the framework API.

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

FlashInfer adaptation: `src/token_comm.py` imports `_nanosleep` and
`spin_wait` from this drop's `moe_sm120_mxfp8_split.moe_utils` instead of the
unrelated `moe_nvfp4_swapab` package. The helper implementations are identical.

`runtime/green_context.py` also injects the native Green Context graph as a
child node when an outer CUDA stream capture is active. CUDA does not permit a
plain `cuGraphLaunch` during capture; child-node injection keeps vLLM/FlashInfer
CUDA Graph capture compatible while preserving the K1/K2 SM partitions.
