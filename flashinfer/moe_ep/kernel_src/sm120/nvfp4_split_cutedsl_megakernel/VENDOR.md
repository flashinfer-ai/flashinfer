# SM120 NVFP4 Split Kernel Drop

The raw source under `src/` comes from `bangyus/cutedsl_megamoe`, branch
`hanyueh/sm120-nvfp4-split`, commit `0047b7a`.

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

FlashInfer adaptation: `runtime/green_context.py` uses the graph-capture-aware
implementation from the SM120 W4A8 integration, allowing the native Green
Context graph to be inserted as a child node during an outer CUDA capture.
