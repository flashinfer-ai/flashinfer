# SM120 NVFP4 x NVFP4 Split MegaMoE

This package implements the SM120 split MegaMoE path with packed E2M1
NVFP4 weights and activations, per-16 E4M3 scale factors, FP32 accumulation,
and BF16 output.

The execution pipeline has three phases:

1. `kernel_dispatch_fc1.py` dispatches routed NVFP4 activation rows, executes
   FC1, applies `fc1_alpha`, SwiGLU and the selected top-k weight, then
   requantizes the result to block-16 NVFP4.
2. `kernel_fc2_combine.py` consumes FC1 ready bundles, executes FC2, applies
   `fc2_alpha`, converts to BF16, and writes each routed partial output back to
   its source rank.
3. `kernel_combine_reduce.py` reduces the top-k BF16 partial outputs.

K1 and K2 execute concurrently in disjoint Green Contexts. K1 publishes FC1
output ready state bundle-by-bundle, so K2 can start before K1 finishes the
whole expert pool. Same-NUMA EP uses direct P2P activation pull and direct
peer-store combine. Cross-NUMA EP keeps local peers on P2P and sends only
cross-NUMA dispatch and return traffic through NCCL GIN. P2P allocations and peer mappings use NCCL LSA; no NVSHMEM initialization, allocator, host library or device bitcode is required.

## Data contract

- Activation and FC1/FC2 weights: packed `torch.float4_e2m1fn_x2` (two logical
  E2M1 elements per byte).
- Activation and weight scales: one `torch.float8_e4m3fn` value per 16 K
  elements.
- FC1 handoff: packed E2M1 data plus per-16 E4M3 scales.
- Accumulator: FP32; final partial and reduced output: BF16.
- `hidden` must be divisible by 32 and `intermediate` by 64.

The numerical order is:

```text
K1: FP32 accumulator -> fc1_alpha -> SwiGLU -> top-k weight
    -> per-16 E4M3 scale + packed E2M1
K2: FP32 accumulator -> fc2_alpha -> BF16 -> source-rank output
K3: BF16 top-k reduction
```

`fc1_alpha`, `fc2_alpha`, and `fc1_norm_const` are explicit per-expert inputs.
They are part of the NVFP4 numerical contract and must not be folded into a
different stage without updating the reference implementation.

## Runtime environment

- Python 3.12, `nvidia-cutlass-dsl==4.6.0`, SM120 GPUs.
- NCCL `nvidia-nccl-cu13==2.32.3`; the packaged device and host bridge
  artifacts are version/hash checked and fail closed on mismatch.
- CUDA toolkit 13.x for the bridge, a compatible driver and an RDMA/GDR stack
  for cross-NUMA GIN. Same-NUMA traffic uses NCCL LSA peer mappings.
- The dependency-removal qualification uses official `torch==2.8.0+cu128`
  and its CUDA 12.8 dependencies. The local low-precision DLPack adapter
  preserves packed FP4/FP8 storage without requiring a newer Torch export.
  Some newer vendor Torch builds themselves link NVSHMEM; using such a
  build does not make the process NVSHMEM-free even though MegaMoE does not
  call it. Use the qualified isolated environment for a dependency-free process.

For every supported positive token count, network dispatch and return use
GIN. There is no exact-token whitelist and no runtime NVSHMEM fallback.
Same-NUMA EP4 uses P2P and does not need network transfers. Inputs outside
measured ranges still require memory-capacity and correctness qualification;
policy coverage does not imply exhaustive GPU testing. T=0 is rejected.

The P2P arena is sized from the input shape. Additional GIN workspaces are
registered separately. `MEGAMOE_SYMMETRIC_BYTES` may enlarge the P2P arena,
but a value below the calculated requirement is rejected.

## Run

From the repository root, run the validated DSV4-flash EP4 correctness case:
DSV4-flash requires `--gate_up_clamp 10.0`; omitting it disables the clamp.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MEGA_STRICT_TORCH_REF=1

torchrun --standalone --nproc_per_node=4 \
  -m moe_sm120_nvfp4_split.mega_runner \
  --num_tokens_per_rank 2048 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --gate_up_clamp 10.0 \
  --data_parallel_size 1 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape \
  --comm_backend p2p_direct \
  --split_launch green_graph
```

Run the same specialization as a CUDA-event benchmark:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m moe_sm120_nvfp4_split.mega_runner \
  --num_tokens_per_rank 2048 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
  --gate_up_clamp 10.0 \
  --data_parallel_size 1 \
  --tensor_parallel_size 1 \
  --route_distribution balanced \
  --enable_static_expert_shape \
  --comm_backend p2p_direct \
  --split_launch green_graph \
  --perf_run --skip_ref_check --use_cuda_events \
  --perf_warmup 50 --perf_iters 200
```

`run_mega_tests.sh` covers EP4 P2P reference cases. `run_hybrid_tests.sh`
covers EP8 route distributions plus dynamic/static 100-replay transport
checks.

## Production API and cache

Framework integrations should import `api.py`, not `mega_runner.py`:

1. Construct `MegaMoEProblemSpec`.
2. Call `select_compile_spec(...)` with topology and SM properties.
3. Cache by `MegaMoECompileSpec.cache_key`.
4. Build with `build_split_kernels(spec)` and allocate the returned local and
   symmetric workspace sizes.

The cache ABI includes the NVFP4 dtype/layout specialization. Do not reuse a
W4A8 or W8A8 compiled-kernel cache entry.

When `rank_local_combine` is enabled, supported layouts use source-predecoded
contributor slots, 128-bit contiguous-lane I/O, and a two-fragment register
pipeline in the existing compute warps. This is an implementation detail, not
an additional option. The original top-k accumulation order and completion
protocol are unchanged. The packed plan supports top-k up to eight; hidden
widths divisible by 2048 guarantee complete 512-channel reduction steps for
every publication group. Other layouts automatically retain the original
reducer without allocating a plan. Cache ABI 21 includes this workspace change.

`green_graph` is the default production launch mode. `sequential` remains only
for bring-up and debugging.

Green Graph binds K1/K2 through retained `CUkernel` handles with `func=NULL`
and an explicit Green Context. A non-null `CUfunction` would make CUDA ignore
the supplied context. The compiled libraries and launch attributes are retained
for the graph lifetime. K3 executes unchanged after the K1/K2 graph on its root
stream; the completion event includes K3 before the caller can reset the next
epoch. Physical probes verify disjoint SM ownership, not isolation of shared
L2, DRAM or PCIe bandwidth.

The current selector screens a frozen candidate catalog, evaluates fitted
costs, then applies qualified robustness gates. See `SELECTOR.md` for the
calibration domain and compatibility fallback. Queue bundle selection is part
of the complete configuration; explicit overrides still win. It is distinct
from FC1 descriptor readiness and token-back combine groups. The old ABI23
shape-specific Q8 rule is historical, not the current selection algorithm.
The current cache ABI is defined in `api.py`; it includes compiler-semantic
changes as well as device layout changes.
