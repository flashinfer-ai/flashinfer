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
cross-NUMA traffic through staged NVSHMEM IBGDA transport.

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

## Validated environment

- Python 3.12
- `nvidia-cutlass-dsl==4.6.0`
- CUDA Toolkit 13.3
- NVSHMEM 3.7.0 and NVSHMEM4Py 0.3.1
- SM120 RTX Pro 5000

The same-NUMA EP4 path requires CUDA peer access and an NVSHMEM symmetric GPU
heap. The cross-NUMA hybrid path additionally requires the matching NVSHMEM
3.7.0 headers/device bitcode and an IBGDA-capable NIC/GDR stack; its
compatibility guard rejects an unknown NVSHMEM device ABI.

## Run

From the repository root, run the validated DSV4-flash EP4 correctness case:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVSHMEM_HEAP_KIND=VIDMEM
export NVSHMEM_SYMMETRIC_SIZE=16G
export MEGA_STRICT_TORCH_REF=1

torchrun --standalone --nproc_per_node=4 \
  -m moe_sm120_nvfp4_split.mega_runner \
  --num_tokens_per_rank 2048 \
  --num_topk 6 \
  --num_total_experts 256 \
  --hidden 4096 \
  --intermediate 4096 \
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

`green_graph` is the default production launch mode. `sequential` remains only
for bring-up and debugging.
