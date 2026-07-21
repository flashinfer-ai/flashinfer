# MoE-EP implementation & usage — `MoEEpLayer` call stack + correctness walkthrough

How the FlashInfer **MoE expert-parallel** API is put together and how to use it, walked
through the multi-GPU correctness test `tests/moe_ep/test_moe_ep_compute_correctness.py`.
For container/build/run and benchmark numbers see `benchmarks/MoE_benchmarks.md`.

`MoEEpLayer` runs one MoE layer split across ranks as **dispatch → per-expert grouped GEMM
→ combine**, over a pluggable transport (`nccl_ep` or `nixl_ep`). The expert GEMM reuses the
unified `flashinfer.fused_moe.MoELayer` as a pure per-expert grouped GEMM (no routing — that
lives in dispatch/combine).

## 1. Public API (`flashinfer.moe_ep`)

| Symbol | Role |
|---|---|
| `MoEEpLayer(bootstrap, fleet_params, weights, fleet_knobs=(), backend=...)` | the layer (`nn.Module`) |
| `MoEEpTensors(hidden_states, topk_ids, topk_weights)` | per-rank forward inputs |
| `BootstrapConfig(world_size, rank, stream, nccl_comm)` | transport bootstrap |
| `FleetParams(num_experts, max_tokens_per_rank, token_hidden_size, dtype_bytes, algorithm, layout)` | EP geometry |
| `EpAlgorithm.{LOW_LATENCY, HIGH_THROUGHPUT}`, `EpLayout.{EXPERT_MAJOR, RANK_MAJOR, FLAT}` | algorithm / receive layout |
| `weights: MoEWeightPack` (layer arg) + `SplitConfig(kernel=FusedMoeKernelConfig(moe_config: MoEConfig))` | the expert GEMM (config from `flashinfer.fused_moe.api`) |

`create_fleet(...)` raises `MoEEpNotBuiltError` (with rebuild hint) if the backend extension
isn't present; `available_backends()` lists what's built.

## 2. Call stack — `MoEEpLayer.forward(t)`

```
forward(t: MoEEpTensors)                                   # flashinfer/moe_ep/layer.py
├─ _ensure_fleet() → create_fleet(bootstrap, fleet_params, backend)   # NcclEpFleet: nccl.ep Group
├─ fleet.create_handle(HandleParams(topk_ids=t.topk_ids),            # NcclEpHandle: binds topk_ids,
│      algo_knobs=[HandleAlgoKnobUserStream(stream),                  #   creates nccl.ep handle for
│                  HandleAlgoKnobTopKWeights(weights=t.topk_weights)])#   the chosen Layout
├─ handle.dispatch(DispatchInputParams(x=[t.hidden_states]))         # → _dispatch_ll / _dispatch_ll_rank_major
│      → nccl.ep dispatch + complete  →  DispatchOutput(expert_tensors, recv_topk_idx/weights)  #   / _dispatch_ht
├─ _inner_compute(d):                                                # the EP→compute bridge
│      EXPERT_MAJOR → build_activation_pack(...)        ┐ flatten 3D recv → token-major
│      RANK_MAJOR/HT → build_activation_pack_rank_major ┘ pack (selected_experts, final_scales)
│      → MoELayer(compute_config)(act_pack, weights)     # per-expert grouped GEMM (top_k=1 local)
│      → reshape_for_combine(out_2d, …)                  # back to the 3D combine layout
├─ handle.combine(CombineInputParams(x=[expert_out], out=empty_like(hidden_states)))
│      → nccl.ep combine
└─ handle.complete()  →  return c.x                                  # [num_tokens, hidden] bf16
```

With `enable_timing=True`, `forward` brackets dispatch/compute/combine with CUDA events and
records `last_timings_ms` (all stages run on the `HandleAlgoKnobUserStream`, so on-stream
events capture their GPU time; `dispatch` host-syncs internally, which doesn't perturb them).

The handle backend (`flashinfer/moe_ep/nccl_ep/handle.py`) is where the three I/O contracts
live, and where the host-call fast path (`NV_FI_EP_FAST_PATH`) and burn-down probes (`EP_PROFILE_HOST`)
are wired (see `benchmarks/MoE_benchmarks.md` §3.2).

## 3. Algorithm / layout contracts

| | recv buffer | compute routing | combine |
|---|---|---|---|
| **LL EXPERT_MAJOR** | `[num_local_experts, max_tok×world, hidden]`, rows pre-assigned to experts | `top_k=1` per dispatched row | reweights per-token **on receive** (`topk_weights`) |
| **LL RANK_MAJOR** | `[world, max_tok, hidden]`, tokens grouped by source rank | real `top_k` over **received** routing; non-local picks masked to 0 | unweighted **sum** across ranks (caller pre-reduced) |
| **HT FLAT** | `[num_recv, hidden]` token-major (recv `topk_idx`/`topk_weights`, `-1`=non-local) | same as RANK_MAJOR (received-routing) | unweighted sum |

RANK_MAJOR/HT carry per-token routing back from dispatch; their received `topk_idx` are
**LOCAL** expert indices (`-1` = non-local pick) and (RANK_MAJOR) **int32** — the bridge
converts local→global and masks non-local picks to weight 0.

## 4. Usage walkthrough — `test_moe_ep_compute_correctness.py`

The test runs `dispatch → compute → combine` per rank over its own 128 tokens and checks the
combined output against a single-process dense MoE built from the **full** (replicated) expert
weights — for both LL layouts. Config: `num_experts=16, top_k=8, 128 tok/rank, hidden=8192,
intermediate=2048, world=8, bf16`.

**Why no cross-rank gather is needed for the reference:** every rank holds the *full* weight
set (constant seed → identical on all ranks), and a token's EP output is the weighted sum over
its top-k experts wherever they live. So each rank can compute the full reference for its own
tokens locally and compare. This isolates dispatch/compute/combine correctness.

Step by step (`_run_one_layout`):

1. **Weights, identical on every rank** — `torch.Generator.manual_seed(2024)`; `w1_full`
   `[E, 2·I, H]`, `w2_full` `[E, H, I]`, scaled `~1/sqrt(fan_in)` so activations stay O(1)
   (bf16-precision regime, `RTOL=ATOL=3e-2`).
2. **Per-rank tokens + routing** — distinct `manual_seed(1000+rank)`; `topk_ids` via
   `scores.topk(top_k)`, softmax `topk_weights`.
3. **Build the expert GEMM** (`_build_bf16_compute`) — `MoEConfig(routing=RoutingConfig(
   num_experts, top_k), quant=QuantConfig(BF16), experts=ExpertConfig(intermediate_size,
   local_expert_offset=offset, local_num_experts), backend=BackendOptions((TrtllmBf16Config(),)),
   execution=ExecutionConfig(tune_max_num_tokens))` over **this rank's local expert slice**
   (`w*_full[offset : offset+local_num_experts]`). `MoEWeightPack.prepare_for("trtllm_bf16_routed",
   {gemm1_weights, gemm2_weights})` after the `_block_major_k` shuffle.
   - **`tune_max_num_tokens`**: EXPERT_MAJOR pads to `local_experts × tok/rank × world`;
     RANK_MAJOR processes `tok/rank × world`.
   - **Gotcha — `_block_major_k` must use `epilogue_tile_m=64`** (`shuffle_matrix_a` then
     `convert_to_block_layout(block_k=128)`, both GEMMs, no gated-act reorder). The latency
     bench used 128 — fine for timing, numerically wrong; this functional test surfaced it.
4. **Construct the layer** —
   ```python
   layer = MoEEpLayer(
       BootstrapConfig(world_size, rank, stream=torch.cuda.current_stream().cuda_stream, nccl_comm=None),
       FleetParams(num_experts, max_tokens_per_rank=TOKENS_PER_RANK, token_hidden_size=HIDDEN,
                   dtype_bytes=2, algorithm=EpAlgorithm.LOW_LATENCY, layout=layout),
       weights=wp,
       backend=SplitConfig(comm=NcclEpConfig(), kernel=FusedMoeKernelConfig(moe_config=cfg)),
   )
   ```
5. **Run** — `y = layer.forward(MoEEpTensors(hidden_states=x, topk_ids=topk_ids,
   topk_weights=topk_weights))`; assert `y.shape == x.shape`.
6. **Reference + assert** — primary reference `_kernel_full_moe_reference` runs the **same**
   `MoELayer` kernel non-EP (all experts local, offset 0, real top_k) so it's immune to
   kernel-convention/weight-shuffle quirks; `torch.testing.assert_close(y, y_kernel, rtol,
   atol)`. A textbook fp32 `_torch_dense_reference` is a secondary diagnostic only. Always
   `layer.destroy()`.

`pytest_generate_tests` parametrizes `layout ∈ {expert_major, rank_major}`.

**Launch:**
```bash
torchrun --nproc_per_node=8 -m pytest \
  tests/moe_ep/test_moe_ep_compute_correctness.py -v -s -m "nvep and gpu_8"
# or, no pytest:
torchrun --nproc_per_node=8 tests/moe_ep/test_moe_ep_compute_correctness.py
```

## 5. Correctness status

Both LL layouts pass on 8× B200 (bf16): EP `dispatch→compute→combine` matches the non-EP
`MoELayer` to **rel-err ≈ 0.0045** (EXPERT_MAJOR and RANK_MAJOR). HT FLAT matches to
**≈ 0.007** at 4096/8192 tok/rank (`tests/moe_ep/test_moe_ep_ht_correctness.py`).

Two multi-rank bugs this functional test surfaced (invisible to the random-data latency
benchmark) and fixed: (1) the trtllm routed runner packed **local** expert ids while the
kernel expects **global** ids filtered by `local_expert_offset` → ranks with offset > 0
dropped their experts; fixed by packing global ids. (2) RANK_MAJOR/HT received `topk_idx` are
**local** (`-1` = non-local), not global; the bridge misread them — fixed with local→global
conversion + masking.

Related single-GPU tests: `test_compute_bridge.py` (layout-bridge unit tests),
`test_layer_single_gpu.py`, `test_config.py`, `test_constraints.py`; transport smoke:
`smoke_nccl_ep.py` / `smoke_nixl_ep.py`.

## 6. Pointers
- Container, how to run, benchmark numbers, the host-call fast path: `benchmarks/MoE_benchmarks.md`.
- Backend handle (I/O contracts, fast path, profiling): `flashinfer/moe_ep/nccl_ep/handle.py`.
- EP→compute bridge: `flashinfer/moe_ep/_compute_bridge.py`; layer: `flashinfer/moe_ep/layer.py`.
