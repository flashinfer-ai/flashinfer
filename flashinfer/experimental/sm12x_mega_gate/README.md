# Fused routing gate on SM120 (CuTe DSL)

`from flashinfer.mega_gate import prepare_mega_gate` dispatches compute
capability 12.x to this backend (SM100a/SM103a keep the generated DeepGEMM
programs of `../deepgemm_mega_gate`). It fuses the BF16 routing GEMM
`x[M,K] @ weight[E,K]^T`, `sqrt(softplus)` scoring, biased top-k selection,
unbiased weight normalization and logical-to-physical expert mapping into one
kernel, with the semantics of DeepGEMM's `bf16_mega_gate`. The prepared call
returns a plan; `plan.run()` launches on the current PyTorch stream and returns
`(expert_indices, weights)`: int64 `[M,topk+shared]` physical expert ids and
FP32 weights scaled by `routed_scaling_factor`.

Status: validated target SM120 (`sm_120a`, RTX PRO 6000 Blackwell / RTX 5090).
SM121 (GB10) runs the same instructions but is not enabled until it is
validated: add `(12, 1): "sm_121a"` to `_ARCHES` in `mega_gate.py`.

## Design

SM120 has no tcgen05/TMEM, so the GEMM runs on warp-level
`mma.sync.m16n8k16` (BF16 in, FP32 accumulators in registers), fed by a TMA
producer warp through a multi-stage mbarrier pipeline and `ldmatrix`
(`kernel.py`). The structure follows DeepGEMM's `sm100_bf16_mega_gate`:

- **Swap-AB.** The weight is the MMA A operand (128 experts per CTA) and the
  tokens the B operand, so decode-sized token counts pad only up to the 8-wide
  MMA N granularity.
- **Decomposition.** A token block of 16/32/48/64 tokens is computed by
  `L = ceil(E/128) * split_k` logical CTAs; each writes the FP32 partial logits
  of its experts and K slice to `scratch[token blocks, split_k, block tokens, aligned E]`.
  The grid is persistent: `num_sms // L` worker groups of L CTAs (at most one
  CTA per SM, so a group is co-resident) stride over the token blocks.
- **Warp roles.** 4 MMA warps (mainloop, partial stores, arrival at the block's
  score barrier, then straight on to the next block), 1 TMA warp and 4 gate
  warps. Once all L CTAs of a block have arrived, every CTA ranks a share of the
  block's tokens (DeepGEMM's order), one warp per token, overlapping the next
  block's GEMM.
- **Score barrier.** One int32 counter per token block in `score_barriers`
  (`red.release.gpu` arrivals, `ld.acquire.gpu` waits). Instead of DeepGEMM's
  `%gridid` tags (a CUDA graph kernel node keeps one grid id across replays), the
  last CTA to exit zeroes the launch's counters, so every launch and graph
  replay starts from zero.
- **Top-k.** DeepGEMM's: the split partials are summed in a fixed order (one
  float4 per wave and split), each lane sorts its 4 experts of every 128-expert
  wave and every round offers the best unconsumed candidate of each wave; the
  warp keeps the largest rank (ties: the smaller expert).
- **Proxy fence.** The MMA warps read the pipeline stages with `ldmatrix`
  (generic proxy) and the TMA refills them (async proxy): the producer issues
  `fence.proxy.async.shared::cta` after each empty-barrier wait. Without it the
  refill can land under a pending read once the gate warps load the memory
  system (the same SM120 hazard fixed in `cake_minimax_h3`, #7455).
- **Schedule.** `heuristics.py` picks (block tokens, split-K) as the argmin of a
  cost model of one CTA's critical path, fitted on every candidate schedule
  measured on a 110-SM board (1.4% slower than the measured best on average).

Compared to the SM100 programs this backend also accepts other hidden sizes
(multiples of 64), 1..1024 experts, top-k up to 32 slots, no bias, token masks
and image bias. Fixed routing (`fix_routing_mask`) and random routing
(`force_random`) are not implemented yet and raise `NotImplementedError`.

Kernels are compiled once per schedule and input signature through
FlashInfer's persistent CuTe DSL cache (`cached_ops/sm12x_mega_gate_v1_sm120a_cute_dsl/`);
the token count is a runtime argument.

## Workspace

Optional caller-owned `scratch` (FP32, `plan.scratch` shape) and
`score_barriers` (uint64 `[token blocks, 16]`, zeroed once before first use;
every launch leaves them zeroed again). E must be a multiple of 4 and the
biases 16-byte aligned (they are read as float4). Preparation owns all allocations and
device queries; `run()` only launches, so it can be captured in a CUDA graph.
Do not use one plan concurrently across streams.

```bash
pytest tests/experimental/test_mega_gate_sm12x.py -q
python benchmarks/bench_mega_gate_sm12x.py --cupti
```
