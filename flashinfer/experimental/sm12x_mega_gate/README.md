# Fused routing gate on SM120 (CuTe DSL)

`from flashinfer.mega_gate import prepare_mega_gate` dispatches compute
capability 12.x to this backend (SM100a/SM103a/SM107a keep the generated DeepGEMM
programs of `../deepgemm_mega_gate`). It fuses the BF16 routing GEMM
`x[M,K] @ weight[E,K]^T`, `sqrt(softplus)` scoring, biased top-k selection,
unbiased weight normalization and logical-to-physical expert mapping into one
kernel, with the semantics of DeepGEMM's `bf16_mega_gate`. The prepared call
returns a plan; `plan.run()` launches on the current PyTorch stream and returns
`(expert_indices, weights)`: int64 `[M,topk+shared]` physical expert ids and
FP32 weights scaled by `routed_scaling_factor`.

Status: experimental. `prepare_mega_gate` is an `@flashinfer_experimental_api`:
calling it is the opt-in (it warns once with `ExperimentalWarning`; no
environment variable is involved), and on compute capability 12.x it always
selects this backend. SM120 (`sm_120a`) is validated and tuned on an RTX PRO
5000 Blackwell (110 SMs). SM121 (GB10) runs the same instructions but is not
enabled until it is validated (its calls raise): add `(12, 1): "sm_121a"` to
`_ARCHES` in `mega_gate.py`. The kernel is a port of DeepGEMM's
`sm100_bf16_mega_gate`; `DEEPGEMM_NOTICE.txt` carries DeepGEMM's MIT notice.

## Design

SM120 has no tcgen05/TMEM, so the GEMM runs on warp-level
`mma.sync.m16n8k16` (BF16 in, FP32 accumulators in registers), fed by a TMA
producer warp through a multi-stage mbarrier pipeline and `ldmatrix`
(`kernel.py`). The structure follows DeepGEMM's `sm100_bf16_mega_gate`:

- **Swap-AB.** The weight is the MMA A operand (128 experts per CTA) and the
  tokens the B operand, so decode-sized token counts pad only up to the 8-wide
  MMA N granularity.
- **Decomposition.** A token block of 16, 32, 48, 64, 96 or 128 tokens is computed by
  `L = ceil(E/128) * split_k` logical CTAs; each writes the FP32 partial logits
  of its experts and K slice to `scratch[token blocks, split_k, block tokens, aligned E]`.
  The grid is persistent: `num_sms // L` worker groups of L CTAs (at most one
  CTA per SM, so a group is co-resident) stride over the token blocks.
- **Warp roles.** 4 MMA warps (mainloop, partial stores, arrival at the block's
  score barrier, then straight on to the next block; 8 for 96- and 128-token
  blocks, two per 32 experts), 1 TMA warp and 4 gate warps (8 when a CTA ranks
  more than 4 tokens of a 64-token or smaller block). Once all L CTAs of a
  block have arrived, every CTA ranks a share of the block's tokens (DeepGEMM's
  order), one warp per token, overlapping the next block's GEMM. When a CTA ranks
  at most 3 tokens of a block (on 110 SMs up to 256 tokens with 384 experts, 288
  with 256), the gate is cooperative: each
  (token, 128-expert wave) item goes to one gate warp (its split-K loads and
  transcendentals), is posted sorted to shared memory, and a selector warp per
  token merges the waves; this takes 4 gate warps for one token and 8 for more,
  and the exchange costs a pipeline stage at 48- and 64-token blocks. When a CTA
  ranks more tokens of a block than it has gate warps (with wide blocks up to 43
  tokens on 4 gate warps; also smaller blocks over few CTAs, e.g. deterministic
  calls of 65-512 tokens), the MMA warps join them on the CTA's last block, whose
  gate work nothing else overlaps.
- **Score barrier.** One 128 B line per token block in `score_barriers`: an
  arrival counter (`red.release.gpu` by the MMA warps, `ld.acquire.gpu` polls by
  one gate lane) and a waiter counter. Every CTA that waited counts itself after
  reading the full count and the last one zeroes the line. This replaces
  DeepGEMM's `%gridid` tags (a CUDA graph kernel node keeps one grid id across
  replays) without a teardown pass, so every launch and graph replay starts
  from zero.
- **Top-k.** DeepGEMM's: the split partials are summed in a fixed order (one
  float4 per wave and split), each lane sorts its 4 experts of every 128-expert
  wave and every round offers the best unconsumed candidate of each wave; the
  warp keeps the largest rank (ties: the smaller expert). SM120 has no float
  `redux.sync`, so a round is two integer reduxes on order-preserving keys. The
  biases and replica counts of a lane's experts are loaded once per CTA, and a
  token's replica of every lane expert is looked up before its top-k (a modulo by
  a precomputed reciprocal), so no dependent lookup follows the selection. The
  square roots use `sqrt.rn`'s fast path without its slow-path branch (bitwise
  equal for every input but NaN payloads), so a lane's square roots interleave.
- **Proxy fence.** The MMA warps read the pipeline stages with `ldmatrix`
  (generic proxy) and the TMA refills them (async proxy): the producer issues
  `fence.proxy.async.shared::cta` after each empty-barrier wait. Without it the
  refill can land under a pending read once the gate warps load the memory
  system (the same SM120 hazard fixed in `cake_minimax_h3`, #7455).
- **Schedule.** Up to 512 tokens `heuristics.py` takes a token block of `M / 4`
  tokens rounded up to 16 and clamped to 16..64; above, a 96- or 128-token block
  (each weight tile then serves more tokens: 5-11% faster from 768 tokens). The
  split-K, and above 512 tokens the block, minimize a cost model of one CTA's
  critical path, with separate coefficients for wide blocks, fitted on measured
  schedules of the 110-SM board with 384 experts (against every schedule
  measured since at 53 token counts from 1 to 16384: 0.4% slower than the best
  on average, 4.7% at worst at 280 tokens, from the block rule). With 256 experts
  (DeepSeek-V4-Flash, K=4096) the model misranks the near-tied one-wave
  schedules by up to 28%, so that shape takes a measured split per token range
  up to 3072 tokens and the smallest block that holds the tokens in one wave
  (`MEASURED_SPLITS`: 0.1% from the best on average, 1.7% at worst). These
  timings are medians over buffer placements: the same schedule moves by up to
  10% with the DRAM placement of its buffers.

Compared to the SM10x programs (K=5120, E=384, top-6) this backend also accepts
other hidden sizes (multiples of 64), up to 1024 experts (a multiple of 4),
top-k up to 32 slots and no bias. Fixed routing (`fix_routing_mask`, e.g. hash-routed layers)
takes the flagged tokens' logical experts from `unmapped_topk_idx` instead of
the top-k and weights them from their unbiased scores, as DeepGEMM does. Random
routing (`force_random`) is not implemented yet and raises
`NotImplementedError`. `num_sms` caps the SMs the kernel uses, leaving the rest
to kernels that run at the same time (communication overlapped with the gate):
uncapped, calls of up to 256 tokens use 30-96 CTAs (5120 x 384) or 32-96 (4096
x 256); capped at 64 SMs they take up to 1.1 us longer (4096 x 256, 1-240
tokens) or 1.9 us (5120 x 384 at 140 tokens; 1 token unchanged). A worker
group's CTAs wait for each other, so they must be resident together: launching
under an SM budget smaller than one group (up to 30 or 32 CTAs at decode sizes, e.g.
a green context or an MPS partition), or next to a kernel that holds SMs while
waiting on the gate, hangs.

Plans launch with programmatic dependent launch by default (`enable_pdl`): the
tokens, maps and workspace are read only after `griddepcontrol.wait`, so the
launch and the prologue overlap the previous kernel (about 0.5 us per call in a
CUDA graph after flashinfer's RMSNorm, 110-SM board), and the CTAs of the last
token block let a PDL dependent launch once only their gate work is left. A
standalone CUPTI timing of a PDL launch also counts its wait for the kernel
before it; `benchmarks/bench_mega_gate_sm12x.py` measures without PDL unless
`--pdl` is given.

Kernels are compiled once per schedule and input signature through
FlashInfer's persistent CuTe DSL cache (`cached_ops/sm12x_mega_gate_v1_sm120a_cute_dsl/`);
the token count is a runtime argument.

## Workspace

Optional caller-owned `scratch` (FP32, `plan.scratch` shape) and
`score_barriers` (uint64 `[token blocks, 16]`, zeroed once before first use;
every launch leaves them zeroed again). E must be a multiple of 4; x and the
weight (TMA), the biases, `logical_count` and `scratch` (float4 loads) must be
16-byte aligned, which preparation checks. Preparation owns all allocations and
device queries; `run()` only launches, so it can be captured in a CUDA graph.
Do not use one plan concurrently across streams.

```bash
pytest tests/experimental/test_mega_gate_sm12x.py -q
python benchmarks/bench_mega_gate_sm12x.py --cupti
```
