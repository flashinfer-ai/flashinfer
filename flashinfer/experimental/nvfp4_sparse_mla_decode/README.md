# NVFP4 sparse MLA decode (SM100/SM103, experimental)

Owner: @stu-cao · Tracking issue: [#5716](https://github.com/flashinfer-ai/flashinfer/issues/5716) · Status: experimental, JIT-only

Decode attention for DSA-style sparse MLA (DeepSeek-V3.2, GLM-5) over an NVFP4 KV cache, reading vLLM's
`nvfp4_ds_mla` rows as stored. Each query token attends to the rows its sparse indexer selected:

```
out[t, h] = bmm2_scale * sum_k softmax_k(bmm1_scale * query[t, h] . K[i_tk]) * V[i_tk]
```

`K` is a row's 576-dim latent (512 NoPE + 64 RoPE), dequantized exactly in shared memory, and `V` its first 512 dims.

## Why

FlashInfer has no SM100 sparse MLA decode that reads an NVFP4 cache. An engine that stores `nvfp4_ds_mla` rows
reaches sparse MLA on SM100 in one of two ways:

- FlashMLA's NVFP4 sparse decode, which pads the 16 heads per rank to 64.
- A gather that restages the selected rows as FP8 for TRTLLM-gen. On GB200 at 15 tokens per step the gather costs
  5.3 us per layer on top of TRTLLM-gen's 12.4 us, plus its traffic through global memory, and re-rounding every
  dequantized value to e4m3 costs 3.0-4.7 % relative error.

This kernel reads the 352-byte rows directly, 39 % fewer bytes per key than FP8's 576 (an NVFP4 cache holds 1.6x the
tokens), and stays within 0.3 % of an exact FP32 result. It is faster than FP8 TRTLLM-gen only in a band of batch sizes
(see Measurements): the reason to store NVFP4 is capacity, and this kernel keeps decode from paying for the staging
gather.

## Usage

Calling the API is the opt-in; it warns once with `ExperimentalWarning`.

```python
import math

import torch

import flashinfer

rows = 64 * 1024
kv_cache = torch.randint(0, 256, (rows, 352), dtype=torch.uint8, device="cuda")  # nvfp4_ds_mla rows, written by the engine
kv_cache[:, 256:] = torch.rand(rows, 96, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)  # RoPE values and scales
query = torch.randn(4, 16, 576, device="cuda").to(torch.float8_e4m3fn)
indices = torch.randint(0, rows, (4, 2048), dtype=torch.int32, device="cuda")
indices[:, 1500:] = -1  # empty slots
out = flashinfer.mla.nvfp4_sparse_mla_decode(query, kv_cache, indices, bmm1_scale=1 / math.sqrt(576))
print(out.shape, out.dtype)  # torch.Size([4, 16, 512]) torch.bfloat16
```

`bmm1_scale` is the softmax scale times the query's dequantization scale. `indices` holds flat row ids
(`block_id * block_size + offset`), so a `[num_blocks, block_size, 352]` cache can be passed as is.
`examples/experimental/nvfp4_sparse_mla_decode.py` runs the same call and checks the result against a PyTorch
reference.

## Cache row layout (`nvfp4_ds_mla`, 352 bytes per token)

| bytes | content |
|---|---|
| 0-255 | 512 NoPE values, e2m1, two per byte; element `2i` in the low nibble |
| 256-319 | 64 RoPE values, e4m3 |
| 320-351 | 32 e4m3 scales, one per 16 NoPE values; block `b`'s scale at byte `320 + 8 * (b % 4) + b // 4` |

A NoPE value is `e2m1 * scale`, with no per-tensor factor.

## Limits

- Compute capability 10.0 (B200, GB200) or 10.3 (B300, GB300): the module is built for `sm_100a` or `sm_103a` to
  match the device.
- 16 query heads per rank, a 576-dim latent with 512-dim values, e4m3 queries, bf16 output. No LSE output, no attention
  sinks.
- `topk` is a multiple of 32 that gives each CTA of a cluster 3 to 32 stages of 32 keys. Tested widths: 512, 1024, 2048.
- `kv_cache` rows are contiguous; `indices` values are not range-checked (`-1` is the only special value).
- Tuned for up to 45-46 query tokens per launch, what one wave of 3-CTA clusters holds on B300 and GB200. Larger
  batches run in several waves. Against FP8 TRTLLM-gen the kernel is ahead only in a band of batch sizes (10 to 25
  tokens per launch on B300) and behind below and above it; see Measurements.

## Design

One thread-block cluster of C CTAs (3 to 8) serves one query token. CTA `r` walks its share of the token's key stages,
32 keys per stage, and 16 warps hold fixed roles:

| warps | role |
|---|---|
| 0-7 | fetch 4 rows each per stage with `cp.async` (a `-1` row is zero-filled without a global read) into a raw ring one slot deeper than the K ring, then expand e2m1 x e4m3 into an f16 K/V tile, which is exact in f16 |
| 8-11 | `S = Q K^T` on `mma.sync` m16n8k16, one k-quarter each with Q in registers, then the online softmax for 4 heads each |
| 12-15 | `O = alpha * O + P V`, 128 output dims each, V fragments through `ldmatrix.trans` |

The CTAs of a cluster then push f16 partial outputs and their (max, sum) over distributed shared memory to the CTA that
owns each 8-dim output block, and merge after one cluster barrier: no second kernel, no workspace. Each CTA uses
218,624 bytes of shared memory, so one CTA runs per SM.

The cluster size is chosen per launch: the largest of 8, 6, 5, 4 and 3 whose clusters for every query token fit on the
device at once, from `cudaOccupancyMaxActiveClusters`. On GB200 (152 SMs) that is 8 CTAs up to 15 tokens, 6 up to 23,
5 up to 28, 4 up to 36 and 3 up to 46; on B300 (148 SMs) 8 up to 15, 6 up to 22, 5 up to 26, 4 up to 33 and 3 up to 45.

Nsight Compute on a GB200 with 4-CTA clusters at 35 tokens: L1/shared 69 %, tensor pipe 24 %, DRAM 16 %. Shared-memory
traffic bounds the kernel, 63 % of its wavefronts the f16 key tile (written once, read by QK and by PV). A tcgen05
version that keeps keys in tensor memory is the natural successor.

## Measurements

Microseconds per launch, top-k 2048, 16 heads, request-shaped indices, CUDA-graph replay.

B300 SXM6 (compute capability 10.3), CUDA 13.0, driver 580.173.02, torch 2.14.0+cu130, this module
(`python benchmarks/bench_nvfp4_sparse_mla_decode.py`, tokens per launch = 5 x requests):

| tokens per launch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| FP8 TRTLLM-gen sparse MLA | 11.0 | 12.4 | 12.8 | 16.5 | 19.0 | 16.7 | 19.2 | 19.8 | 22.3 |
| this kernel | 12.0 | 12.2 | 12.4 | 14.9 | 17.0 | 19.6 | 25.4 | 25.3 | 25.3 |

The kernel is ahead of FP8 TRTLLM-gen at 10 to 25 tokens per launch (by 2 to 10 %) and behind at 5 tokens and from 30
(by 9 % at 5, 14 to 33 % from 30). From 34 tokens B300 fits no more than 33 4-CTA clusters in a wave, so the plan drops
to 3-CTA clusters, where each CTA walks more key stages.

GB200, CUDA 13.0, the same kernel body built outside FlashInfer:

| tokens per launch | 15 | 20 | 25 | 35 |
|---|---|---|---|---|
| FP8 TRTLLM-gen sparse MLA | 13.7 | 18.0 | 18.4 | 20.3 |
| this kernel | 11.8-12.2 | 14.8 | 16.8 | 19.6 |

Inside vLLM decode steps (rank-0 traces at equal tokens per step, from a build that also compacted indices, 0.3-1.2 us
slower than this one): 15.6 against 17.0 us at 20 tokens, 17.8 against 17.3 at 25, 20.6 against 17.2 at 35.

`benchmarks/bench_nvfp4_sparse_mla_decode.py` reproduces the kernel comparison.

## Validation status

- B300 (SM103, CUDA 13.0, driver 580.173.02): on a fresh JIT build of this module on top of FlashInfer `main`
  (13122fcf), all 93 tests in `tests/experimental/test_nvfp4_sparse_mla_decode.py` pass, including every cluster size,
  padded indices and CUDA-graph replay, and `examples/experimental/nvfp4_sparse_mla_decode.py` reports 0.25 %
  relative error. The largest relative error against the exact FP32 reference is 0.29 % over 5 to 64 tokens and three
  padding patterns.
- SM100: the kernel body is the one validated on GB200 inside vLLM, end to end, and in an exact-reference harness at
  0.26 % relative error, including padded indices. For `sm_100a` this module's kernel compiles to the same SASS as that
  build with its debug timestamps removed: 2,160 instructions with identical encodings, 128 registers, no spills. The
  packaged tests have not run on a compute capability 10.0 device yet.

## `backend="cake"` (generated program, experimental)

`flashinfer.mla.nvfp4_sparse_mla_decode(..., backend="cake")` serves the same operator, tensor ABI and `-1`
semantics with a generated program of the same design family: one thread-block cluster of `C` CTAs per query
token (grid `(T, C, 1)`, cluster `(1, C, 1)`, 448 threads: 8 math warps, 4 softmax warps, 2 loader warps), CTA
`c` owning keys `[c * keys_per_cta, (c + 1) * keys_per_cta)` in 32-key stages. `C = 1` writes the normalized
output directly; a cluster merges its fp32 partials `(O, m, l)` through distributed shared memory, rank `G % C`
owning 8-dim output group `G`. By design every e2m1 x e4m3 product is formed exactly in f16, S and O accumulate
in fp32 and P is f16; the acceptance run below confirms the tolerance of the tests.

- Cluster sizes 1, 2, 3, 4, 5, 6 and 8: one JIT module per (architecture, `C`), registered in `cake_jit.py`
  with its sources under `csrc/cake_nvfp4_sparse_mla_decode/{sm_100a,sm_103a}/`. `num_ctas_per_token=7` is
  rejected (`ValueError`), not re-planned.
- Plan (`cake_backend.plan_ctas`): the largest of 8, 6, 5, 4, 3, 2 that gives every CTA a full 32-key stage and
  leaves no CTA without keys (`is_valid_split`; 512 keys skip 5), whose `num_tokens` clusters are co-resident
  in one wave (`cudaOccupancyMaxActiveClusters` of each module, queried once per device), otherwise the smallest
  such split in several waves; 1 when no split qualifies. `keys_per_cta = ceil(topk / C)` rounded up to a
  multiple of 32, at most 2048. A forced split that leaves a CTA without keys is rejected.
- Dynamic shared memory 192,256 bytes (`C = 1`) or 226,816 bytes (`C > 1`), the schedule's pool plus its mbarriers: one CTA per SM.
- `cake_backend.prepare_nvfp4_sparse_mla_decode(...)` returns a runner that launches without allocation (CUDA
  Graph safe); `cake_backend.generated_program_available(device)` reports whether the program is registered in
  the checkout. The tests and the benchmark skip the arm otherwise.

Cold-L2 microseconds per launch of the generated program (`python benchmarks/bench_nvfp4_sparse_mla_decode.py
--cold-l2`, 16 heads, request-shaped indices, CUDA-graph replay), to be filled from the acceptance run:

| T | topk | B200 (sm_100a) | GB300 (sm_103a) |
|---|---|---|---|
| 5 | 2048 | | |
| 10 | 2048 | | |
| 15 | 2048 | | |
| 20 | 2048 | | |
| 25 | 2048 | | |
| 30 | 2048 | | |
| 35 | 2048 | | |
| 40 | 2048 | | |
| 45 | 2048 | | |
| 5 | 1024 | | |
| 10 | 1024 | | |
| 15 | 1024 | | |
| 20 | 1024 | | |
| 25 | 1024 | | |
| 30 | 1024 | | |
| 35 | 1024 | | |
| 40 | 1024 | | |
| 45 | 1024 | | |

Tracking: flashinfer-ai/flashinfer#5716 (DSA NVFP4 sparse MLA decode), tracker #4254.

## Graduation plan

1. Run the tests and the benchmark on B200/GB200 in CI.
2. vLLM: let the FlashInfer sparse-MLA backend accept `nvfp4_ds_mla` and call this API. The vLLM PR that does both is
   [vllm-project/vllm#59342](https://github.com/vllm-project/vllm/pull/59342).
3. Replace the `mma.sync` pipeline with tcgen05 once it is faster at the same shapes.
