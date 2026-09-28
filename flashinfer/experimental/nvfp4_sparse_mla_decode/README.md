# NVFP4 sparse MLA decode (SM100, experimental)

Owner: @stu-cao · Tracking issue: to be opened with the upstream PR · Status: experimental, JIT-only

Decode attention for DSA-style sparse MLA (DeepSeek-V3.2, GLM-5) over an NVFP4 KV cache, reading vLLM's
`nvfp4_ds_mla` rows as stored. Each query token attends to the rows its sparse indexer selected:

```
out[t, h] = bmm2_scale * sum_k softmax_k(bmm1_scale * query[t, h] . K[i_tk]) * V[i_tk]
```

`K` is a row's 576-dim latent (512 NoPE + 64 RoPE), dequantized exactly in shared memory, and `V` its first 512 dims.

## Why

On SM100 an NVFP4 cache otherwise reaches sparse MLA in one of two slower ways:

- FlashMLA's NVFP4 sparse decode pads 16 heads per rank to 64. At 5-15 tokens per step it took 26 us per layer in our
  measurements, against about 14 us for FP8 TRTLLM-gen.
- A gather that restages the selected rows as FP8 for TRTLLM-gen adds 5.3 us per layer at 15 tokens (17.7 against
  12.4 us) plus the staging traffic, and re-rounding every dequantized value to e4m3 costs 3.0-4.7 % relative error.

This kernel reads the 352-byte rows directly (39 % fewer bytes per key than FP8's 576) and stays within 0.26 % of an
exact FP32 result.

## Usage

Calling the API is the opt-in; it warns once with `ExperimentalWarning`.

```python
import math

import torch

import flashinfer

rows = 64 * 1024
kv_cache = torch.zeros(rows, 352, dtype=torch.uint8, device="cuda")  # nvfp4_ds_mla rows, written by the engine
query = torch.randn(4, 16, 576, device="cuda").to(torch.float8_e4m3fn)
indices = torch.randint(0, rows, (4, 2048), dtype=torch.int32, device="cuda")
indices[:, 1500:] = -1  # empty slots
out = flashinfer.mla.nvfp4_sparse_mla_decode(query, kv_cache, indices, bmm1_scale=1 / math.sqrt(576))
print(out.shape, out.dtype)  # torch.Size([4, 16, 512]) torch.bfloat16
```

`bmm1_scale` is the softmax scale times the query's dequantization scale. `indices` holds flat row ids
(`block_id * block_size + offset`), so a `[num_blocks, block_size, 352]` cache can be passed as is.

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
  batches run in several waves and fall behind FP8 TRTLLM-gen.

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

Nsight Compute on the 3-CTA plan at 35 tokens: L1/shared 69 %, tensor pipe 24 %, DRAM 16 %. Shared-memory traffic
bounds the kernel, 63 % of it the f16 key tile. A tcgen05 version that keeps keys in tensor memory is the natural
successor.

## Measurements

Microseconds per launch, top-k 2048, 16 heads, request-shaped indices, CUDA-graph replay.

B300, CUDA 13.0, this module (`benchmarks/bench_nvfp4_sparse_mla_decode.py`):

| tokens per launch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| FP8 TRTLLM-gen sparse MLA | 10.6 | 11.9 | 13.0 | 16.7 | 16.9 | 16.8 | 18.9 | 19.5 | 22.0 |
| this kernel | 12.2 | 12.4 | 12.4 | 15.0 | 17.1 | 19.5 | 25.5 | 25.5 | 25.5 |

From 34 tokens B300 fits no more than 33 4-CTA clusters in a wave, so the plan drops to 3-CTA clusters, where each CTA
walks more key stages.

GB200, CUDA 13.0, the same kernel body built outside FlashInfer:

| tokens per launch | 15 | 20 | 25 | 35 |
|---|---|---|---|---|
| FP8 TRTLLM-gen sparse MLA | 13.7 | 18.0 | 18.4 | 20.3 |
| this kernel | 11.8-12.2 | 14.8 | 16.8 | 19.6 |

Inside vLLM decode steps (rank-0 traces at equal tokens per step, from a build that also compacted indices, 0.3-1.2 us
slower than this one): 15.6 against 17.0 us at 20 tokens, 17.8 against 17.3 at 25, 20.6 against 17.2 at 35.

`benchmarks/bench_nvfp4_sparse_mla_decode.py` reproduces the kernel comparison.

## Validation status

- B300 (SM103, CUDA 13.0, driver 580): all 93 tests in `tests/experimental/test_nvfp4_sparse_mla_decode.py` pass,
  including every cluster size, padded indices and CUDA-graph replay. The largest relative error against the exact
  FP32 reference is 0.29 % over 5 to 64 tokens and three padding patterns.
- SM100: the kernel body is the one validated on GB200 inside vLLM, end to end, and in an exact-reference harness at
  0.26 % relative error, including padded indices. For `sm_100a` this module's kernel compiles to the same SASS as that
  build with its debug timestamps removed: 2,160 instructions with identical encodings, 128 registers, no spills. The
  packaged tests have not run on a compute capability 10.0 device yet.

## Graduation plan

1. Run the tests and the benchmark on B200/GB200 in CI.
2. vLLM: let the FlashInfer sparse-MLA backend accept `nvfp4_ds_mla` and call this API. A branch that does both is
   [stu-cao/vllm `feat/nvfp4-ds-mla-flashinfer-sparse`](https://github.com/stu-cao/vllm/tree/feat/nvfp4-ds-mla-flashinfer-sparse).
3. Replace the `mma.sync` pipeline with tcgen05 once it is faster at the same shapes.
