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
traffic bounds the kernel, 63 % of its wavefronts the f16 key tile (written once, read by QK and by PV). The tcgen05
programs of `backend="cake"` (below) keep keys in tensor memory instead.

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
  (13122fcf), the tests in `tests/experimental/test_nvfp4_sparse_mla_decode.py` pass, including every cluster size,
  padded indices and CUDA-graph replay, and `examples/experimental/nvfp4_sparse_mla_decode.py` reports 0.25 %
  relative error. The largest relative error against the exact FP32 reference is 0.29 % over 5 to 64 tokens and three
  padding patterns.
- SM100: the kernel body is the one validated on GB200 inside vLLM, end to end, and in an exact-reference harness at
  0.26 % relative error, including padded indices. For `sm_100a` this module's kernel compiles to the same SASS as that
  build with its debug timestamps removed: 2,160 instructions with identical encodings, 128 registers, no spills. The
  packaged tests have not run on a compute capability 10.0 device yet.

## `backend="cake"` (generated tcgen05 programs, experimental)

`flashinfer.mla.nvfp4_sparse_mla_decode(..., backend="cake")` serves the same operator, tensor ABI and `-1` semantics
as `backend="cuda"`: e4m3 `[T, 16, 576]` queries, 352-byte `nvfp4_ds_mla` cache rows, int32 `[T, topk]` row ids with
`-1` an empty slot, and a bf16 `[T, 16, 512]` output. A token whose slots are all empty writes zeros.

The backend runs a family of generated kernels on the Blackwell tensor core (tcgen05) and its tensor memory (TMEM).
One thread-block cluster of `C` CTAs serves one query token: grid `(T, C, 1)`, cluster `(1, C, 1)`, 512 threads
(16 warps). CTA `c` owns keys `[c * keys_per_cta, (c + 1) * keys_per_cta)`, `keys_per_cta = ceil(topk / C)`, and
walks them in 128-key tiles. Keys and output dims, not the 16 heads, fill the MMA's M dimension, so no MMA row is
spent on padding: `S^T[key, head] = K . Q^T` (M = 128 keys, N = 16 heads) and `O^T[dim, head] += V^T . P^T`
(M = 128 dims per block, N = 16). The 16 warps hold fixed roles:

| count | role |
|---|---|
| 2 | issue TMA gather4 loads of the raw rows' e2m1 bytes into a 128-byte-swizzled ring (a `-1` row lands as zeros) |
| 4 | copy tile 0's raw rows with `cp.async` (a cold gather4 lands about a microsecond later) |
| 4 | layout, one thread per key: widen the e4m3 RoPE part to f16 in TMEM, lay out the block scale factors |
| 1 | issue the K dequant and the QK MMAs |
| 1 | issue the V dequant and the PV MMAs |
| 4 | softmax: read S^T from TMEM (lane = key), reduce head maxima, write P^T, rescale O^T in TMEM when one grows |

Both NoPE dequants run on the tensor core as block-scaled `kind::mxf4nvf4` products of the rows' raw e2m1 values
`K_raw` with an e2m1 identity matrix `I`, exact in the fp32 accumulator. `K = K_raw . I`, with the rows' e4m3 block
scales as scale_A, feeds a `kind::tf32` QK MMA with A = K in TMEM and B = Q^T; `V^T = I . K_raw^T` feeds the
`kind::tf32` PV MMA. The RoPE part uses a `kind::f16` MMA.

Every e2m1 x e4m3 product is exact in fp32 and the e4m3 query widens exactly. S and O accumulate in fp32 on the
tensor core, the softmax runs in fp32 (base 2), and P is rounded once to f16, then widened exactly to the tf32 operand.

With `C > 1`, every CTA pushes its fp32 O^T pieces and (m, l) through distributed shared memory (`st.async`) to the
rank that owns each 8-dim output group, which merges the ranks in rank order; `C = 1` writes the normalized output
directly. Either way the decode is one kernel with no workspace.

- Programs: one per cluster size (1, 2, 3, 4, 5, 6, 8) and split bound. A launch whose CTAs own at most 64, 128, 192
  or 256 keys runs the program built for that bound, and any larger split runs the general program: 7 x 5 = 35
  programs, each delivered for `sm_100a` and `sm_103a` (70 `build.json` entries). A bounded program has a constant
  tile count, so its loops unroll and no ring stage is reused, and its code, which every launch fetches cold, is a
  third to a half smaller than the general program's. A half tile skips the second 64-key half of V. Cluster size 7
  is not a variant: `num_ctas_per_token=7` is rejected with `ValueError`.
- Plan (`cake_backend.plan_ctas`): the candidates are the sizes among 8, 6, 5, 4, 3, 2 that give every CTA at least
  32 keys and leave no CTA without keys (`is_valid_split`) and whose `num_tokens` clusters are co-resident in one
  wave (`cudaOccupancyMaxActiveClusters` through each program's `max_active_clusters` occupancy entry, queried once
  per device). The plan takes the candidate with the least work per CTA (`split_cost`: 64-key halves of a 128-key
  tile up to 256 keys per CTA, whole tiles above), then the one with the fewest CTAs. Without a one-wave candidate
  it takes the smallest valid size in several waves, and 1 when no split qualifies. A forced split that leaves a CTA
  without keys is rejected.
- Dynamic shared memory per CTA: 218,880 bytes (`C = 1`) to 220,800 bytes (`C > 1`) for the general program and the
  192- and 256-key programs; 96,000 bytes (`C = 1`) or 129,152 to 130,688 bytes (`C > 1`) for the 64- and 128-key
  programs.
- `cake_backend.prepare_nvfp4_sparse_mla_decode(...)` returns a runner that launches without allocation (CUDA Graph
  safe); `cake_backend.generated_program_available(device)` reports whether the programs are delivered for the
  device; the tests and the benchmark skip the arm otherwise.

### Package layout

`cake_jit.py` is the registry: one `MODULES` record per program carrying the host plan (the architectures it is
delivered for, its source build keywords, its launch record, the argument plan of its `run` entry, the
kernel-parameter to logical-argument map, `cluster` and `max_keys`, the `tokens_by_cluster` grid rule, the name of
its occupancy entry and its workspace rule, `None`) and the loader glue (`program_max_keys`, `record_for`,
`select_module`, `launch_grid`, `launch_args`, `load`). The kernel reads raw rows through a TMA gather map that the
binding encodes from the `[rows, 352]` byte pool, passed as the logical argument `kv_rows` beside `kv` (the pool as
int32 words). `build.json` records per program the kernel, binding and occupancy-query units under
`csrc/cake_nvfp4_sparse_mla_decode/`, their digests and nvcc flags (`--use_fast_math` on every entry), one entry per
(program, architecture). `cake_build.py` loads a prebuilt library from `aot/` when a builder compiled one and
otherwise JIT-builds the manifest entry from the same sources and flags. `scripts/check_cake_build_manifests.py`
checks the manifest against the registry and the sources without importing the package. The hand-written
`backend="cuda"` sources stay at the root of `csrc/`, compiled by `jit.py`.

### Measured against `backend="cuda"`

Microseconds per launch, `backend="cuda"` against `backend="cake"` on the same inputs, 16 heads. P rows cover T from
5 to 45 at top-k 512, 1024 and 2048 without padding; W rows the wave-boundary token counts of each card's planner
table at top-k 2048; S rows suffix padding at top-k 2048 (short contexts, the tail of every third token empty).

Each launch is timed as its CUPTI kernel span, as in `flashinfer.testing.bench_gpu_time`, with a cold L2: the
persisting L2 is reset and a buffer of twice the L2 size is flushed before every launch. Each arm runs 20 warm-up
and 200 timed launches per group in 8 interleaved groups of alternating arm order; each value is the median over all
samples. B200 (148 SMs, `sm_100a`): driver 580.82.07; GB300 (152 SMs, `sm_103a`): driver 580.159.03; CUDA 13.4 and
torch 2.14.1+cu130 on both. Ratio = cake / cuda; `-` marks a wave-boundary row that exists only on the other card's
planner table. Relative error against the exact fp32 reference is 1.9e-3 to 3.5e-3 on every row for both
backends.

| row | T | topk | padding | B200 cuda us | B200 cake us | B200 ratio | GB300 cuda us | GB300 cake us | GB300 ratio |
|---|---|---|---|---|---|---|---|---|---|
| P | 5 | 2048 | none | 13.41 | 10.08 | 0.752 | 12.67 | 9.57 | 0.755 |
| P | 10 | 2048 | none | 13.79 | 10.43 | 0.756 | 12.93 | 9.86 | 0.762 |
| P | 15 | 2048 | none | 14.30 | 10.98 | 0.767 | 13.38 | 10.18 | 0.761 |
| P | 20 | 2048 | none | 17.25 | 13.98 | 0.811 | 16.10 | 12.99 | 0.807 |
| P | 25 | 2048 | none | 19.49 | 15.94 | 0.818 | 18.11 | 14.88 | 0.822 |
| P | 30 | 2048 | none | 22.27 | 16.32 | 0.733 | 20.71 | 15.10 | 0.729 |
| P | 35 | 2048 | none | 28.29 | 20.32 | 0.718 | 20.90 | 15.33 | 0.734 |
| P | 40 | 2048 | none | 28.61 | 20.64 | 0.721 | 26.50 | 18.88 | 0.713 |
| P | 45 | 2048 | none | 28.80 | 20.90 | 0.726 | 26.66 | 19.01 | 0.713 |
| P | 5 | 1024 | none | 9.70 | 7.68 | 0.792 | 9.22 | 7.36 | 0.799 |
| P | 10 | 1024 | none | 10.05 | 8.10 | 0.806 | 9.50 | 7.68 | 0.808 |
| P | 15 | 1024 | none | 10.43 | 8.51 | 0.816 | 9.82 | 8.00 | 0.814 |
| P | 20 | 1024 | none | 12.42 | 10.43 | 0.840 | 11.65 | 9.76 | 0.838 |
| P | 25 | 1024 | none | 13.57 | 11.33 | 0.835 | 12.77 | 10.72 | 0.840 |
| P | 30 | 1024 | none | 14.62 | 11.68 | 0.799 | 13.66 | 10.94 | 0.801 |
| P | 35 | 1024 | none | 17.57 | 14.50 | 0.825 | 13.92 | 11.20 | 0.805 |
| P | 40 | 1024 | none | 17.73 | 14.66 | 0.827 | 16.54 | 13.73 | 0.830 |
| P | 45 | 1024 | none | 17.95 | 14.91 | 0.831 | 16.70 | 13.86 | 0.830 |
| P | 5 | 512 | none | 9.60 | 6.85 | 0.713 | 9.18 | 6.53 | 0.711 |
| P | 10 | 512 | none | 9.89 | 7.20 | 0.728 | 9.41 | 6.78 | 0.721 |
| P | 15 | 512 | none | 10.08 | 7.52 | 0.746 | 9.54 | 7.10 | 0.745 |
| P | 20 | 512 | none | 10.40 | 8.38 | 0.806 | 9.86 | 7.97 | 0.808 |
| P | 25 | 512 | none | 10.62 | 8.54 | 0.804 | 10.05 | 8.13 | 0.809 |
| P | 30 | 512 | none | 10.78 | 8.90 | 0.825 | 10.18 | 8.45 | 0.830 |
| P | 35 | 512 | none | 12.77 | 10.59 | 0.829 | 10.30 | 8.45 | 0.820 |
| P | 40 | 512 | none | 12.86 | 10.88 | 0.846 | 12.10 | 10.08 | 0.833 |
| P | 45 | 512 | none | 13.09 | 11.04 | 0.844 | 12.32 | 10.34 | 0.839 |
| W | 16 | 2048 | none | 16.99 | 13.89 | 0.817 | 15.90 | 13.06 | 0.821 |
| W | 22 | 2048 | none | 17.44 | 14.40 | 0.826 | - | - | - |
| W | 23 | 2048 | none | 19.36 | 16.03 | 0.828 | 16.29 | 13.28 | 0.815 |
| W | 24 | 2048 | none | - | - | - | 18.08 | 14.94 | 0.827 |
| W | 26 | 2048 | none | 19.62 | 16.22 | 0.827 | - | - | - |
| W | 27 | 2048 | none | 22.14 | 16.32 | 0.737 | - | - | - |
| W | 28 | 2048 | none | - | - | - | 18.24 | 15.14 | 0.830 |
| W | 29 | 2048 | none | - | - | - | 20.67 | 15.14 | 0.732 |
| W | 33 | 2048 | none | 22.53 | 16.54 | 0.734 | - | - | - |
| W | 34 | 2048 | none | 28.22 | 20.26 | 0.718 | - | - | - |
| W | 36 | 2048 | none | - | - | - | 20.99 | 15.46 | 0.736 |
| W | 37 | 2048 | none | - | - | - | 26.40 | 18.78 | 0.711 |
| W | 46 | 2048 | none | 54.34 | 24.61 | 0.453 | 26.69 | 19.10 | 0.716 |
| W | 47 | 2048 | none | - | - | - | 50.43 | 22.85 | 0.453 |
| W | 64 | 2048 | none | 54.88 | 25.41 | 0.463 | 51.20 | 23.26 | 0.454 |
| W | 128 | 2048 | none | 82.50 | 48.80 | 0.592 | 76.61 | 44.77 | 0.584 |
| S | 20 | 2048 | suffix | 17.09 | 14.05 | 0.822 | 16.03 | 13.12 | 0.818 |
| S | 45 | 2048 | suffix | 28.77 | 20.74 | 0.721 | 26.82 | 18.94 | 0.706 |

Geomean ratio B200 0.759 (39 rows), GB300 0.756 (40 rows); largest ratio 0.846 (B200, the P row T = 40 topk = 512)
/ 0.840 (GB300, the P row T = 25 topk = 1024); `backend="cake"` is faster on every row of both cards: True.

Tracking: flashinfer-ai/flashinfer#5716 (DSA NVFP4 sparse MLA decode), tracker #4254.

## Graduation plan

1. Run the tests and the benchmark on B200/GB200 in CI.
2. vLLM: let the FlashInfer sparse-MLA backend accept `nvfp4_ds_mla` and call this API. The vLLM PR that does both is
   [vllm-project/vllm#59342](https://github.com/vllm-project/vllm/pull/59342).
3. The tcgen05 programs (`backend="cake"`) are faster than the `mma.sync` kernel at every measured shape on B200 and
   GB300; the next step is to make `backend="cake"` the default once CI runs it.
