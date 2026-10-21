# Experimental Kimi-K3 fused MoE router (generated program)

Both this API and its generated-program backend are experimental and may
change without backward compatibility. Calling the API explicitly opts into
the experimental feature and emits FlashInfer's experimental API warning.
There is no automatic backend selection.

`flashinfer.fused_moe.kimi_k3_fused_router(logits, bias, block_m=...)` routes
FP32 gate logits of the Kimi-K3 MoE layer (**896 experts, top-16**) and builds
the expert-aligned route plan consumed by grouped MoE GEMMs in **one launch**:

- `scores = sigmoid(logits)`; the 16 experts with the largest
  `scores + bias` are selected (exact ties resolve toward the lower expert
  id); `topk_ids` are returned in ascending order and `topk_weights` are the
  selected sigmoid scores divided by their sum (a zero sum divides by one).
  The sigmoid and the division reproduce the SGLang router's `__expf` /
  `__fdividef` arithmetic, so `scores` can differ from `torch.sigmoid` in the
  last FP32 bit (and the rounding of that bit can differ between SM100 and
  SM103). Two candidates whose biased scores agree to within `2**-22` may
  therefore be ordered either way at the top-16 boundary; the tests accept
  exactly that tolerance and nothing else.
- The route plan follows the `moe_align_block_size` layout: pair indices
  `token * 16 + route` grouped by expert in ascending pair order, every expert
  segment padded to a multiple of `block_m` (8 or 16) with the sentinel
  `num_tokens * 16`, one `expert_ids` entry per block,
  `num_tokens_post_padded` naming the valid extent, plus per-expert
  `expert_counts`, padded `expert_offsets` (897 entries) and
  `expert_scatter_offsets` (equal to the counts after the launch).
- The program is a family of kernels. A per-architecture table measured at
  the 28 cells `num_tokens in {1, 2, 4, ..., 8192}` (powers of two) x
  `block_m in {8, 16}` names one dispatch arm per cell; every other token
  count takes the arm of the next measured cell up (the arms read
  `num_tokens` at runtime), so **any `num_tokens` from 1 to 8192** is served;
  larger counts are rejected (every arm rebuilds the route plan from
  per-expert token bitmaps sized for 8192 rows). The arms: for the smallest batches one kernel per
  token count (a single CTA for one token, otherwise one cluster of
  `num_tokens` CTAs -- 2, 4, 8 or 16 -- exchanging the selected ids through
  distributed shared memory; these are the only kernels bound to their exact
  token count, so 3 or 17 tokens take the 32-token cell's arm), one-join plan
  builders for up to 128 tokens (the 32- to 128-token cells use a second
  kernel of the small-batch builder's family which loads the bias and the
  first row's logits into registers before its prologue barrier; the plain
  builder keeps 128 tokens x `block_m = 16` on both architectures plus 128
  tokens x `block_m = 8` on SM100 and 64 tokens x `block_m = 16` on SM103), a
  one-join bitmap builder for the 256-token cell (129 to 256 tokens; its owner
  scratch holds 256 rows on SM100 and 2048 on SM103), a 4-CTA-cluster variant
  at four CTAs per SM for 257 to 2048 tokens (the 2048-token cell uses a
  second kernel of that family which prefetches the next row's logits into
  registers) and a warp-per-row two-join persistent kernel from 2049 tokens
  up. The two architectures' tables differ only in two small-batch cells.
- Every arm but the small-batch per-token-count one is a cooperative persistent launch
  whose grid is bounded by the device SM count (three CTAs per SM on CC 10.0,
  four on CC 10.3; the 4-CTA-cluster arms and the largest-batch arm use their
  own launch bounds of four CTAs per SM, and the two small-batch one-join
  arms launch at least their 128 plan-owner CTAs).
  The two 4-CTA-cluster arms are additionally bounded by the number of
  co-resident clusters the driver reports for the kernel
  (`cudaOccupancyMaxActiveClusters`, queried once per device and program
  through a small helper linked next to the generated binding of exactly
  those programs). The 16-CTA cluster of the `num_tokens = 16` kernel exceeds
  the portable cluster size; the generated program sets the non-portable
  cluster-size attribute on that kernel before launching it, and preparation
  requires the driver to admit at least one such cluster on the device.
- One source per kernel serves both architectures (architecture-specific
  lines sit behind `__CUDA_ARCH__` guards) and both route block alignments
  (`-DBLOCK_M=8` / `-DBLOCK_M=16` on the compile line); the registry in
  `cake_jit.py` lists each program once and maps the per-architecture
  dispatch keys to it.
- Nothing is planned on the host and nothing is allocated at launch: a
  prepared runner (or a CUDA Graph capturing it) replays for new `logits` /
  `bias` values written into the same buffers.

| Tensor | Shape | dtype |
| --- | --- | --- |
| `logits` | `[num_tokens, 896]` | float32 |
| `bias` | `[896]` | float32 |
| `topk_weights` | `[num_tokens, 16]` | float32 |
| `topk_ids` | `[num_tokens, 16]` | int32 |
| `sorted_token_ids` | `>= max_route_blocks * block_m` | int32 |
| `expert_ids` | `>= max_route_blocks` | int32 |
| `num_tokens_post_padded` | `[1]` | int32 |
| `expert_counts`, `expert_scatter_offsets` | `[896]` | int32 |
| `expert_offsets` | `[897]` | int32 |

`max_route_blocks = min(896, pairs) + (pairs - min(896, pairs)) // block_m`
with `pairs = num_tokens * 16` (`cake_backend.max_route_blocks`).

```python
import torch
from flashinfer.fused_moe import (
    allocate_kimi_k3_route_plan,
    kimi_k3_fused_router,
    prepare_kimi_k3_fused_router,
)

num_tokens, block_m = 1000, 8
logits = torch.randn(num_tokens, 896, device="cuda")
bias = 0.05 * torch.randn(896, device="cuda")

plan = kimi_k3_fused_router(logits, bias, block_m=block_m)  # allocates the plan

# Steady state / CUDA Graph: bind a caller-owned plan once, launch many times.
plan = allocate_kimi_k3_route_plan(num_tokens, block_m, logits.device)
runner = prepare_kimi_k3_fused_router(logits, bias, block_m=block_m, plan=plan)
runner()  # launches on the current stream, returns plan
logits.copy_(torch.randn_like(logits))
runner()  # same runner (or a captured graph), new logits
```

Preparation validates shapes, dtypes, contiguity, plan capacity and that no
two buffers overlap, selects the dispatch arm for the device architecture and
token count, loads the arm's JIT module and computes the launch grid from the
device facts cached per device (compute capability, SM count, cluster
occupancy); it makes no host copy of the inputs.
`cake_backend.generated_program_available(device)` reports whether the
generated program for the device architecture is registered in this checkout.

Limits of the current route: FP32 logits and bias, exactly 896 experts and
top-16, `block_m` 8 or 16, SM100 (B200) and SM103 (B300 / GB300) only. Finite
logits are expected. Token counts between the 28 measured cells are served by
the neighbouring cell's arm and validated for correctness; their timings were
measured at the cells. Token counts above 8192 raise `ValueError`.

See `tests/experimental/test_cake_kimi_k3_fused_router.py` for the torch
reference and validated shape set (all 28 measured cells, token counts between
them, CUDA Graph replay, no-allocation launch) and
`benchmarks/bench_cake_kimi_k3_fused_router.py` for the benchmark against
SGLang's `moe_route_radix` + `moe_align_block_size` route on the same tensors
(when `sglang` is importable).
