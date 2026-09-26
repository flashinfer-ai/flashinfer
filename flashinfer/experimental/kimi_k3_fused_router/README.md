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
- The program is a family of kernels. A per-architecture table maps each
  exact `(num_tokens, block_m)` shape of the routed set,
  `num_tokens in {1, 2, 4, ..., 8192}` (powers of two) x `block_m in {8, 16}`,
  to one dispatch arm (a single cluster of 2, 4 or 8 CTAs exchanging the
  selected ids through distributed shared memory for the smallest batches,
  one-join plan builders for small and medium batches, a 4-CTA-cluster
  variant at four CTAs per SM for 512 to 2048 tokens and a two-join
  persistent kernel for the largest batches). Other shapes raise
  `NotImplementedError`.
- Every arm but the small-batch cluster one is a cooperative persistent launch
  whose grid is bounded by the device SM count (three CTAs per SM on CC 10.0,
  four on CC 10.3; the 4-CTA-cluster arm and the largest-batch arm use their
  own launch bounds of four (or six on CC 10.3 for the largest batches) CTAs
  per SM, and the one-join arm launches at least its 128 plan-owner CTAs).
  The 4-CTA-cluster arm is additionally bounded by the number of
  co-resident clusters the driver reports for the kernel
  (`cudaOccupancyMaxActiveClusters`, queried once at preparation through a
  small helper linked next to the generated binding).
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

num_tokens, block_m = 1024, 8
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
shape, loads the arm's JIT module and computes the launch grid; it makes no
host copy of the inputs. `cake_backend.generated_program_available(device)`
reports whether the generated program for the device architecture is
registered in this checkout.

Limits of the current route: FP32 logits and bias, exactly 896 experts and
top-16, the 28 routed `(num_tokens, block_m)` shapes listed above, SM100
(B200) and SM103 (B300 / GB300) only. Finite logits are expected.

See `tests/experimental/test_cake_kimi_k3_fused_router.py` for the torch
reference and validated shape set (all 28 routed shapes, CUDA Graph replay,
no-allocation launch) and `benchmarks/bench_cake_kimi_k3_fused_router.py` for
the benchmark against SGLang's `moe_route_radix` + `moe_align_block_size`
route on the same tensors (when `sglang` is importable).
