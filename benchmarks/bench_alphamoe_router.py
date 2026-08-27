"""Benchmark the VibeCUDA AlphaMoE router against FlashInfer main.

The optimized CAKE baseline is still under review in PR #4339. Its separately
verified result is reported in the pull request description; this benchmark
does not vendor or compile unpublished reference code.
"""

import functools
from dataclasses import dataclass

import numpy as np
import torch

from flashinfer.fused_moe import allocate_alphamoe_route_plan, alphamoe_fused_router
from flashinfer.testing import bench_gpu_time
from flashinfer.trace.templates.moe import _alphamoe_fused_router_reference

DRY_RUN_ITERS = 5
REPEAT_ITERS = 10


@dataclass(frozen=True)
class RouterConfig:
    name: str
    num_tokens: int
    num_experts: int
    top_k: int
    block_m: int
    has_shared_expert: bool


CONFIGS = (
    RouterConfig("single-1tok-e512-shared", 1, 512, 2, 16, True),
    RouterConfig("decode-8tok-e257-shared", 8, 257, 9, 8, True),
    RouterConfig("batch-32tok-e512", 32, 512, 8, 16, False),
    RouterConfig("batch-128tok-e512", 128, 512, 8, 16, False),
)


def _make_logits(cfg: RouterConfig, case_index: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(29001 + case_index)
    return torch.randn(
        cfg.num_tokens,
        cfg.num_experts,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )


def _timed_us(fn, args: tuple) -> float:
    samples = bench_gpu_time(
        fn,
        input_args=args,
        enable_cupti=True,
        dry_run_iters=DRY_RUN_ITERS,
        repeat_iters=REPEAT_ITERS,
        cold_l2_cache=True,
        use_cuda_graph=False,
    )
    return float(np.median(samples)) * 1e3


def _assert_matches_reference(
    cfg: RouterConfig, candidate: tuple[torch.Tensor, ...], reference: tuple
) -> None:
    torch.testing.assert_close(candidate[0], reference[0], rtol=1e-5, atol=1e-7)
    for index in (1, 3, 4, 5, 6, 7):
        torch.testing.assert_close(candidate[index], reference[index])

    # Scatter order within one expert is intentionally unspecified. Compare
    # routed token ids as multisets and require every padding slot to be the
    # documented sentinel.
    sorted_ids = candidate[2]
    ref_sorted_ids = reference[2]
    counts = candidate[5].tolist()
    offsets = candidate[6].tolist()
    sentinel = cfg.num_tokens * cfg.top_k
    for expert, count in enumerate(counts):
        begin, end = offsets[expert], offsets[expert + 1]
        got, _ = torch.sort(sorted_ids[begin : begin + count])
        want, _ = torch.sort(ref_sorted_ids[begin : begin + count])
        torch.testing.assert_close(got, want)
        padding = sorted_ids[begin + count : end]
        if padding.numel() and not bool((padding == sentinel).all()):
            raise AssertionError(f"expert {expert}: non-sentinel padding")


def _bench_one(cfg: RouterConfig, case_index: int) -> float:
    logits = _make_logits(cfg, case_index)
    kwargs = {
        "top_k": cfg.top_k,
        "block_m": cfg.block_m,
        "has_shared_expert": cfg.has_shared_expert,
    }
    reference = _alphamoe_fused_router_reference(logits, **kwargs)
    candidate = alphamoe_fused_router(logits, backend="vibecuda", **kwargs)
    _assert_matches_reference(cfg, candidate, reference)

    plan = allocate_alphamoe_route_plan(logits, **kwargs)
    planned_candidate = alphamoe_fused_router(logits, plan=plan, backend="vibecuda")
    _assert_matches_reference(cfg, planned_candidate, reference)
    torch.cuda.synchronize()

    candidate_fn = functools.partial(
        alphamoe_fused_router, plan=plan, backend="vibecuda"
    )
    reference_fn = functools.partial(_alphamoe_fused_router_reference, **kwargs)
    candidate_us = _timed_us(candidate_fn, (logits,))
    reference_us = _timed_us(reference_fn, (logits,))
    speedup = reference_us / candidate_us
    print(
        f"{cfg.name:28s} reference {reference_us:10.2f} us  "
        f"vibecuda {candidate_us:8.2f} us  {speedup:8.2f}x"
    )
    return speedup


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError(f"CC 10.0 or 10.3 required, got {capability}")

    print(
        "CUPTI, cold L2, eager replay, "
        f"dry_run={DRY_RUN_ITERS}, repeats={REPEAT_ITERS}"
    )
    speedups = np.asarray(
        [_bench_one(config, index) for index, config in enumerate(CONFIGS)]
    )
    print(f"arithmetic mean: {float(speedups.mean()):.4f}x")
    print(f"geometric mean: {float(np.exp(np.log(speedups).mean())):.4f}x")


if __name__ == "__main__":
    main()
