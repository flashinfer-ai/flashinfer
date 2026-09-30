"""Shared timing and numerical correctness helpers for the FP6 benchmarks."""

from __future__ import annotations

import math
import statistics
from typing import Callable, List

import torch
import torch.nn.functional as F

from benchmarks.experimental.b12x.common import (  # noqa: F401  (re-exported for the fp6 benches)
    make_l2_flush_fn,
    resolve_l2_flush_bytes,
)


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


def fmt_us(times_ms: List[float]) -> str:
    med = statistics.median(times_ms) * 1000
    mn = min(times_ms) * 1000
    return f"{med:7.1f} us (min {mn:.1f})"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return F.cosine_similarity(a_f, b_f, dim=0).item()


def check_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
    cosine_threshold: float,
    rel_rmse_threshold: float = 0.15,
) -> None:
    """Correctness gate: finite values, direction (cosine), and scale (rel RMSE).

    Cosine similarity alone is invariant to positive scaling, so a global
    scale/dequant bug (e.g. a UE8M0 exponent off by one = exactly 2x) would
    pass it with cos=1.0. ``rel_rmse`` closes that hole: it is the L2-relative
    error ``||candidate - reference|| / ||reference||`` (normalized by the
    reference RMS, not mean-abs, so it relates directly to cosine:
    ``rel ~= sqrt(2 * (1 - cos))`` at matched scale). The 0.15 default is the
    L2 analog of ``cosine_threshold=0.99`` plus scale headroom: known-good
    W6A8-vs-BF16 comparisons measure ~0.065, while any realistic scale bug is
    a power of two and lands at >= 0.5.
    """
    cand_finite = bool(torch.isfinite(candidate).all().item())
    ref_finite = bool(torch.isfinite(reference).all().item())
    if not cand_finite or not ref_finite:
        raise CorrectnessError(
            f"non-finite output detected during correctness check vs {label}: "
            f"candidate_finite={cand_finite}, reference_finite={ref_finite}"
        )
    diff = (candidate.float() - reference.float()).abs()
    max_abs = diff.max().item()
    rmse = diff.square().mean().sqrt().item()
    ref_rms = reference.float().square().mean().sqrt().item()
    if ref_rms > 0.0:
        rel_rmse = rmse / ref_rms
    else:
        # All-zero reference: the candidate must also be (exactly) zero.
        rel_rmse = 0.0 if rmse == 0.0 else math.inf
    cos = cosine_similarity(candidate, reference)
    print(
        f"    check vs {label}: max_abs={max_abs:.8f} rmse={rmse:.8f} "
        f"rel_rmse={rel_rmse:.6f} cos={cos:.10f}"
    )
    if not math.isfinite(cos):
        raise CorrectnessError(
            f"cosine similarity vs {label} is non-finite: "
            f"max_abs={max_abs:.8f}, rmse={rmse:.8f}, cos={cos}"
        )
    if cos < cosine_threshold:
        raise CorrectnessError(
            f"cosine similarity vs {label} fell below threshold "
            f"{cosine_threshold:.6f}: got {cos:.10f}"
        )
    if rel_rmse > rel_rmse_threshold:
        raise CorrectnessError(
            f"relative RMSE vs {label} exceeded threshold "
            f"{rel_rmse_threshold:.4f}: got {rel_rmse:.6f} "
            f"(rmse={rmse:.8f}, ref_rms={ref_rms:.8f})"
        )


def unswizzled_ue8m0_grid(w_bf16: torch.Tensor) -> torch.Tensor:
    """Per-K/32 UE8M0 scale bytes, unswizzled ``[E, rows, K//32]`` uint8.

    Mirrors the offline quantizer's block-scale rule
    (``_swizzled_block_scales`` minus the swizzle): the packed codes were
    quantized against exactly these exponents, and ``prepare_weights``
    (``prepare_w6a8_mxfp6_weights``) applies the MMA swizzle itself. Shared
    here so the two FP6 MoE benches cannot silently diverge from the rule.
    """
    from b12x._lib.fp6 import (
        FLOAT6_E2M3_MAX,
        SF_VEC_SIZE_FP6,
        _ue8m0_scale_from_block_max,
    )

    e, rows, cols = w_bf16.shape
    blocks = cols // SF_VEC_SIZE_FP6
    block_max = (
        w_bf16.float().abs().view(e, rows, blocks, SF_VEC_SIZE_FP6).amax(dim=-1)
    )
    return _ue8m0_scale_from_block_max(block_max, FLOAT6_E2M3_MAX).contiguous()


def bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n):
    """Reference BF16 gated-SiLU MoE grouped by expert ([up; gate] row order).

    Row order matches the fused kernel's ``w13_layout="w13"`` contract
    (``_gated_row_slices``): FC1 rows ``[0:N]`` are the up projection and rows
    ``[N:2N]`` are the gate, i.e. ``inter = silu(h[:, n:]) * h[:, :n]``.
    """
    m, k = x.shape
    out = torch.zeros(m, k, device=x.device, dtype=torch.float32)
    xf = x.float()
    for e in range(w1_bf.shape[0]):
        sel = topk_ids == e
        if not bool(sel.any()):
            continue
        rows, cols = sel.nonzero(as_tuple=True)
        xe = xf[rows]
        h = xe @ w1_bf[e].float().T
        up, gate = h[:, :n], h[:, n:]
        inter = F.silu(gate) * up
        down = inter @ w2_bf[e].float().T
        out.index_add_(0, rows, down * topk_weights[rows, cols].float().unsqueeze(1))
    return out


def capture_graph_replay(fn: Callable[[], None]) -> Callable[[], None]:
    # Warm eager launch state before capture so compile/cache work does not
    # leak into the replay measurement.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    replay._b12x_benchmark_call = fn
    return replay
