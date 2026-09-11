# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""W4A16 candidate policy and collective full-forward graph timing."""

from __future__ import annotations

import math
import statistics
import warnings
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import torch


class _KnobFrontend(Protocol):
    def apply_knobs(self, knobs: Dict[str, Any]) -> None: ...


class _CollectiveGraphTimingError(RuntimeError):
    """A private graph timing failure that must stop the collective sweep."""


def _autotune_knobs_impl(
    frontend: _KnobFrontend,
    warmup_launch: Callable[[], None],
    sample_seconds: Callable[[int], Sequence[float]],
    candidates: List[Dict[str, Any]],
    *,
    label: str,
    warmup_iters: int,
    timed_iters: int,
    on_winner: Optional[Callable[[Dict[str, Any], float], None]],
) -> Dict[str, Any]:
    """Shared sweep; sample_seconds returns local seconds with no live graph."""
    if not candidates:
        raise ValueError("autotune_knobs needs a non-empty candidate list.")

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import ensure_not_capturing

    # The sweep owns host-side compile, allocation, timing and collectives;
    # it must finish before the caller captures its serving graph.
    ensure_not_capturing("knobs='auto' collective autotune sweep")

    import torch.distributed as dist

    collective = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if collective else 0

    def _barrier() -> None:
        if collective:
            dist.barrier()

    scores: List[float] = []
    for knobs in candidates:
        # A candidate failure (ctor reject / compile error) is deterministic
        # across ranks -- same static problem, same knobs -- so scoring it inf
        # keeps the collective iteration aligned.
        try:
            frontend.apply_knobs(knobs)
            _barrier()
            for _ in range(warmup_iters):  # first launch compiles
                warmup_launch()
            _barrier()
            scores.append(statistics.median(sample_seconds(timed_iters)))
        except _CollectiveGraphTimingError:
            # A peer may already be waiting in a captured collective. Never
            # advance to a different candidate after graph timing fails.
            raise
        except Exception as exc:  # noqa: BLE001 -- score-and-continue by design
            warnings.warn(
                f"[cutedsl-autotune] {label}: candidate {knobs} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            scores.append(math.inf)
        _barrier()

    t = torch.tensor(scores, dtype=torch.float64, device="cuda")
    if collective:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank = real latency
    best = int(torch.argmin(t).item())
    if not math.isfinite(float(t[best])):
        raise RuntimeError(
            f"[cutedsl-autotune] {label}: every candidate failed to compile/run."
        )
    winner = candidates[best]
    frontend.apply_knobs(winner)
    if on_winner is not None:
        on_winner(winner, float(t[best]))
    if rank == 0:
        ranked = sorted(zip(t.tolist(), candidates, strict=False), key=lambda kv: kv[0])
        summary = "\n".join(f"    {us * 1e6:10.1f} us  {knobs}" for us, knobs in ranked)
        print(
            f"[cutedsl-autotune] {label}: winner {winner} "
            f"({float(t[best]) * 1e6:.1f} us median, max across ranks) "
            f"out of {len(candidates)} candidates:\n{summary}",
            flush=True,
        )
    return winner


def w4a16_candidates() -> List[Dict[str, Any]]:
    """Twelve M256 W4A16 tactics with explicit scheduler depth 2.

    All geometries use two-CTA instructions and two dequantization warp
    groups. Explicit M128 configurations remain supported by the kernel.
    """
    return [
        dict(
            cluster_shape_mnk=(2, 1, 1),
            group_hint=512,
            epi_flag_batch=(2, 4),
            load_balance_mode="atomic_counter",
            mma_tiler_mnk=tile,
            use_2cta_instrs=True,
            flag_batch=flag_batch,
            token_back_mode=token_back,
            in_kernel_fc2_reduce=False,
            num_sched_stages=2,
        )
        for tile in ((256, 128, 256), (256, 64, 256), (256, 256, 256))
        for flag_batch in (4, 8)
        for token_back in ("epi_warps", "reuse_dispatch_warps")
    ]


def _sample_graph_seconds(
    launch: Callable[[], None], timed_iters: int
) -> Sequence[float]:
    """Own one full-forward graph until all replay work and reset complete."""
    import torch.distributed as dist

    from flashinfer.testing.utils import bench_gpu_time_with_cuda_event

    collective = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if collective else 0
    graph: Optional[torch.cuda.CUDAGraph] = None
    capture_error: Optional[Exception] = None
    try:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
        except Exception as exc:
            capture_error = exc

        # Capture records the collective but does not execute it. All ranks
        # must finish capture successfully before any rank starts replaying.
        ready = capture_error is None
        if collective:
            status = torch.tensor(int(ready), dtype=torch.int32, device="cuda")
            dist.all_reduce(status, op=dist.ReduceOp.MIN)
            ready = bool(status.item())
        if not ready:
            raise _CollectiveGraphTimingError(
                "W4A16 autotune graph capture failed on an EP rank."
            ) from capture_error
        assert graph is not None

        def select_own_rank(values: Sequence[float]) -> float:
            # Keep local samples: the sweep takes MAX of rank medians. The
            # event utility's default per-iteration MAX changes that metric.
            return values[rank]

        milliseconds = bench_gpu_time_with_cuda_event(
            graph.replay,
            dry_run_iters=0,
            repeat_iters=timed_iters,
            cold_l2_cache=False,
            sleep_after_run=False,
            input_args=(),
            input_kwargs={},
            aggregate_op=select_own_rank,
        )
        return [sample / 1000.0 for sample in milliseconds]
    except _CollectiveGraphTimingError:
        raise
    except Exception as exc:
        raise _CollectiveGraphTimingError(
            "W4A16 autotune graph timing failed; stop the collective sweep."
        ) from exc
    finally:
        if graph is not None:
            try:
                # apply_knobs frees the symmetric workspace. Even exception
                # tracebacks must not retain a graph referencing freed storage.
                torch.cuda.synchronize()
                graph.reset()
            except Exception as exc:
                raise _CollectiveGraphTimingError(
                    "W4A16 autotune graph cleanup failed; "
                    "stop tuning before applying another candidate."
                ) from exc


def autotune_w4a16_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Collectively tune W4A16 on staged BF16 inputs and prepared NVFP4 weights.

    Time the asynchronous full wrapper, including post-FC2 routing reduction,
    with one forward per private CUDA graph. The score is the maximum of
    each rank's median GPU-event time on the same hot, staged inputs.
    Output overwrites ``y``. Every EP rank must call outside graph capture.
    At least one eager preparation forward runs before each capture, even
    when ``warmup_iters=0``, to compile the fused kernel and reducer.
    """
    from .frontend import w4a16_mega_moe

    def launch(*, sync: bool) -> None:
        w4a16_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=sync,
        )

    def warmup_launch() -> None:
        launch(sync=True)

    def launch_async() -> None:
        launch(sync=False)

    def sample_seconds(count: int) -> Sequence[float]:
        return _sample_graph_seconds(launch_async, count)

    cfg = symm_buffer._frontend.config

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        if cfg.rank == 0:
            from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import record_knobs

            record_knobs(
                winner,
                dtype="w4a16",
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                combine_dtype="bf16",
                p50_us=p50_s * 1e6,
                source="autotune_graph_events",
            )

    return _autotune_knobs_impl(
        symm_buffer._frontend,
        warmup_launch,
        sample_seconds,
        w4a16_candidates() if candidates is None else candidates,
        label="w4a16_mega",
        warmup_iters=max(1, warmup_iters),
        timed_iters=timed_iters,
        on_winner=_record,
    )


__all__ = ["autotune_w4a16_mega_moe", "w4a16_candidates"]
