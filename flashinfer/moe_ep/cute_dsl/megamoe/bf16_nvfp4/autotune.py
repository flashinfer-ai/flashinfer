# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""W4A16 candidate policy and collective full-forward graph timing."""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch


class _CollectiveGraphTimingError(RuntimeError):
    """A private graph timing failure that must stop the collective sweep."""


def bf16_nvfp4_candidates(
    *, enable_in_kernel_fc2_reduce: bool = False
) -> List[Dict[str, Any]]:
    """Four M256 W4A16 tactics with flag batch 4 and scheduler depth 2.

    Both geometries use two-CTA instructions and two dequantization warp
    groups. Explicit M128 configurations remain supported by the kernel.
    Opting into in-kernel reduction adds two dispatch-return tactics.
    """
    return [
        dict(
            cluster_shape_mnk=(2, 1, 1),
            group_hint=512,
            epi_flag_batch=(2, 4),
            load_balance_mode="atomic_counter",
            mma_tiler_mnk=tile,
            use_2cta_instrs=True,
            flag_batch=4,
            token_back_mode=token_back,
            in_kernel_fc2_reduce=in_kernel,
            num_sched_stages=2,
        )
        for in_kernel in ((False, True) if enable_in_kernel_fc2_reduce else (False,))
        for tile in ((256, 128, 256), (256, 64, 256))
        for token_back in ("epi_warps", "reuse_dispatch_warps")
        if not in_kernel or token_back == "reuse_dispatch_warps"
    ]


def _sample_graph_seconds(
    launch: Callable[[], None], timed_iters: int, process_group: Any = None
) -> Sequence[float]:
    """Own one full-forward graph until all replay work and reset complete."""
    import torch.distributed as dist

    collective = dist.is_available() and dist.is_initialized()
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
            dist.all_reduce(status, op=dist.ReduceOp.MIN, group=process_group)
            ready = bool(status.item())
        if not ready:
            raise _CollectiveGraphTimingError(
                "W4A16 autotune graph capture failed on an EP rank."
            ) from capture_error
        assert graph is not None

        # Keep the event helper's six untimed replays, but sample locally:
        # its implicit WORLD gathers would include ranks outside this EP group.
        for _ in range(6):
            graph.replay()
        torch.cuda.synchronize()
        if collective:
            dist.barrier(group=process_group)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        return [
            start.elapsed_time(end) / 1000.0
            for start, end in zip(starts, ends, strict=True)
        ]
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


def autotune_bf16_nvfp4_mega_moe(
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
    process_group: Any = None,
) -> Dict[str, Any]:
    """Collectively tune W4A16 on staged BF16 inputs and prepared NVFP4 weights.

    Time the asynchronous full wrapper, including post-FC2 routing reduction,
    with one forward per private CUDA graph. The score is the maximum of
    each rank's median GPU-event time on the same hot, staged inputs.
    Output overwrites ``y``. Every EP rank must call outside graph capture.
    At least one eager preparation forward runs before each capture, even
    when ``warmup_iters=0``, to compile the fused kernel and reducer.
    ``process_group`` selects the EP group; ``None`` uses the default group.
    Candidate failures abort the sweep; distributed failure recovery belongs
    to the caller, since preparation also allocates symmetric storage.
    """
    from .frontend import bf16_nvfp4_mega_moe

    def launch(*, sync: bool) -> None:
        bf16_nvfp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=sync,
        )

    def launch_async() -> None:
        launch(sync=False)

    cfg = symm_buffer._frontend.config
    frontend = symm_buffer._frontend
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import _session_candidates, tuner

    if candidates is None:
        candidates = bf16_nvfp4_candidates(
            enable_in_kernel_fc2_reduce=cfg.enable_in_kernel_fc2_reduce
        )
    candidates = _session_candidates(
        candidates,
        cfg,
        tuner.is_valid_bf16_nvfp4_for_config,
        what="BF16/NVFP4 MegaMoE",
    )
    warmup_iters = max(1, warmup_iters)
    label = "bf16_nvfp4_mega"
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import ensure_not_capturing

    # The sweep owns host-side compile, allocation, timing and collectives;
    # it must finish before the caller captures its serving graph.
    ensure_not_capturing("knobs='auto' collective autotune sweep")

    import torch.distributed as dist

    collective = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank(group=process_group) if collective else 0

    def _barrier() -> None:
        if collective:
            dist.barrier(group=process_group)

    scores: List[float] = []
    for knobs in candidates:
        # Rank-local allocation/compile errors need not be deterministic.
        # Abort the sweep; continuing could enter a different collective from
        # a peer still preparing or launching this candidate.
        frontend.apply_knobs(knobs)
        _barrier()
        for _ in range(warmup_iters):  # first launch compiles
            launch(sync=True)
        _barrier()
        scores.append(
            statistics.median(
                _sample_graph_seconds(launch_async, timed_iters, process_group)
            )
        )
        _barrier()

    t = torch.tensor(scores, dtype=torch.float64, device="cuda")
    if collective:
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=process_group)
    best = int(torch.argmin(t).item())
    winner = candidates[best]
    frontend.apply_knobs(winner)
    p50_s = float(t[best])
    if cfg.rank == 0:
        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import record_knobs

        record_knobs(
            winner,
            dtype="bf16_nvfp4",
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


__all__ = ["autotune_bf16_nvfp4_mega_moe", "bf16_nvfp4_candidates"]
