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
    *, enable_in_kernel_fc2_reduce: bool = False, batch_invariant: bool = False
) -> List[Dict[str, Any]]:
    """W4A16 tactics across geometry, return mode and FC1 group size.

    M128/M256 use one/two-CTA instructions and two dequantization warp
    groups, with flag batch 4 and scheduler depth 2. Group sizes 512/1024
    trade phase-transition overhead against earlier FC2 work and token return.
    Opting into in-kernel reduction adds dispatch-return tactics.
    """
    if batch_invariant:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.a16_batch_invariant import (
            a16_batch_invariant_default_knobs,
        )

        base = a16_batch_invariant_default_knobs()
        return [
            {**base, "load_balance_mode": schedule, "token_back_mode": carrier}
            for schedule in ("static", "atomic_counter")
            for carrier in ("epi_warps", "reuse_dispatch_warps")
        ]
    return [
        dict(
            cluster_shape_mnk=(2, 1, 1),
            group_hint=group_hint,
            epi_flag_batch=(2, 4),
            load_balance_mode="atomic_counter",
            mma_tiler_mnk=tile,
            use_2cta_instrs=tile[0] == 256,
            flag_batch=4,
            token_back_mode=token_back,
            in_kernel_fc2_reduce=in_kernel,
            num_sched_stages=2,
        )
        for in_kernel in ((False, True) if enable_in_kernel_fc2_reduce else (False,))
        for group_hint in (512, 1024)
        for tile in ((128, 32, 256), (256, 32, 256), (256, 64, 256), (256, 128, 256))
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
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
) -> Dict[str, Any]:
    """Collectively tune W4A16 on staged BF16 inputs and prepared NVFP4 weights.

    Time the asynchronous full wrapper, including post-FC2 routing reduction,
    with one forward per private CUDA graph. The score is the maximum of
    each rank's median GPU-event time on the same hot, staged inputs.
    Output overwrites ``y``. Every EP rank must call outside graph capture.
    At least one eager preparation forward runs before each capture, even
    when ``warmup_iters=0``, to compile the integrated kernel.
    ``process_group`` selects the EP group; ``None`` uses the default group.
    Candidate failures abort the sweep; distributed failure recovery belongs
    to the caller, since preparation also allocates symmetric storage.
    """
    from .frontend import bf16_nvfp4_mega_moe

    def launch(*, sync: bool = False) -> None:
        bf16_nvfp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=sync,
        )

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        _session_candidates,
        autotune_knobs,
        resolve_gate_up_clamp,
        tuner,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.activation_batch_invariant import (
        batch_invariant_variant_kwargs,
    )
    from .frontend import MegaMoEBf16Nvfp4Inputs

    frontend = symm_buffer._frontend
    cfg = frontend.config
    overrides: Dict[str, Any] = {}
    if swiglu_alpha is not None or swiglu_beta is not None:
        overrides.update(swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    if clamp is not None:
        overrides["gate_up_clamp"] = clamp
    if candidates is None:
        candidates = bf16_nvfp4_candidates(
            enable_in_kernel_fc2_reduce=cfg.enable_in_kernel_fc2_reduce,
            batch_invariant=cfg.batch_invariant,
        )

    def prepare_launch():
        n = cfg.num_tokens_per_rank if num_tokens is None else num_tokens
        if type(n) is not int or not 0 <= n <= cfg.num_tokens_per_rank:
            raise ValueError("num_tokens must be within the workspace capacity")
        current = frontend.config
        if current.use_custom_finalize:
            if y is not None:
                raise ValueError("use_custom_finalize=True requires y=None")
        elif (
            y is None
            or y.dtype != torch.bfloat16
            or tuple(y.shape) != (n, cfg.hidden)
            or not y.is_cuda
            or not y.is_contiguous()
        ):
            raise ValueError(
                f"y must be contiguous CUDA bfloat16 with shape ({n}, {cfg.hidden})"
            )
        inputs = MegaMoEBf16Nvfp4Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l1[1],
            symm_buffer.fc1_alpha,
            transformed_l2[0],
            transformed_l2[1],
            symm_buffer.fc2_alpha,
            symm_buffer.kernel_combine_output,
            None if current.in_kernel_fc2_reduce else y,
            symm_buffer._staging_inputs,
            symm_buffer.fc1_norm_const if symm_buffer._use_fc1_norm_const else None,
        )
        # Compilation and argument conversion are local. Every EP rank agrees
        # on their success before the first eager communication-kernel launch.
        frontend._validate(inputs, n)
        mega = frontend._ensure_compiled(inputs)
        frontend._runtime_kwargs(inputs, mega)
        return launch

    def measure(candidate_launch):
        import torch.distributed as dist

        # Keep the full-wrapper graph-event timing used by ordinary W4A16.
        for _ in range(max(1, warmup_iters)):
            launch(sync=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=process_group)
        return statistics.median(
            _sample_graph_seconds(candidate_launch, timed_iters, process_group)
        )

    def record(winner, p50_s):
        from flashinfer.moe_ep.core.tuning import tuning_live_tokens
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import record_knobs

        current = frontend.config
        live_tokens = tuning_live_tokens(
            num_tokens, current.num_tokens_per_rank, process_group
        )
        if current.rank == 0:
            record_knobs(
                winner,
                dtype="bf16_nvfp4",
                world_size=current.world_size,
                hidden=current.hidden,
                intermediate=current.intermediate,
                num_experts=current.num_total_experts,
                topk=current.num_topk,
                max_tokens=current.num_tokens_per_rank,
                live_tokens=live_tokens,
                combine_dtype="bf16",
                apply_routing_weights_before_fc2=current.apply_routing_weights_before_fc2,
                use_custom_finalize=current.use_custom_finalize,
                gate_up_clamp=current.gate_up_clamp,
                batch_invariant=current.batch_invariant,
                handoff_norm=current.has_fc1_norm_const,
                **(
                    batch_invariant_variant_kwargs(current)
                    if current.batch_invariant
                    else {}
                ),
                p50_us=p50_s * 1e6,
                source="autotune_graph_events",
            )

    return autotune_knobs(
        frontend,
        launch,
        candidates,
        label="bf16_nvfp4_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=record,
        prepare_launch=prepare_launch,
        config_overrides=overrides,
        candidate_filter=lambda proposed: _session_candidates(
            proposed,
            cfg,
            tuner.is_valid_bf16_nvfp4_for_config,
            what="BF16/NVFP4 MegaMoE",
        ),
        process_group=process_group,
        measure_launch=measure,
    )


__all__ = ["autotune_bf16_nvfp4_mega_moe", "bf16_nvfp4_candidates"]
