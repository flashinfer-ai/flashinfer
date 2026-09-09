# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Online (warmup-time) knob autotuning for the CuTeDSL MegaMoE frontends.

Times a curated candidate knob set on the live problem and applies the winner
to the session's frontend, replacing the static two-profile heuristic in
:mod:`.tuner` with a measured choice.  The candidate space mirrors the
restricted sweep used with the kernel team's tester
(``tester.tester --sweep --use_knob ...``).  For NVFP4 it includes
``in_kernel_fc2_reduce`` (the tester's overall winners at 8 and 2048 tokens
are in-flight-reduce candidates): the symm buffer's ``output_activation`` is
always sym-heap allocated, so the knob can flip per-compile.  Note an ikr
winner makes the session's output nondeterministic in accumulation order;
callers that need bit-reproducible outputs should pin
``in_kernel_fc2_reduce=False`` via explicit knobs instead of autotuning.
For MXFP8 the knob stays owned by the config / caller.

The tune is a COLLECTIVE operation: the mega kernel's dispatch/combine spans
all EP ranks, so every rank must call the autotune entry point in the same
iteration with the same candidate list.  Ranks compile and launch each
candidate in lockstep (barriers around compile and timing), and the winner is
agreed on by all-reducing per-candidate times with MAX (the slowest rank is
the real latency of a collective kernel) — the argmin index is then identical
everywhere.

Cost: one ``cute.compile`` per candidate (minutes each), paid once per
session at the first launch.  Narrow ``candidates`` to trade quality for
startup time.
"""

from __future__ import annotations

import math
import statistics
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import torch

from .tuner import default_knobs, is_valid

# Shared base of the sweep restriction (values that won every profile so far).
_SWEEP_BASE: Dict[str, Any] = {
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "epi_flag_batch": (2, 4),
    "load_balance_mode": "atomic_counter",
}


def nvfp4_candidates(
    *,
    combine_format: str = "bf16",
    allow_in_kernel_fc2_reduce: bool = True,
) -> List[Dict[str, Any]]:
    """Default NVFP4 candidate knob dicts (tile x flag_batch x token-back x ikr).

    24 candidates for the default bf16 combine (the ikr axis doubles the
    12-candidate sweep and with it the one-time compile cost); quantized
    ``combine_format`` values prune to the valid subset (dispatch-warp
    token-back only, no ikr).  Pass ``allow_in_kernel_fc2_reduce=False`` when
    the session cannot run ikr (``apply_topk_in_fc1=False``) or must stay
    deterministic.
    """
    out: List[Dict[str, Any]] = []
    for tile in ((256, 128, 256), (256, 256, 256)):
        for flag_batch in (4, 8):
            for token_back in (
                "epi_warps",
                "standalone_warps",
                "reuse_dispatch_warps",
            ):
                for ikr in (False, True) if allow_in_kernel_fc2_reduce else (False,):
                    knobs = dict(
                        _SWEEP_BASE,
                        mma_tiler_mnk=tile,
                        flag_batch=flag_batch,
                        token_back_mode=token_back,
                        in_kernel_fc2_reduce=ikr,
                    )
                    if is_valid(knobs, combine_format=combine_format):
                        out.append(knobs)
    return out


def mxfp8_candidates(
    *,
    in_kernel_fc2_reduce: bool = False,
) -> List[Dict[str, Any]]:
    """Default MXFP8 candidate knob dicts (4: flag_batch x token-back).

    The MXFP8 kernel's tile is fixed at ``(256, 256)`` so no tile axis, and its
    config exposes token-back as the ``token_back_by_dispatch`` bool, so the
    two dispatch-warp modes collapse to one candidate.  The ikr knob stays
    owned by the config (unlike NVFP4, the MXFP8 kernel rejects ikr together
    with dispatch-warp token-back); pass the session's value so those combos
    are pruned instead of failing at compile.
    """
    out: List[Dict[str, Any]] = []
    for flag_batch in (4, 8):
        for token_back in ("epi_warps", "reuse_dispatch_warps"):
            if in_kernel_fc2_reduce and token_back != "epi_warps":
                continue
            knobs = dict(
                _SWEEP_BASE,
                flag_batch=flag_batch,
                token_back_mode=token_back,
            )
            if is_valid(knobs):
                out.append(knobs)
    return out


def bf16_candidates() -> List[Dict[str, Any]]:
    """Return the currently supported BF16 tuning candidate.

    This is intentionally a one-entry autotune surface. Keeping the same
    collective autotune lifecycle as the other Mega kernels means additional
    validated geometries can be added without changing the public API.
    """
    return [default_knobs(0, dtype="bf16")]


def w4a16_candidates() -> List[Dict[str, Any]]:
    """Sixteen depth3 tactics and two curated depth2 W4A16 variants.

    The kernel keeps two dequantization warp groups and derives its pipeline
    depths from the existing resource fitters. Precision, clamps and the
    post-FC2 routing/reduction contract are unchanged across candidates.
    """
    out = [
        dict(
            _SWEEP_BASE,
            mma_tiler_mnk=tile,
            cluster_shape_mnk=cluster,
            use_2cta_instrs=tile[0] == 256,
            flag_batch=flag_batch,
            token_back_mode=token_back,
            in_kernel_fc2_reduce=False,
            num_sched_stages=3,
        )
        for tile, cluster in (
            ((256, 128, 256), (2, 1, 1)),
            ((256, 64, 256), (2, 1, 1)),
            ((128, 64, 256), (2, 1, 1)),
            ((128, 64, 256), (1, 1, 1)),
        )
        for flag_batch in (4, 8)
        for token_back in ("epi_warps", "reuse_dispatch_warps")
    ]

    # Explicit depth3 above resets the config after a depth2 candidate.
    out += [
        dict(knobs, num_sched_stages=2)
        for knobs in out
        if knobs["cluster_shape_mnk"] == (2, 1, 1)
        and (knobs["mma_tiler_mnk"], knobs["flag_batch"], knobs["token_back_mode"])
        in (
            ((256, 64, 256), 8, "epi_warps"),
            ((256, 128, 256), 4, "reuse_dispatch_warps"),
        )
    ]
    return out


def autotune_knobs(
    frontend: Any,
    launch: Callable[[], None],
    candidates: List[Dict[str, Any]],
    *,
    label: str,
    warmup_iters: int = 3,
    timed_iters: int = 10,
    on_winner: Optional[Callable[[Dict[str, Any], float], None]] = None,
) -> Dict[str, Any]:
    """Time each candidate on the live problem and apply the winner.

    ``frontend`` is a NVFP4/MXFP8 mega frontend (must have ``apply_knobs``);
    ``launch`` is a zero-arg closure that runs one synchronized forward with
    the caller's real staged inputs (e.g. a ``nvfp4_mega_moe(...)`` call).

    ``on_winner`` (optional) is called once with ``(winner, p50_seconds)``
    after the winner is applied — used to persist the result in the knob
    cache. It runs on every rank; the callback decides who writes.

    COLLECTIVE: every EP rank must call this in the same iteration with the
    same ``candidates`` (order included).  Returns the winning knob dict.
    """

    def sample_seconds(count: int) -> Sequence[float]:
        samples = []
        for _ in range(count):
            start = time.perf_counter()
            launch()  # The exported contract is a synchronized forward.
            samples.append(time.perf_counter() - start)
        return samples

    return _autotune_knobs_impl(
        frontend,
        launch,
        sample_seconds,
        candidates,
        label=label,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=on_winner,
    )


class _KnobFrontend(Protocol):
    def apply_knobs(self, knobs: Dict[str, Any]) -> None: ...


class _CollectiveGraphTimingError(RuntimeError):
    """A private graph timing failure that must stop the collective sweep."""


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

    from .comm import ensure_not_capturing

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


def autotune_nvfp4_mega_moe(
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
    """Autotune the NVFP4 mega session on the caller's staged inputs.

    Arguments mirror :func:`.nvfp4.nvfp4_mega_moe`; ``y`` is clobbered by the
    candidate launches.  Apply the winner and return its knob dict; subsequent
    ``nvfp4_mega_moe`` calls on ``symm_buffer`` reuse the winning compile.
    COLLECTIVE -- see :func:`autotune_knobs`.
    """
    from .nvfp4 import COMBINE_FORMAT_NAMES, nvfp4_mega_moe

    def launch() -> None:
        # sync=True: the tune loop times launches with perf_counter, so the
        # call must block until the kernel (and output copy) complete.
        nvfp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    cfg = symm_buffer._frontend.config
    if candidates is None:
        # Session-aware default sweep: prune ikr when the config can't run it
        # and quantized-combine-invalid combos up front.
        candidates = nvfp4_candidates(
            combine_format=COMBINE_FORMAT_NAMES[cfg.combine_dtype],
            allow_in_kernel_fc2_reduce=cfg.apply_topk_in_fc1,
        )

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            record_knobs(
                winner,
                dtype="nvfp4",
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                combine_dtype=cfg.combine_dtype,
                p50_us=p50_s * 1e6,
                source="autotune",
            )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="nvfp4_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
    )


def autotune_mxfp8_mega_moe(
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
    """MXFP8 twin of :func:`autotune_nvfp4_mega_moe` (COLLECTIVE)."""
    from .mxfp8 import mxfp8_mega_moe

    def launch() -> None:
        # sync=True: the tune loop times launches with perf_counter, so the
        # call must block until the kernel (and output copy) complete.
        mxfp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    cfg = symm_buffer._frontend.config
    if candidates is None:
        candidates = mxfp8_candidates(
            in_kernel_fc2_reduce=cfg.in_kernel_fc2_reduce,
        )

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            record_knobs(
                winner,
                dtype=cfg.kind,
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                p50_us=p50_s * 1e6,
                source="autotune",
            )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="mxfp8_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
    )


def autotune_bf16_mega_moe(
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
    """Autotune the BF16 MegaMoE session on its supported geometry.

    The initial candidate list has exactly one fixed-geometry configuration.
    It still uses the collective autotune path so later supported geometries
    can be introduced without changing runtime behavior.
    """
    from .bf16 import bf16_mega_moe

    def launch() -> None:
        bf16_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        bf16_candidates() if candidates is None else candidates,
        label="bf16_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )


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
    from .w4a16 import w4a16_mega_moe

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
            from .knob_cache import record_knobs

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


__all__ = [
    "autotune_knobs",
    "autotune_bf16_mega_moe",
    "autotune_mxfp8_mega_moe",
    "autotune_nvfp4_mega_moe",
    "autotune_w4a16_mega_moe",
    "bf16_candidates",
    "mxfp8_candidates",
    "nvfp4_candidates",
    "w4a16_candidates",
]
