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
from typing import Any, Callable, Dict, List, Optional

import torch

from .tuner import (
    CORRECTNESS_KNOBS,
    default_knobs,
    describe_invalid_knobs,
    is_valid,
    is_valid_bf16,
    is_valid_bf16_for_config,
    is_valid_bf16_mxfp8,
    is_valid_bf16_mxfp8_for_config,
)

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


def bf16_candidates(
    *,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: str = "epi_warps",
) -> List[Dict[str, Any]]:
    """Return supported BF16 candidates for a session's output contract.

    This is intentionally a one-entry autotune surface. Keeping the same
    collective autotune lifecycle as the other Mega kernels means additional
    validated geometries can be added without changing the public API.
    """
    knobs = default_knobs(0, dtype="bf16")
    knobs["in_kernel_fc2_reduce"] = in_kernel_fc2_reduce
    knobs["token_back_mode"] = token_back_mode
    return [knobs] if is_valid_bf16(knobs) else []


def bf16_mxfp8_candidates(
    *,
    in_kernel_fc2_reduce: bool = False,
) -> List[Dict[str, Any]]:
    """Default BF16×MXFP8 candidate knob dicts (impl tuple × flag_batch × token-back).

    Twelve candidates: three legal implementation tuples (N128/tmem,
    N256/smem, N256/tmem-overlap) × ``flag_batch`` {1, 4} ×
    ``token_back_mode`` {``epi_warps``, ``reuse_dispatch_warps``}.
    ``standalone_warps`` is unsupported.  ikr stays config-owned (the symm
    buffer's combine plane is shaped for it), so pass the session's value to
    stamp it on every candidate; unlike pure MXFP8 the mixed kernel runs ikr
    with either token-back mode, so it prunes nothing.
    """
    out: List[Dict[str, Any]] = []
    impl_specs = (
        {
            "mma_tiler_mnk": (256, 128, 128),
            "transform_buffer": "tmem",
            "accumulator_overlap": False,
            "transform_k_tile": 128,
        },
        {
            "mma_tiler_mnk": (256, 256, 128),
            "transform_buffer": "smem",
            "accumulator_overlap": False,
            "transform_k_tile": 128,
        },
        {
            "mma_tiler_mnk": (256, 256, 128),
            "transform_buffer": "tmem",
            "accumulator_overlap": True,
            "transform_k_tile": 64,
        },
    )
    base: Dict[str, Any] = {
        "cluster_shape_mnk": (2, 1, 1),
        "use_2cta_instrs": True,
        "group_hint": 512,
        "epi_flag_batch": (2, 4),
        "load_balance_mode": "static",
    }
    for impl in impl_specs:
        for flag_batch in (1, 4):
            for token_back in ("epi_warps", "reuse_dispatch_warps"):
                knobs = dict(
                    base,
                    **impl,
                    flag_batch=flag_batch,
                    token_back_mode=token_back,
                    in_kernel_fc2_reduce=in_kernel_fc2_reduce,
                )
                if is_valid_bf16_mxfp8(knobs):
                    out.append(knobs)
    return out


def _session_candidates(
    candidates: List[Dict[str, Any]],
    config: Any,
    predicate: Callable[[Any, Dict[str, Any]], bool],
    *,
    what: str,
) -> List[Dict[str, Any]]:
    """Drop candidates a session cannot run (e.g. they would flip its ikr).

    Every drop is reported with the offending knob, so a caller never loses a
    pinned value silently.  The filter is a pure function of the config and the
    candidate list, both identical on every rank, so the surviving order stays
    in collective lockstep (see :func:`autotune_knobs`).
    """
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for knobs in candidates:
        if predicate(config, knobs):
            kept.append(knobs)
        else:
            dropped.append(
                f"    {knobs}\n      -> {describe_invalid_knobs(config, knobs, predicate)}"
            )
    if not kept:
        detail = ("; all were rejected:\n" + "\n".join(dropped)) if dropped else ""
        raise ValueError(
            f"no valid {what} autotune candidates for this session{detail}"
        )
    if dropped:
        warnings.warn(
            f"[cutedsl-autotune] {what}: dropped {len(dropped)}/{len(candidates)} "
            f"candidate(s) this session cannot run:\n" + "\n".join(dropped),
            RuntimeWarning,
            stacklevel=3,
        )
    return kept


def _session_correctness_knobs(config: Any) -> Dict[str, Any]:
    """Snapshot the output-affecting knobs a config currently carries."""
    # ``token_back_by_dispatch`` is the MXFP8 config's spelling of token-back.
    return {
        name: getattr(config, name)
        for name in (*CORRECTNESS_KNOBS, "token_back_by_dispatch")
        if hasattr(config, name)
    }


def _warn_on_overridden_session_knobs(
    config: Any, before: Dict[str, Any], *, label: str
) -> None:
    """Warn when the winner replaced an output-affecting session setting.

    The sweep owns the correctness knobs it enumerates, so a winner may
    legitimately replace what the caller configured; say so rather than let an
    explicitly requested value disappear into the tuned config.
    """
    overridden = [
        f"{name}: {was!r} -> {getattr(config, name)!r}"
        for name, was in before.items()
        if getattr(config, name) != was
    ]
    if overridden:
        warnings.warn(
            f"[cutedsl-autotune] {label}: the measured winner replaced "
            f"session-configured output-affecting knobs ({', '.join(overridden)}); "
            f"pin an explicit knobs dict instead of 'auto' to keep them.",
            RuntimeWarning,
            stacklevel=3,
        )


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
    if not candidates:
        raise ValueError("autotune_knobs needs a non-empty candidate list.")

    session_knobs = _session_correctness_knobs(frontend.config)

    from .comm import ensure_not_capturing

    # The sweep barriers, compiles per candidate, wall-clock times with
    # internal syncs, and all_reduces -- none of it can run mid-capture.
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
                launch()
            _barrier()
            iters: List[float] = []
            for _ in range(timed_iters):  # launch() syncs internally
                t0 = time.perf_counter()
                launch()
                iters.append(time.perf_counter() - t0)
            scores.append(statistics.median(iters))
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
    _warn_on_overridden_session_knobs(frontend.config, session_knobs, label=label)
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

    cfg = symm_buffer._frontend.config
    if candidates is None:
        candidates = bf16_candidates(
            in_kernel_fc2_reduce=cfg.in_kernel_fc2_reduce,
            token_back_mode=cfg.token_back_mode,
        )
    candidates = _session_candidates(
        candidates, cfg, is_valid_bf16_for_config, what="BF16 MegaMoE"
    )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="bf16_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )


def autotune_bf16_mxfp8_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Autotune the BF16×MXFP8 MegaMoE session on staged inputs."""
    from .bf16_mxfp8 import bf16_mxfp8_mega_moe

    def launch() -> None:
        bf16_mxfp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            sync=True,
        )

    cfg = symm_buffer._frontend.config
    if candidates is None:
        candidates = bf16_mxfp8_candidates(
            in_kernel_fc2_reduce=cfg.in_kernel_fc2_reduce,
        )
    candidates = _session_candidates(
        candidates,
        cfg,
        is_valid_bf16_mxfp8_for_config,
        what="mixed BF16/MXFP8 MegaMoE",
    )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="bf16_mxfp8_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )


__all__ = [
    "autotune_knobs",
    "autotune_bf16_mega_moe",
    "autotune_bf16_mxfp8_mega_moe",
    "autotune_mxfp8_mega_moe",
    "autotune_nvfp4_mega_moe",
    "bf16_candidates",
    "bf16_mxfp8_candidates",
    "mxfp8_candidates",
    "nvfp4_candidates",
]
