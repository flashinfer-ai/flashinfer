# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Online (warmup-time) knob autotuning for the CuTeDSL MegaMoE frontends.

Times a curated candidate knob set on the live problem and applies the winner
to the session's frontend, replacing the static two-profile heuristic in
:mod:`.tuner` with a measured choice.  The candidate space mirrors the
restricted sweep used with the kernel team's tester
(``tester.tester --sweep --use_knob ...``), minus ``in_kernel_fc2_reduce``:
that knob changes the output placement (sym-heap REDG target) and determinism,
so it stays owned by the config / caller.

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

from .tuner import is_valid

# Shared base of the sweep restriction (values that won every profile so far).
_SWEEP_BASE: Dict[str, Any] = {
    "cluster_shape_mnk": (2, 1, 1),
    "group_hint": 512,
    "epi_flag_batch": (2, 4),
    "load_balance_mode": "atomic_counter",
}


def nvfp4_candidates() -> List[Dict[str, Any]]:
    """Default NVFP4 candidate knob dicts (12: tile x flag_batch x token-back)."""
    out: List[Dict[str, Any]] = []
    for tile in ((256, 128, 256), (256, 256, 256)):
        for flag_batch in (4, 8):
            for token_back in (
                "epi_warps",
                "standalone_warps",
                "reuse_dispatch_warps",
            ):
                knobs = dict(
                    _SWEEP_BASE,
                    mma_tiler_mnk=tile,
                    flag_batch=flag_batch,
                    token_back_mode=token_back,
                )
                if is_valid(knobs):
                    out.append(knobs)
    return out


def mxfp8_candidates() -> List[Dict[str, Any]]:
    """Default MXFP8 candidate knob dicts (4: flag_batch x token-back).

    The MXFP8 kernel's tile is fixed at ``(256, 256)`` so no tile axis, and its
    config exposes token-back as the ``token_back_by_dispatch`` bool, so the
    two dispatch-warp modes collapse to one candidate.
    """
    out: List[Dict[str, Any]] = []
    for flag_batch in (4, 8):
        for token_back in ("epi_warps", "reuse_dispatch_warps"):
            knobs = dict(
                _SWEEP_BASE,
                flag_batch=flag_batch,
                token_back_mode=token_back,
            )
            if is_valid(knobs):
                out.append(knobs)
    return out


def autotune_knobs(
    frontend: Any,
    launch: Callable[[], None],
    candidates: List[Dict[str, Any]],
    *,
    label: str,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Time each candidate on the live problem and apply the winner.

    ``frontend`` is a NVFP4/MXFP8 mega frontend (must have ``apply_knobs``);
    ``launch`` is a zero-arg closure that runs one synchronized forward with
    the caller's real staged inputs (e.g. a ``nvfp4_mega_moe(...)`` call).

    COLLECTIVE: every EP rank must call this in the same iteration with the
    same ``candidates`` (order included).  Returns the winning knob dict.
    """
    if not candidates:
        raise ValueError("autotune_knobs needs a non-empty candidate list.")

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
    if rank == 0:
        ranked = sorted(zip(t.tolist(), candidates), key=lambda kv: kv[0])
        summary = "\n".join(
            f"    {us * 1e6:10.1f} us  {knobs}" for us, knobs in ranked
        )
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
    from .nvfp4 import nvfp4_mega_moe

    def launch() -> None:
        nvfp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
        )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        nvfp4_candidates() if candidates is None else candidates,
        label="nvfp4_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
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
        mxfp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
        )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        mxfp8_candidates() if candidates is None else candidates,
        label="mxfp8_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )


__all__ = [
    "autotune_knobs",
    "autotune_mxfp8_mega_moe",
    "autotune_nvfp4_mega_moe",
    "mxfp8_candidates",
    "nvfp4_candidates",
]
