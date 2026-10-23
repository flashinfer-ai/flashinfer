# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tactic sweep for native MXFP8 x MXFP4 MegaMoE.

Selection and GPU timing use FlashInfer's general autotuner, with maximum
rank latency as the score. Winners are also exported to the legacy knob
cache for existing serving configurations (without a wall-clock p50).

Invoked through the :mod:`flashinfer.moe_ep.tune` CLI shim (``--dtype
mxfp8_mxfp4``). The default sweep times :func:`.config.candidate_tactics` for
the shape (128 pair-tile tactics at 256-aligned shapes, both weight L2
policies); ``--sweep schedule``
pins the rest of the default (or ``--base-knobs``) and sweeps
``load_balance_mode`` x ``group_hint``. Winners land in the knob cache under
dtype ``mxfp8_mxfp4``, keyed on ``--max-tokens``; ``knobs=None`` workspaces
resolve through it.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from ...tuning import finish_sweep, run_tuning as _run_tuning, schedule_candidates
from .config import candidate_tactics, default_tactic, validate_tactic


def _candidates(args, max_tokens: int, rank: int) -> list[dict]:
    if args.sweep != "schedule":
        return [
            dict(tactic)
            for tactic in candidate_tactics(
                hidden=args.hidden, intermediate=args.intermediate
            )
        ]
    base: dict[str, Any] = dict(
        default_tactic(max_tokens, hidden=args.hidden, intermediate=args.intermediate)
    )
    if args.base_knobs:
        pinned = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in json.loads(args.base_knobs).items()
        }
        validate_tactic(pinned)
        base.update(pinned)
    if rank == 0:
        print(f"[moe_ep-tune] schedule sweep base: {base}", flush=True)
    candidates = []
    for candidate in schedule_candidates(base):
        try:
            validate_tactic(candidate, partial=False)
        except ValueError:
            continue
        candidates.append(candidate)
    return candidates


def tune_one(args, rank: int, world_size: int, max_tokens: int) -> dict:
    """Run one max-token bucket; invoked by the shared sweep loop."""

    from ......cute_dsl.megamoe.mxfp8_mxfp4.integration import (
        autotune_mxfp8_mxfp4_mega_moe,
        create_dummy_mxfp8_mxfp4_inputs,
    )

    live_tokens = args.live_tokens if args.live_tokens is not None else max_tokens
    if live_tokens > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")
    candidates = _candidates(args, max_tokens, rank)
    workspace: Any = None
    try:
        output, fc1, fc2, workspace = create_dummy_mxfp8_mxfp4_inputs(
            rank,
            world_size,
            args.num_experts,
            max_tokens,
            live_tokens,
            args.topk,
            args.hidden,
            args.intermediate,
            gate_up_clamp=args.gate_up_clamp,
            seed=args.seed,
            capacity_tactics=candidates,
        )
        return finish_sweep(
            args,
            rank,
            max_tokens,
            live_tokens,
            workspace,
            output,
            fc1,
            fc2,
            candidates,
            functools.partial(
                autotune_mxfp8_mxfp4_mega_moe, gate_up_clamp=args.gate_up_clamp
            ),
        )
    finally:
        if workspace is not None:
            workspace.destroy()


def run_tuning(args) -> int:
    return _run_tuning(args, tune_one)


__all__ = ["run_tuning", "tune_one"]
