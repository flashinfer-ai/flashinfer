"""Offline knob tuner for the cutedsl mega-MoE path (CLI shim).

This module is only the command-line frontend: it parses arguments and
dispatches to the tuner that lives NEXT TO the backend being tuned
(``backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/tuner.py`` for
``--dtype nvfp4``, ``.../mxfp8_mxfp8_bf16_cutedsl/tuner.py`` for the mxfp8
kinds).  Shared sweep machinery lives in ``backends/mega/kernel/tuning.py``.

The sweep runs the collective autotune OUTSIDE any serving engine and
persists the winners in the knob cache (see
``kernel_src/cutedsl_megamoe/shim/knob_cache.py``). After tuning, an engine
that constructs the mega layer with ``knobs=None`` (the default) resolves the
recorded winner with a pure dict lookup — no compiles, no collectives, no
timing on the hot path.

Run with the SAME EP world size, GPU model, and geometry as production.
Multi-rank (matches a 4-GPU EP deployment)::

    torchrun --nproc_per_node=4 -m flashinfer.moe_ep.tune \\
        --dtype nvfp4 --hidden 7168 --intermediate 2048 \\
        --num-experts 256 --topk 8 --max-tokens 8 512 2048

Single-rank (no torchrun)::

    MEGA_NO_DIST=1 python -m flashinfer.moe_ep.tune --dtype nvfp4 ...

``--intermediate`` is the model's post-activation width (the
``*MegaMoeConfig.intermediate_size`` convention); the shim-level conversion
(NVFP4 sessions size FC1 as ``2 * intermediate`` except native single-plane
ReLU2, which uses ``intermediate``) is applied internally, so recorded cache
keys match engine-time lookups exactly.

Nondeterministic candidates (``in_kernel_fc2_reduce``) are EXCLUDED by
default; pass ``--allow-nondeterministic`` to sweep them (a recorded ikr
winner makes the engine's output accumulation order nondeterministic).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m flashinfer.moe_ep.tune",
        description="Offline cutedsl mega-MoE knob tuner (writes the knob cache).",
    )
    parser.add_argument(
        "--dtype", choices=("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"), default="nvfp4"
    )
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument(
        "--intermediate",
        type=int,
        required=True,
        help="model post-activation intermediate size "
        "(*MegaMoeConfig.intermediate_size convention)",
    )
    parser.add_argument(
        "--activation",
        choices=("swiglu", "relu2"),
        default="swiglu",
        help="NVFP4 FC1 activation (relu2 uses a semantic I-wide W1 plane)",
    )
    parser.add_argument(
        "--relu2-kernel",
        choices=("padded", "single_plane"),
        default="padded",
        help="ReLU2 physical layout: compatibility 2*I padding or native I-wide FC1",
    )
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        required=True,
        help="buffer capacities (tokens/rank) to tune, one "
        "sweep each — use the engine's actual buffer size(s)",
    )
    parser.add_argument(
        "--combine-dtype",
        choices=("bf16", "mxfp8", "nvfp4"),
        default="bf16",
        help="cross-rank combine wire (nvfp4 dtype only)",
    )
    parser.add_argument("--gate-up-clamp", type=float, default=None)
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="also sweep in_kernel_fc2_reduce candidates",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="truncate the candidate list (smoke testing)",
    )
    parser.add_argument(
        "--live-tokens",
        type=int,
        default=None,
        help="live token count to stage and time (default: the bucket size). "
        "Use a decode-like count (e.g. 256) to tune for decode steps while "
        "keeping the engine's buffer bucket; the cache entry is still keyed "
        "on --max-tokens, so write decode-tuned winners to a separate cache "
        "file (FLASHINFER_MOE_EP_KNOB_CACHE).",
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=None,
        help="target per-launch expert-load skew (max-load/mean-load) for the "
        "tuning routing, e.g. 18 for the DSV4-measured mean. Default keeps "
        "the near-uniform random routing — which CANNOT discriminate "
        "skew-sensitive knobs (load_balance_mode, scheduling); pass the "
        "measured production ratio (FI_MOE_EP_LOAD_STATS cold run).",
    )
    parser.add_argument(
        "--sweep",
        choices=("default", "schedule"),
        default="default",
        help="'default' sweeps tile/flag_batch/token-back(/ikr); 'schedule' "
        "pins those from --base-knobs (or the current cache winner) and "
        "sweeps load_balance_mode x group_hint — the skew-sensitive axes.",
    )
    parser.add_argument(
        "--base-knobs",
        type=str,
        default=None,
        help="JSON knob dict used as the base for --sweep schedule "
        "(default: resolve the current cache/heuristic winner for this key)",
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.combine_dtype != "bf16" and args.dtype != "nvfp4":
        print("--combine-dtype is only wired for --dtype nvfp4", file=sys.stderr)
        return 2
    if args.activation != "swiglu" and args.dtype != "nvfp4":
        print("--activation relu2 is only wired for --dtype nvfp4", file=sys.stderr)
        return 2
    if args.activation == "relu2" and args.gate_up_clamp is not None:
        print("--gate-up-clamp is not valid with --activation relu2", file=sys.stderr)
        return 2
    if args.relu2_kernel == "single_plane" and args.activation != "relu2":
        print(
            "--relu2-kernel single_plane requires --activation relu2",
            file=sys.stderr,
        )
        return 2

    if args.dtype == "nvfp4":
        from .backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.tuner import (
            run_tuning,
        )
    else:
        from .backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.tuner import (
            run_tuning,
        )
    return run_tuning(args)


if __name__ == "__main__":
    sys.exit(main())
