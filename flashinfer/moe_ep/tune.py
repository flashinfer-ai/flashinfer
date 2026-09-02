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

Hopper Humming MXFP4 fused (the scale ABI is selected automatically)::

    torchrun --nproc_per_node=4 -m flashinfer.moe_ep.tune \\
        --dtype sm90_mxfp4 --execution-mode fused \\
        --hidden 7168 --intermediate 3072 --num-experts 384 --topk 6 \\
        --max-tokens 8 32 64 128 256 512 1024 2048

Single-rank (no torchrun)::

    MEGA_NO_DIST=1 python -m flashinfer.moe_ep.tune --dtype nvfp4 ...

``--intermediate`` is the model's post-SwiGLU width (the
``*MegaMoeConfig.intermediate_size`` convention); the shim-level conversion
(NVFP4 sessions size fc1 as ``2 * intermediate``) is applied internally, so
recorded cache keys match engine-time lookups exactly.

Nondeterministic candidates (``in_kernel_fc2_reduce``) are EXCLUDED by
default; pass ``--allow-nondeterministic`` to sweep them (a recorded ikr
winner makes the engine's output accumulation order nondeterministic).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m flashinfer.moe_ep.tune",
        description="Offline cutedsl mega-MoE knob tuner (writes the knob cache).",
    )
    parser.add_argument(
        "--dtype",
        choices=(
            "nvfp4",
            "mxfp8_e4m3",
            "mxfp8_e5m2",
            "sm90_fp8_e4m3",
            "sm90_fp8_e5m2",
            "sm90_mxfp4",
        ),
        default="nvfp4",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("fused", "split"),
        default="fused",
        help="execution identity (sm90_mxfp4 only; split currently requires "
        "its dedicated session-rebuild tuner)",
    )
    parser.add_argument(
        "--routing-profile",
        choices=(
            SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
            SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        ),
        default=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        help="canonical routing workload identity (sm90_mxfp4 only; default: "
        "published exact-balanced routing)",
    )
    parser.add_argument(
        "--fp8-scale-mode",
        choices=("per_tensor", "blockwise", "mxfp4_hybrid"),
        default=None,
        help="scale ABI: per_tensor by default for existing dtypes; fixed to "
        "mxfp4_hybrid for sm90_mxfp4",
    )
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument(
        "--intermediate",
        type=int,
        required=True,
        help="model post-SwiGLU intermediate size "
        "(*MegaMoeConfig.intermediate_size convention)",
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._routing_profile_specified = any(
        value == "--routing-profile" or value.startswith("--routing-profile=")
        for value in raw_argv
    )
    if args.fp8_scale_mode is None:
        args.fp8_scale_mode = (
            "mxfp4_hybrid" if args.dtype == "sm90_mxfp4" else "per_tensor"
        )
    if args.gate_up_clamp is None and args.dtype == "sm90_mxfp4":
        args.gate_up_clamp = 10.0
    return args


def _argument_error(args: argparse.Namespace) -> Optional[str]:
    """Return a fail-closed CLI error without changing existing FP8 defaults."""
    is_mxfp4 = args.dtype == "sm90_mxfp4"
    if args.combine_dtype != "bf16" and args.dtype != "nvfp4":
        return "--combine-dtype is only wired for --dtype nvfp4"

    if not is_mxfp4:
        if args._routing_profile_specified:
            return "--routing-profile is only wired for --dtype sm90_mxfp4"
        if args.execution_mode != "fused":
            return "--execution-mode is only wired for --dtype sm90_mxfp4"
        if args.fp8_scale_mode == "mxfp4_hybrid":
            return "--fp8-scale-mode mxfp4_hybrid requires --dtype sm90_mxfp4"
        return None

    if args.fp8_scale_mode != "mxfp4_hybrid":
        return "--dtype sm90_mxfp4 fixes --fp8-scale-mode mxfp4_hybrid"
    if args.combine_dtype != "bf16":
        return "--dtype sm90_mxfp4 fixes --combine-dtype bf16"
    try:
        normalize_sm90_routing_profile(args.routing_profile)
    except ValueError as exc:
        return str(exc)
    if args.seed != 0:
        return (
            "--dtype sm90_mxfp4 requires --seed 0 so weights, activations, "
            "routing IDs, and manifest-derived tactics keep their certified identity"
        )
    if args.allow_nondeterministic:
        return (
            "--allow-nondeterministic is not applicable to sm90_mxfp4; "
            "the manifest candidate union fixes in-kernel reduce off"
        )
    if args.sweep != "default" or args.base_knobs is not None:
        return (
            "sm90_mxfp4 accepts only --sweep default with no --base-knobs; "
            "candidates come exclusively from hopper_mxfp4_candidates()"
        )
    if args.skew is not None:
        return (
            "--skew is not applicable to sm90_mxfp4 offline tuning; its "
            "canonical input recipe fixes deterministic balanced routing"
        )
    if args.max_candidates is not None and args.max_candidates <= 0:
        return "--max-candidates must be positive"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    error = _argument_error(args)
    if error is not None:
        print(error, file=sys.stderr)
        return 2

    if args.dtype == "nvfp4":
        from .backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.tuner import (
            run_tuning as nvfp4_run_tuning,
        )

        return nvfp4_run_tuning(args)
    elif args.dtype.startswith("sm90_fp8"):
        from .backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.tuner import (
            run_tuning as sm90_fp8_run_tuning,
        )

        return sm90_fp8_run_tuning(args)
    elif args.dtype == "sm90_mxfp4":
        from .backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.tuner import (
            run_tuning as sm90_mxfp4_run_tuning,
        )

        return sm90_mxfp4_run_tuning(args)
    else:
        from .backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.tuner import (
            run_tuning as mxfp8_run_tuning,
        )

        return mxfp8_run_tuning(args)


if __name__ == "__main__":
    sys.exit(main())
