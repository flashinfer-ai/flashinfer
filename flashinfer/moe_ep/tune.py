"""Offline tuning for CuTe DSL MegaMoE backends.

``--arch auto`` selects SM107 on Rubin and SM100 otherwise. The
``sm90_fp8_*`` and ``sm90_mxfp4`` dtypes select the Hopper tuner. BF16 and
mixed BF16/MXFP8 are supported by the SM100 tuner only. MXFP4/MXFP8 is
wired for SM107.

Match the deployment's GPU, EP world size, geometry, and token capacity::

    torchrun --nproc_per_node=4 -m flashinfer.moe_ep.tune \
        --dtype nvfp4 --hidden 7168 --intermediate 2048 \
        --num-experts 256 --topk 8 --max-tokens 8 512 2048

Hopper Humming MXFP4 fused (the scale ABI is selected automatically)::

    torchrun --nproc_per_node=4 -m flashinfer.moe_ep.tune \\
        --dtype sm90_mxfp4 \\
        --hidden 7168 --intermediate 3072 --num-experts 384 --topk 6 \\
        --max-tokens 8 32 64 128 256 512 1024 2048

``--intermediate`` is the width after activation. Use ``MEGA_NO_DIST=1`` for
single-rank tuning. Atomic reduction candidates require
``--allow-nondeterministic`` and an engine configuration that enables IKR.
"""

from __future__ import annotations

import argparse
import importlib
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
            "mxfp4_mxfp8",
            "sm90_fp8_e4m3",
            "sm90_fp8_e5m2",
            "sm90_mxfp4",
            "bf16",
            "bf16_mxfp8_e4m3",
            "bf16_mxfp8_e5m2",
        ),
        default="nvfp4",
    )
    parser.add_argument(
        "--routing-profile",
        choices=(
            SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
            SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        ),
        default=None,
        help="canonical routing workload identity (sm90_mxfp4 only; default: "
        "the PR4688 block-permutation workload)",
    )
    parser.add_argument(
        "--fp8-scale-mode",
        choices=("per_tensor", "blockwise", "mxfp4_hybrid"),
        default=None,
        help="scale ABI: per_tensor by default for existing dtypes; fixed to "
        "mxfp4_hybrid for sm90_mxfp4",
    )
    parser.add_argument(
        "--arch",
        choices=("auto", "sm90", "sm100", "sm107"),
        default="auto",
        help="backend family; auto selects sm107 on Rubin, sm100 otherwise; "
        "sm90_fp8_* and sm90_mxfp4 dtypes select sm90",
    )
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument(
        "--intermediate",
        type=int,
        required=True,
        help="model width after activation "
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
        help="cross-rank FC2 return format (SM100 NVFP4 or SM107)",
    )
    parser.add_argument(
        "--kernel-variant",
        choices=("inference", "genphase"),
        default="inference",
        help="SM107 kernel composition; GenPhase requires at most 1024 tokens/rank",
    )
    parser.add_argument(
        "--gate-up-clamp",
        type=float,
        default=None,
        help="numeric gate/up clamp (default: 10 for sm90_mxfp4; disabled "
        "for existing dtypes)",
    )
    parser.add_argument(
        "--activation",
        choices=("swiglu", "situ"),
        default="swiglu",
        help="SM107 activation; SiTU requires both beta arguments",
    )
    parser.add_argument("--situ-beta", type=float)
    parser.add_argument("--situ-linear-beta", type=float)
    parser.add_argument(
        "--input-norm-const",
        type=float,
        default=1.0,
        help="SM107 NVFP4 input quantization normalization",
    )
    parser.add_argument(
        "--fc1-alpha",
        type=float,
        default=1.0,
        help="SM107 NVFP4 FC1 accumulator multiplier for every expert",
    )
    parser.add_argument(
        "--fc2-alpha",
        type=float,
        default=1.0,
        help="SM107 NVFP4 FC2 accumulator multiplier for every expert",
    )
    parser.add_argument(
        "--fc1-norm-const",
        type=float,
        default=1.0,
        help="SM107 NVFP4 intermediate quantization normalization",
    )
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
        "file (FLASHINFER_MOE_EP_KNOB_CACHE). For sm90_mxfp4, an explicit "
        "value must equal every --max-tokens bucket.",
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
    args._routing_profile_specified = args.routing_profile is not None
    if args.routing_profile is None:
        args.routing_profile = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    if args.fp8_scale_mode is None:
        args.fp8_scale_mode = (
            "mxfp4_hybrid" if args.dtype == "sm90_mxfp4" else "per_tensor"
        )
    if args.gate_up_clamp is None and args.dtype == "sm90_mxfp4":
        args.gate_up_clamp = 10.0
    return args


def _argument_error(args: argparse.Namespace) -> Optional[str]:
    """Return an error for unsupported dtype-specific tuning options."""
    is_mxfp4 = args.dtype == "sm90_mxfp4"

    if not is_mxfp4:
        if args._routing_profile_specified:
            return "--routing-profile is only wired for --dtype sm90_mxfp4"
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
            "--dtype sm90_mxfp4 requires --seed 0 to match the tuning "
            "workload used by the cache"
        )
    if args.live_tokens is not None and any(
        args.live_tokens != max_tokens for max_tokens in args.max_tokens
    ):
        return (
            "--dtype sm90_mxfp4 requires --live-tokens to equal every "
            "--max-tokens value because live tokens are not part of the "
            "persistent cache key; omit --live-tokens for per-bucket tuning"
        )
    if args.allow_nondeterministic:
        return (
            "--allow-nondeterministic is not applicable to sm90_mxfp4; "
            "MXFP4 candidates disable in-kernel FC2 reduction"
        )
    if args.sweep != "default" or args.base_knobs is not None:
        return (
            "sm90_mxfp4 accepts only --sweep default with no --base-knobs; "
            "candidates come from hopper_mxfp4_candidates()"
        )
    if args.skew is not None:
        return (
            "--skew is not applicable to sm90_mxfp4 offline tuning; its "
            "canonical input recipe fixes deterministic balanced routing"
        )
    if args.max_candidates is not None and args.max_candidates <= 0:
        return "--max-candidates must be positive"
    return None


def _resolve_arch(arch: str) -> str:
    if arch != "auto":
        return arch
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7):
        return "sm107"
    return "sm100"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    error = _argument_error(args)
    if error is not None:
        print(error, file=sys.stderr)
        return 2

    if args.dtype == "sm90_mxfp4":
        if args.arch not in ("auto", "sm90"):
            print("sm90_mxfp4 requires --arch auto or sm90", file=sys.stderr)
            return 2
        family = "sm90"
        backend = "fp8_mxfp4_bf16_pull_cutedsl"
    elif args.dtype.startswith("sm90_fp8"):
        if args.arch not in ("auto", "sm90"):
            print("sm90_fp8_* dtypes require --arch auto or sm90", file=sys.stderr)
            return 2
        family = "sm90"
        backend = "fp8_fp8_bf16_pull_cutedsl"
    else:
        family = _resolve_arch(args.arch)
        if family == "sm90" or (family == "sm107" and args.dtype.startswith("bf16")):
            print(f"--dtype {args.dtype} is unsupported on {family}", file=sys.stderr)
            return 2
        if args.dtype == "mxfp4_mxfp8":
            if family != "sm107":
                print("--dtype mxfp4_mxfp8 requires --arch sm107", file=sys.stderr)
                return 2
            backend = "mxfp8_mxfp4_bf16_cutedsl"
        elif args.dtype == "nvfp4":
            backend = "nvfp4_nvfp4_bf16_cutedsl"
        elif args.dtype == "bf16":
            backend = "bf16_bf16_bf16_cutedsl"
        elif args.dtype.startswith("bf16_mxfp8"):
            backend = "bf16_mxfp8_bf16_cutedsl"
        else:
            backend = "mxfp8_mxfp8_bf16_cutedsl"

    if family != "sm107" and args.combine_dtype != "bf16" and args.dtype != "nvfp4":
        print("--combine-dtype requires SM100 NVFP4 or SM107", file=sys.stderr)
        return 2
    if args.kernel_variant == "genphase" and args.combine_dtype != "bf16":
        print("GenPhase requires --combine-dtype bf16", file=sys.stderr)
        return 2

    if args.kernel_variant != "inference" and family != "sm107":
        print("--kernel-variant genphase requires --arch sm107", file=sys.stderr)
        return 2

    activation_requested = (
        args.activation != "swiglu"
        or args.situ_beta is not None
        or args.situ_linear_beta is not None
    )
    scaling_requested = any(
        getattr(args, name) != 1.0
        for name in ("input_norm_const", "fc1_alpha", "fc2_alpha", "fc1_norm_const")
    )
    if family != "sm107" and (activation_requested or scaling_requested):
        print(
            "activation and normalization options require --arch sm107", file=sys.stderr
        )
        return 2
    if scaling_requested and args.dtype != "nvfp4":
        print("normalization options require --dtype nvfp4", file=sys.stderr)
        return 2

    tuner = importlib.import_module(
        f".backends.mega.kernel.{family}.{backend}.tuner", __package__
    )
    return tuner.run_tuning(args)


if __name__ == "__main__":
    sys.exit(main())
