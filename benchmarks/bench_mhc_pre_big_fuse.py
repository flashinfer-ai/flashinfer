import argparse
import math
import sys
from collections.abc import Callable, Sequence

import torch
from torch.profiler import ProfilerActivity, profile

from flashinfer.mhc import (
    _mhc_pre_big_fuse_impl,
    _mhc_pre_big_fuse_with_prenorm_impl,
)


DEFAULT_SEQUENCE_LENGTHS = (1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_HIDDEN_SIZES = (4096, 7168)
DEFAULT_NUM_SPLITS = (1, 16)
DEFAULT_ROTATE_L2_FACTOR = 2
DEFAULT_MAX_ROTATIONS = 4096
HC = 4
MIX = HC * (2 + HC)
BF16_BYTES = 2
FP32_BYTES = 4
RMS_EPS = 1e-6
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT_VALUE = 1.0
SINKHORN_REPEAT = 20


def _make_static_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "mhc_scale": (
            torch.randn((3,), dtype=torch.float32, device=device) * 0.1
        ).contiguous(),
        "mhc_base": (
            torch.randn((MIX,), dtype=torch.float32, device=device) * 0.1
        ).contiguous(),
    }


def _make_common_case(
    tokens: int,
    hidden_size: int,
    static_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    residual = (
        torch.randn((tokens, HC, hidden_size), dtype=torch.float32, device=device)
        * 0.01
    ).bfloat16()
    return {
        "residual": residual.contiguous(),
        "mhc_scale": static_inputs["mhc_scale"],
        "mhc_base": static_inputs["mhc_base"],
        "post_mix": torch.empty((tokens, HC), dtype=torch.float32, device=device),
        "comb_mix": torch.empty((tokens, HC, HC), dtype=torch.float32, device=device),
        "layer_input": torch.empty(
            (tokens, hidden_size), dtype=torch.bfloat16, device=device
        ),
    }


def _make_pure_case(
    tokens: int,
    hidden_size: int,
    num_splits: int,
    static_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs = _make_common_case(tokens, hidden_size, static_inputs, device)
    k = HC * hidden_size
    if num_splits == 1:
        dot_mix = torch.randn((tokens, MIX), dtype=torch.float32, device=device) * 0.01
        sqrsum = torch.rand((tokens,), dtype=torch.float32, device=device) * float(k)
    else:
        dot_mix = (
            torch.randn((num_splits, tokens, MIX), dtype=torch.float32, device=device)
            * 0.01
        )
        sqrsum = torch.rand((num_splits, tokens), dtype=torch.float32, device=device)
        sqrsum = sqrsum * (float(k) / num_splits)
    inputs["dot_mix"] = dot_mix.contiguous()
    inputs["sqrsum"] = sqrsum.contiguous()
    return inputs


def _make_prenorm_case(
    tokens: int,
    hidden_size: int,
    static_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs = _make_common_case(tokens, hidden_size, static_inputs, device)
    inputs["dot_mix"] = (
        torch.randn((tokens, MIX), dtype=torch.float32, device=device) * 0.01
    ).contiguous()
    return inputs


def _estimate_case_bytes(
    tokens: int, hidden_size: int, num_splits: int, mode: str
) -> int:
    residual_bytes = tokens * HC * hidden_size * BF16_BYTES
    output_bytes = (
        tokens * HC * FP32_BYTES
        + tokens * HC * HC * FP32_BYTES
        + tokens * hidden_size * BF16_BYTES
    )
    if mode == "pure":
        factor = num_splits if num_splits > 1 else 1
        dot_mix_bytes = factor * tokens * MIX * FP32_BYTES
        sqrsum_bytes = factor * tokens * FP32_BYTES
    else:
        dot_mix_bytes = tokens * MIX * FP32_BYTES
        sqrsum_bytes = 0
    return residual_bytes + output_bytes + dot_mix_bytes + sqrsum_bytes


def _select_rotation_count(
    *,
    tokens: int,
    hidden_size: int,
    num_splits: int,
    mode: str,
    device: torch.device,
    requested_rotations: int | None,
    rotate_l2_factor: int,
    max_rotations: int,
    cold_l2_cache: bool,
) -> int:
    if not cold_l2_cache:
        return 1
    if requested_rotations is not None:
        return max(1, requested_rotations)

    l2_size = torch.cuda.get_device_properties(device).L2_cache_size
    target_bytes = l2_size * rotate_l2_factor
    case_bytes = _estimate_case_bytes(tokens, hidden_size, num_splits, mode)
    if case_bytes >= target_bytes:
        return 1

    rotations = max(2, math.ceil(target_bytes / case_bytes) + 1)
    if max_rotations > 0:
        rotations = min(rotations, max_rotations)
    return rotations


def _make_pure_cases(
    tokens: int,
    hidden_size: int,
    num_splits: int,
    rotations: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    static_inputs = _make_static_inputs(device)
    return [
        _make_pure_case(tokens, hidden_size, num_splits, static_inputs, device)
        for _ in range(rotations)
    ]


def _make_prenorm_cases(
    tokens: int,
    hidden_size: int,
    rotations: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    static_inputs = _make_static_inputs(device)
    return [
        _make_prenorm_case(tokens, hidden_size, static_inputs, device)
        for _ in range(rotations)
    ]


def _call_pure(
    inputs: dict[str, torch.Tensor],
    hidden_size: int,
    num_splits: int,
    block_size: int,
) -> None:
    _mhc_pre_big_fuse_impl(
        inputs["post_mix"],
        inputs["comb_mix"],
        inputs["layer_input"],
        inputs["dot_mix"],
        inputs["sqrsum"],
        inputs["residual"],
        inputs["mhc_scale"],
        inputs["mhc_base"],
        HC * hidden_size,
        RMS_EPS,
        MHC_PRE_EPS,
        MHC_SINKHORN_EPS,
        MHC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        num_splits,
        block_size,
    )


def _call_prenorm(inputs: dict[str, torch.Tensor], block_size: int) -> None:
    _mhc_pre_big_fuse_with_prenorm_impl(
        inputs["post_mix"],
        inputs["comb_mix"],
        inputs["layer_input"],
        inputs["dot_mix"],
        inputs["residual"],
        inputs["mhc_scale"],
        inputs["mhc_base"],
        RMS_EPS,
        MHC_PRE_EPS,
        MHC_SINKHORN_EPS,
        MHC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
        block_size,
    )


def _profile_rotate(
    cases: Sequence[dict[str, torch.Tensor]],
    call: Callable[[dict[str, torch.Tensor]], None],
    warmup_iters: int,
    profile_iters: int,
) -> float:
    # Intentional: this cold-L2 benchmark uses torch.profiler to capture CUDA
    # self time while rotating input buffers, rather than the generic hot-path
    # bench_gpu_time helper.
    torch.cuda.synchronize()
    for i in range(warmup_iters):
        call(cases[i % len(cases)])
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(profile_iters):
            call(cases[i % len(cases)])
        torch.cuda.synchronize()

    total_us = 0.0
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0 and evt.count > 0:
            total_us += evt.self_device_time_total
    return total_us / profile_iters


def _bench_pure(
    tokens: int,
    hidden_size: int,
    num_splits: int,
    block_size: int,
    rotations: int,
    warmup_iters: int,
    profile_iters: int,
    device: torch.device,
) -> float:
    cases = _make_pure_cases(tokens, hidden_size, num_splits, rotations, device)

    def call(inputs: dict[str, torch.Tensor]) -> None:
        _call_pure(inputs, hidden_size, num_splits, block_size)

    return _profile_rotate(cases, call, warmup_iters, profile_iters)


def _bench_prenorm(
    tokens: int,
    hidden_size: int,
    block_size: int,
    rotations: int,
    warmup_iters: int,
    profile_iters: int,
    device: torch.device,
) -> float:
    cases = _make_prenorm_cases(tokens, hidden_size, rotations, device)

    def call(inputs: dict[str, torch.Tensor]) -> None:
        _call_prenorm(inputs, block_size)

    return _profile_rotate(cases, call, warmup_iters, profile_iters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="token counts to benchmark",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_SIZES,
        help="hidden sizes to benchmark",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_SPLITS,
        help="pure big-fuse split counts to benchmark",
    )
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument(
        "--rotations",
        type=int,
        default=None,
        help="Fixed input-buffer rotation count. Default auto-selects enough data to exceed L2.",
    )
    parser.add_argument(
        "--rotate-l2-factor",
        type=int,
        default=DEFAULT_ROTATE_L2_FACTOR,
        help="Auto-rotation target as a multiple of L2 cache size.",
    )
    parser.add_argument(
        "--max-rotations",
        type=int,
        default=DEFAULT_MAX_ROTATIONS,
        help="Cap auto-selected rotations. Use 0 for no cap.",
    )
    parser.add_argument(
        "--no-cold-l2-cache",
        action="store_true",
        help="Disable rotate-buffer cold-L2 mode and reuse one input buffer.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=50,
        help="Warmup iterations before profiling.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=200,
        help="Profiled iterations.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)
    for num_splits in args.num_splits:
        if num_splits not in (1, 2, 4, 8, 16):
            print("num_splits must be one of {1, 2, 4, 8, 16}", file=sys.stderr)
            sys.exit(1)
    if args.warmup_iters < 0 or args.profile_iters <= 0:
        print(
            "--warmup-iters must be >= 0 and --profile-iters must be > 0",
            file=sys.stderr,
        )
        sys.exit(1)
    if any(n <= 0 for n in args.sequence_lengths) or any(
        h <= 0 for h in args.hidden_sizes
    ):
        print("--sequence-lengths and --hidden-sizes must be > 0", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    cold_l2_cache = not args.no_cold_l2_cache
    header = (
        f"{'mode':>10} {'N':>5} {'H':>5} {'splits':>6} {'cold_l2':>7} "
        f"{'rot':>5} {'iters':>6} {'avg us':>12}"
    )
    print(header)
    print("-" * len(header))

    for hidden_size in args.hidden_sizes:
        for tokens in args.sequence_lengths:
            for num_splits in args.num_splits:
                torch.manual_seed(0)
                rotations = _select_rotation_count(
                    tokens=tokens,
                    hidden_size=hidden_size,
                    num_splits=num_splits,
                    mode="pure",
                    device=device,
                    requested_rotations=args.rotations,
                    rotate_l2_factor=args.rotate_l2_factor,
                    max_rotations=args.max_rotations,
                    cold_l2_cache=cold_l2_cache,
                )
                avg_us = _bench_pure(
                    tokens,
                    hidden_size,
                    num_splits,
                    args.block_size,
                    rotations,
                    args.warmup_iters,
                    args.profile_iters,
                    device,
                )
                print(
                    f"{'pure':>10} {tokens:5d} {hidden_size:5d} {num_splits:6d} "
                    f"{str(cold_l2_cache):>7} {rotations:5d} {args.profile_iters:6d} "
                    f"{avg_us:12.3f}"
                )
                torch.cuda.empty_cache()

            torch.manual_seed(0)
            rotations = _select_rotation_count(
                tokens=tokens,
                hidden_size=hidden_size,
                num_splits=1,
                mode="prenorm",
                device=device,
                requested_rotations=args.rotations,
                rotate_l2_factor=args.rotate_l2_factor,
                max_rotations=args.max_rotations,
                cold_l2_cache=cold_l2_cache,
            )
            avg_us = _bench_prenorm(
                tokens,
                hidden_size,
                args.block_size,
                rotations,
                args.warmup_iters,
                args.profile_iters,
                device,
            )
            print(
                f"{'prenorm':>10} {tokens:5d} {hidden_size:5d} {1:6d} "
                f"{str(cold_l2_cache):>7} {rotations:5d} {args.profile_iters:6d} "
                f"{avg_us:12.3f}"
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
