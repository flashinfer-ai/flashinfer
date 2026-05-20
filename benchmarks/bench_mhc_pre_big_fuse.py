import argparse
import sys

import numpy as np
import torch

from flashinfer.mhc import (
    _mhc_pre_big_fuse_impl,
    _mhc_pre_big_fuse_with_prenorm_impl,
)
from flashinfer.testing import bench_gpu_time


DEFAULT_SEQUENCE_LENGTHS = (1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_HIDDEN_SIZES = (4096, 7168)
DEFAULT_NUM_SPLITS = (1, 16)
HC = 4
MIX = HC * (2 + HC)
RMS_EPS = 1e-6
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT_VALUE = 1.0
SINKHORN_REPEAT = 20


def _make_common_inputs(
    tokens: int,
    hidden_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    residual = (
        torch.randn((tokens, HC, hidden_size), dtype=torch.float32, device=device)
        * 0.01
    ).bfloat16()
    mhc_scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
    mhc_base = torch.randn((MIX,), dtype=torch.float32, device=device) * 0.1
    post_mix = torch.empty((tokens, HC), dtype=torch.float32, device=device)
    comb_mix = torch.empty((tokens, HC, HC), dtype=torch.float32, device=device)
    layer_input = torch.empty(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    return {
        "residual": residual.contiguous(),
        "mhc_scale": mhc_scale.contiguous(),
        "mhc_base": mhc_base.contiguous(),
        "post_mix": post_mix,
        "comb_mix": comb_mix,
        "layer_input": layer_input,
    }


def _make_pure_inputs(
    tokens: int,
    hidden_size: int,
    num_splits: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs = _make_common_inputs(tokens, hidden_size, device)
    k = HC * hidden_size
    if num_splits == 1:
        dot_mix = torch.randn((tokens, MIX), dtype=torch.float32, device=device) * 0.01
        sqrsum = torch.rand((tokens,), dtype=torch.float32, device=device) * float(k)
    else:
        dot_mix = (
            torch.randn((num_splits, tokens, MIX), dtype=torch.float32, device=device)
            * 0.01
        )
        sqrsum = torch.rand(
            (num_splits, tokens), dtype=torch.float32, device=device
        ) * (float(k) / num_splits)
    inputs["dot_mix"] = dot_mix.contiguous()
    inputs["sqrsum"] = sqrsum.contiguous()
    return inputs


def _make_prenorm_inputs(
    tokens: int,
    hidden_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs = _make_common_inputs(tokens, hidden_size, device)
    inputs["dot_mix"] = (
        torch.randn((tokens, MIX), dtype=torch.float32, device=device) * 0.01
    ).contiguous()
    return inputs


def _bench_pure(
    tokens: int,
    hidden_size: int,
    num_splits: int,
    block_size: int,
    device: torch.device,
) -> tuple[float, float]:
    inputs = _make_pure_inputs(tokens, hidden_size, num_splits, device)
    k = HC * hidden_size

    def call() -> None:
        _mhc_pre_big_fuse_impl(
            inputs["post_mix"],
            inputs["comb_mix"],
            inputs["layer_input"],
            inputs["dot_mix"],
            inputs["sqrsum"],
            inputs["residual"],
            inputs["mhc_scale"],
            inputs["mhc_base"],
            k,
            RMS_EPS,
            MHC_PRE_EPS,
            MHC_SINKHORN_EPS,
            MHC_POST_MULT_VALUE,
            SINKHORN_REPEAT,
            num_splits,
            block_size,
        )

    measurements = bench_gpu_time(call)
    return np.median(measurements) * 1e3, np.mean(measurements) * 1e3


def _bench_prenorm(
    tokens: int,
    hidden_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[float, float]:
    inputs = _make_prenorm_inputs(tokens, hidden_size, device)

    def call() -> None:
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

    measurements = bench_gpu_time(call)
    return np.median(measurements) * 1e3, np.mean(measurements) * 1e3


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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)
    for num_splits in args.num_splits:
        if num_splits not in (1, 2, 4, 8, 16):
            print("num_splits must be one of {1, 2, 4, 8, 16}", file=sys.stderr)
            sys.exit(1)

    device = torch.device("cuda")
    header = (
        f"{'mode':>10} {'N':>5} {'H':>5} {'splits':>6} "
        f"{'median us':>12} {'mean us':>12}"
    )
    print(header)
    print("-" * len(header))

    for hidden_size in args.hidden_sizes:
        for tokens in args.sequence_lengths:
            for num_splits in args.num_splits:
                median_us, mean_us = _bench_pure(
                    tokens, hidden_size, num_splits, args.block_size, device
                )
                print(
                    f"{'pure':>10} {tokens:5d} {hidden_size:5d} {num_splits:6d} "
                    f"{median_us:12.3f} {mean_us:12.3f}"
                )
                torch.cuda.empty_cache()

            median_us, mean_us = _bench_prenorm(
                tokens, hidden_size, args.block_size, device
            )
            print(
                f"{'prenorm':>10} {tokens:5d} {hidden_size:5d} {1:6d} "
                f"{median_us:12.3f} {mean_us:12.3f}"
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
