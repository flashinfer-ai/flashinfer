import argparse
import sys

import numpy as np
import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


DEFAULT_SEQUENCE_LENGTHS = (1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_HIDDEN_SIZES = (4096, 7168)
DEFAULT_HC = 4


def _make_inputs(
    tokens: int,
    hidden_size: int,
    hc: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = torch.randn((tokens, hidden_size), dtype=torch.bfloat16, device=device)
    residual = torch.randn(
        (tokens, hc, hidden_size), dtype=torch.bfloat16, device=device
    )
    post_layer_mix = torch.randn((tokens, hc, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((tokens, hc, hc), dtype=torch.float32, device=device)
    return {
        "x": x.contiguous(),
        "residual": residual.contiguous(),
        "post_layer_mix": post_layer_mix.contiguous(),
        "comb_res_mix": comb_res_mix.contiguous(),
    }


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
    parser.add_argument("--hc", type=int, default=DEFAULT_HC)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)
    if args.hc != 4:
        print("flashinfer.mhc.mhc_post currently supports hc=4 only", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    header = f"{'N':>5} {'H':>5} {'HC':>3} {'median us':>12} {'mean us':>12}"
    print(header)
    print("-" * len(header))
    for hidden_size in args.hidden_sizes:
        for tokens in args.sequence_lengths:
            torch.manual_seed(0)
            inputs = _make_inputs(tokens, hidden_size, args.hc, device)

            def call(inputs: dict[str, torch.Tensor] = inputs) -> None:
                flashinfer.mhc.mhc_post(
                    inputs["x"],
                    inputs["residual"],
                    inputs["post_layer_mix"],
                    inputs["comb_res_mix"],
                )

            measurements = bench_gpu_time(call)
            median_us = np.median(measurements) * 1e3
            mean_us = np.mean(measurements) * 1e3
            print(
                f"{tokens:5d} {hidden_size:5d} {args.hc:3d} "
                f"{median_us:12.3f} {mean_us:12.3f}"
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
