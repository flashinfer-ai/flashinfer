import argparse
import sys

import torch

import flashinfer


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


def _cuda_event_time_us(prof: torch.profiler.profile) -> float:
    total_us = 0.0
    for event in prof.events():
        if "CUDA" not in str(getattr(event, "device_type", "")):
            continue
        total_us += float(
            getattr(event, "cuda_time_total", None)
            or getattr(event, "device_time_total", 0.0)
        )
    if total_us > 0.0:
        return total_us

    return sum(
        float(getattr(event, "self_cuda_time_total", 0.0))
        for event in prof.key_averages()
    )


def _bench(call, rep: int, warmup: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(rep):
            call()
    torch.cuda.synchronize()
    return _cuda_event_time_us(prof) / rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rep", type=int, default=100, help="profiled iterations per measurement"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="warmup iterations before profiling"
    )
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
        print("flashinfer.mhc_post currently supports hc=4 only", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    header = f"{'N':>5} {'H':>5} {'HC':>3} {'FlashInfer us':>14}"
    print(header)
    print("-" * len(header))
    for hidden_size in args.hidden_sizes:
        for tokens in args.sequence_lengths:
            torch.manual_seed(0)
            inputs = _make_inputs(tokens, hidden_size, args.hc, device)

            def call() -> None:
                flashinfer.mhc_post(
                    inputs["x"],
                    inputs["residual"],
                    inputs["post_layer_mix"],
                    inputs["comb_res_mix"],
                )

            latency_us = _bench(call, rep=args.rep, warmup=args.warmup)
            print(f"{tokens:5d} {hidden_size:5d} {args.hc:3d} {latency_us:14.3f}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
