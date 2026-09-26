# SPDX-License-Identifier: Apache-2.0
"""Single-GPU quant+pack A/B. No transport or attention timing is implied.

python benchmarks/comm/bench_ulysses_fp8_pack.py --rows 18944 --heads 56 --world 2
"""

import argparse
import json
import statistics

import torch

from flashinfer.comm.ulysses_experimental import pack_ulysses_qkv_fp8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--heads", type=int, default=56)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument(
        "--world",
        type=int,
        default=2,
        help="simulated layout destinations, not GPU count",
    )
    p.add_argument("--iterations", type=int, default=50)
    args = p.parse_args()
    if (
        min(args.rows, args.heads, args.dim, args.world, args.iterations) <= 0
        or args.heads % args.world
    ):
        p.error("positive geometry and H divisible by world are required")
    torch.manual_seed(173)
    qkv = tuple(
        torch.randn(
            args.rows, args.heads, args.dim, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(3)
    )
    scales = tuple(t.float().abs().amax((0, 2)).clamp_min(1e-12) / 448 for t in qkv)
    out = torch.empty(
        args.world,
        args.rows,
        args.heads // args.world,
        3 * args.dim,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )

    def reference():
        chunks = []
        for t, s in zip(qkv, scales, strict=True):
            quant = (t.float() / s[None, :, None]).clamp(-448, 448).to(out.dtype)
            chunks.append(
                quant.view(
                    args.rows, args.world, args.heads // args.world, args.dim
                ).permute(1, 0, 2, 3)
            )
        return torch.cat(chunks, -1).contiguous()

    def candidate():
        return pack_ulysses_qkv_fp8(*qkv, scales, world_size=args.world, out=out)

    expected = reference()
    assert torch.equal(candidate().view(torch.uint8), expected.view(torch.uint8))
    for _ in range(10):
        reference()
        candidate()
    torch.cuda.synchronize()
    samples = {"reference": [], "fused": []}
    for i in range(args.iterations):
        cases = (("reference", reference), ("fused", candidate))
        for name, fn in cases if i % 2 == 0 else reversed(cases):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            del result
    med = {name: statistics.median(values) for name, values in samples.items()}
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "gpu_count": 1,
                "torch": torch.__version__,
                "shape": vars(args),
                "scope": "quant+pack only; precomputed scales; reference includes Torch temporaries",
                "median_ms": med,
                "speedup": med["reference"] / med["fused"],
                "bit_exact": True,
                "samples_ms": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
