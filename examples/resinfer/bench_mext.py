# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer MEXT vs upstream CuTe GEMM plus separate row-pair reduction.

Run from the fork: python examples/resinfer/bench_mext.py --m 32 --n 4096 --k 4096
Weight preparation is excluded. Activation quantization is included only in
measurements named quantize_and_*. These are kernel pipelines, not model E2E.
The upstream baseline rounds paired outputs to BF16 before folding; ResInfer
folds FP32 accumulators before conversion, so they need not be bit-identical.
"""

import argparse
import json
import statistics

import torch

from flashinfer import mm_fp4
from flashinfer.resinfer import mm_nvfp4_residue, nvfp4_quantize_residue
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("m", 32), ("n", 4096), ("k", 4096)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    torch.manual_seed(42)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([64.0], device="cuda")
    alpha = scale.reciprocal()
    w = torch.randint(0, 256, (args.n, args.k // 2), device="cuda", dtype=torch.uint8)
    ws = (
        torch.ones(((args.n + 127) // 128) * ((args.k + 63) // 64) * 512, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    a, sf = nvfp4_quantize_residue(x, scale)

    def folded(a, sf):
        return mm_nvfp4_residue(a, w, sf, ws, alpha)

    def separate(a, sf):
        y = mm_fp4(a, w.T, sf, ws, alpha, backend="cute-dsl")
        return y.reshape(args.m, 2, args.n).sum(dim=1)

    paths = {
        "fold": lambda: folded(a, sf),
        "upstream_separate_fold": lambda: separate(a, sf),
        "quantize_and_fold": lambda: folded(*nvfp4_quantize_residue(x, scale)),
        "quantize_and_upstream_separate_fold": lambda: separate(
            *nvfp4_quantize_residue(x, scale)
        ),
    }
    # Warm all paths; fail loudly on compilation or shape errors before timing.
    outputs = {name: fn() for name, fn in paths.items()}
    for y in outputs.values():
        assert torch.isfinite(y).all()
    raw = {
        name: bench_gpu_time(fn, dry_run_iters=10, repeat_iters=30)
        for name, fn in paths.items()
    }
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "shape": vars(args),
                "milliseconds": raw,
                "median_ms": {name: statistics.median(v) for name, v in raw.items()},
                "fold_vs_baseline_max_abs": float(
                    (
                        outputs["fold"].float()
                        - outputs["upstream_separate_fold"].float()
                    )
                    .abs()
                    .max()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
