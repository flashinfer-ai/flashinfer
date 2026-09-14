# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Paired CuTe/Triton RMSNorm NVFP4 benchmark, with a PDL-disabled control.

python benchmarks/bench_triton_rmsnorm_fp4quant.py --output results.jsonl
python benchmarks/bench_triton_rmsnorm_fp4quant.py --reverse --output repeat.jsonl

Resident inputs; explicit global scale; no allocation/copy inside the measured
preallocated path. Public-allocation and eager timing are selectable diagnostics.
Each JSON line records its timing mode; do not pool different modes.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time

import torch

import flashinfer
from flashinfer.norm import rmsnorm_fp4quant
from flashinfer.testing import bench_gpu_time

SHAPES = [
    (1, 4096),
    (2, 4096),
    (3, 4096),
    (7, 4096),
    (8, 4096),
    (24, 4096),
    (128, 4096),
    (512, 4096),
    (1, 7168),
    (4, 7168),
    (16, 7168),
    (256, 7168),
    (4096, 7168),
    (1024, 4096),
    (16384, 4096),
]


def decode(q, sf, m, k, swizzled, scale):
    raw = q.view(torch.uint8)
    levels = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=q.device,
    )
    values = torch.stack(
        (levels[(raw & 15).long()], levels[(raw >> 4).long()]), -1
    ).reshape(m, k // 16, 16)
    if swizzled:
        mt, kt = (m + 127) // 128, (k // 16 + 3) // 4
        sf = (
            sf.reshape(mt, kt, 32, 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(mt * 128, kt * 4)[:m, : k // 16]
        )
    return (values * sf.float()[..., None]).reshape(m, k) / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--timing", choices=["graph", "eager", "cupti"], default="graph"
    )
    parser.add_argument(
        "--allocation", choices=["preallocated", "public"], default="preallocated"
    )
    parser.add_argument(
        "--layout", choices=["swizzled", "row-major"], default="swizzled"
    )
    parser.add_argument("--global-scales", type=float, nargs="+", default=[1.0, 32.0])
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark targets SM120")
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(flashinfer.__file__).resolve().parent
    sources = [
        "norm/__init__.py",
        "cute_dsl/rmsnorm_fp4quant.py",
        "cute_dsl/fp4_common.py",
        "experimental/triton_rmsnorm_fp4quant/backend.py",
        "experimental/triton_rmsnorm_fp4quant/kernel.py",
    ]
    with args.output.open("x") as output:

        def emit(kind, **values):
            print(
                json.dumps(
                    dict(kind=kind, wall_time=time.time(), **values), allow_nan=False
                ),
                file=output,
                flush=True,
            )

        emit(
            "environment",
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            flashinfer=flashinfer.__version__,
            packages={
                p: importlib.metadata.version(p)
                for p in ["triton", "nvidia-cutlass-dsl", "cuda-bindings"]
            },
            sources={
                p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
            },
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            options={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            protocol="resident inputs; 5 randomized rounds; 50 calls/graph; 20 replay samples; 10 warmup iterations; CUDA events unless CUPTI explicitly selected",
        )
        shapes = [(7, 80), (128, 4096)] if args.smoke else SHAPES
        seeds = [73] if args.smoke else [73, 109]
        cases = [
            (seed, m, k, g)
            for seed in seeds
            for m, k in shapes
            for g in args.global_scales
        ]
        if args.reverse:
            cases.reverse()
        for seed, m, k, g in cases:
            torch.manual_seed(seed)
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            w = (0.8 + 0.4 * torch.rand(k, device="cuda")).bfloat16()
            scale = torch.tensor([g], device="cuda", dtype=torch.float32)
            swizzled = args.layout == "swizzled"
            reference = (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                * w.float()
            )
            methods = {}
            for name, backend, pdl in [
                ("cute_default", "cute-dsl", None),
                ("cute_no_pdl", "cute-dsl", False),
                ("triton", "triton", False),
            ]:
                kwargs = dict(
                    global_scale=scale,
                    is_sf_swizzled_layout=swizzled,
                    backend=backend,
                    enable_pdl=pdl,
                )
                q, sf = rmsnorm_fp4quant(x, w, **kwargs)
                values = decode(q, sf, m, k, swizzled, scale)
                error = (
                    (
                        (values - reference).double().square().sum()
                        / reference.double().square().sum()
                    )
                    .sqrt()
                    .item()
                )
                if not error < 0.2:
                    raise AssertionError((name, m, k, g, error))
                emit(
                    "correctness",
                    seed=seed,
                    M=m,
                    K=k,
                    global_scale=g,
                    method=name,
                    relative_rms=error,
                )
                if args.allocation == "preallocated":
                    kwargs.update(y_fp4=q, block_scale=sf)

                def call(kwargs=kwargs):
                    return rmsnorm_fp4quant(x, w, **kwargs)

                methods[name] = call
            rng = random.Random(seed + m + k)
            for round_id in range(5):
                order = list(methods)
                rng.shuffle(order)
                for name in order:
                    samples = bench_gpu_time(
                        methods[name],
                        dry_run_iters=10,
                        repeat_iters=20,
                        enable_cupti=args.timing == "cupti",
                        use_cuda_graph=args.timing != "eager",
                        num_iters_within_graph=50,
                        cold_l2_cache=False,
                    )
                    emit(
                        "timing",
                        seed=seed,
                        M=m,
                        K=k,
                        global_scale=g,
                        method=name,
                        round=round_id,
                        median_us=1000 * statistics.median(samples),
                        samples_us=[1000 * t for t in samples],
                    )
        emit("done", cases=len(cases))


if __name__ == "__main__":
    main()
