"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Paired GPU-time comparison of prepared Hopper VSA backends on the 20 representative shapes.

Run from an editable FlashInfer checkout on H100:

    python benchmarks/bench_cake_vsa_sm90.py --output /tmp/vsa-sm90-cake.json
    python benchmarks/bench_cake_vsa_sm90.py --output ... --baseline vsa_sm90_blk64   # PR #5470 checkout
    python benchmarks/bench_cake_vsa_sm90.py --output ... --warm-l2                    # PR #5470 regime

Planning, host-to-device metadata copies, JIT compilation, input generation and output
allocation happen before timing; the same prepared ``run`` is measured for both arms in
alternating AB/BA order.  CUPTI measures kernel activity (a fallback to CUDA events is an
error) with a cold L2 cache by default; ``--warm-l2`` reproduces the warm regime of
FlashInfer PR #5470 and must never be mixed with cold numbers.  This is not an
end-to-end host latency benchmark.
"""

import argparse
import json
import math
import statistics
import warnings
from pathlib import Path

import torch

from flashinfer.sparse import VariableBlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time

CASES = [
    {
        "uuid": "h1-m64-n64-k1-ragged0-scale0",
        "axes": {"H": 1, "M": 64, "N": 64, "K": 1, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m256-n256-k1-ragged0-scale0",
        "axes": {"H": 4, "M": 256, "N": 256, "K": 1, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m256-n256-k3-ragged0-scale0",
        "axes": {"H": 8, "M": 256, "N": 256, "K": 3, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m1024-n1024-k4-ragged0-scale0",
        "axes": {"H": 8, "M": 1024, "N": 1024, "K": 4, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m1024-n1024-k12-ragged0-scale0",
        "axes": {"H": 8, "M": 1024, "N": 1024, "K": 12, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m2048-n2048-k8-ragged0-scale0",
        "axes": {"H": 8, "M": 2048, "N": 2048, "K": 8, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m2048-n2048-k16-ragged0-scale0",
        "axes": {"H": 8, "M": 2048, "N": 2048, "K": 16, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m4096-n4096-k16-ragged0-scale0",
        "axes": {"H": 8, "M": 4096, "N": 4096, "K": 16, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h8-m4096-n4096-k32-ragged0-scale0",
        "axes": {"H": 8, "M": 4096, "N": 4096, "K": 32, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m512-n1024-k4-ragged0-scale0",
        "axes": {"H": 4, "M": 512, "N": 1024, "K": 4, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m1024-n512-k4-ragged1-scale0",
        "axes": {"H": 4, "M": 1024, "N": 512, "K": 4, "RAGGED": 1, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m1024-n1024-k8-ragged1-scale0",
        "axes": {"H": 4, "M": 1024, "N": 1024, "K": 8, "RAGGED": 1, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m4096-n4096-k16-ragged1-scale0",
        "axes": {"H": 7, "M": 4096, "N": 4096, "K": 16, "RAGGED": 1, "SCALE_HALF": 0},
    },
    {
        "uuid": "h4-m512-n512-k4-ragged0-scale1",
        "axes": {"H": 4, "M": 512, "N": 512, "K": 4, "RAGGED": 0, "SCALE_HALF": 1},
    },
    {
        "uuid": "h1-m1024-n1024-k16-ragged0-scale0",
        "axes": {"H": 1, "M": 1024, "N": 1024, "K": 16, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m16384-n16384-k32-ragged0-scale0",
        "axes": {"H": 7, "M": 16384, "N": 16384, "K": 32, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m32768-n32768-k64-ragged0-scale0",
        "axes": {"H": 7, "M": 32768, "N": 32768, "K": 64, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m65536-n65536-k64-ragged0-scale0",
        "axes": {"H": 7, "M": 65536, "N": 65536, "K": 64, "RAGGED": 0, "SCALE_HALF": 0},
    },
    {
        "uuid": "h7-m109632-n109632-k64-ragged0-scale0",
        "axes": {
            "H": 7,
            "M": 109632,
            "N": 109632,
            "K": 64,
            "RAGGED": 0,
            "SCALE_HALF": 0,
        },
    },
    {
        "uuid": "h7-m109632-n109632-k64-ragged1-scale0",
        "axes": {
            "H": 7,
            "M": 109632,
            "N": 109632,
            "K": 64,
            "RAGGED": 1,
            "SCALE_HALF": 0,
        },
    },
]


def make_inputs(axes, seed):
    """Deterministic inputs identical to ``benchmarks/bench_vsa_sm90.py`` of PR #5470."""
    h, m, n, topk = (axes[name] for name in ("H", "M", "N", "K"))
    cpu_rng = torch.Generator(device="cpu").manual_seed(seed)
    gpu_rng = torch.Generator(device="cuda").manual_seed(seed)
    indices = (
        torch.rand((h, m // 64, n // 64), generator=cpu_rng).topk(topk, dim=-1).indices
    )
    indices = indices.sort(dim=-1).values
    if axes["RAGGED"]:
        counts = torch.randint(1, topk + 1, (h, m // 64), generator=cpu_rng)
        counts[:, 0] = topk
    else:
        counts = torch.full((h, m // 64), topk)
    mask = torch.zeros((h, m // 64, n // 64), dtype=torch.bool)
    mask.scatter_(-1, indices, torch.arange(topk) < counts.unsqueeze(-1))
    q, k, v = (
        torch.randn(
            (h, length, 128), device="cuda", dtype=torch.bfloat16, generator=gpu_rng
        )
        for length in (m, n, n)
    )
    scale = 0.5 if axes["SCALE_HALF"] else 128**-0.5
    return mask, q, k, v, scale


def prepare(mask, backend, scale):
    h, mb, nb = mask.shape
    workspace_size = 0 if backend in ("cake", "vsa_sm90_blk64") else 128 * 1024 * 1024
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(workspace_size, dtype=torch.uint8, device="cuda"), backend=backend
    )
    mask = mask.to("cuda")
    wrapper.plan(
        mask,
        torch.full((h, mb), 64, dtype=torch.int32, device="cuda"),
        torch.full((h, nb), 64, dtype=torch.int32, device="cuda"),
        h,
        h,
        128,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        non_blocking=False,
    )
    return wrapper


def main():
    parser = argparse.ArgumentParser(
        description="Paired comparison of prepared Hopper VSA backends; requires H100 and CUPTI."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--case", help="Run one named workload instead of all 20")
    parser.add_argument(
        "--candidate", default="cake", help="Backend under test (default: cake)"
    )
    parser.add_argument(
        "--baseline",
        default="fa3",
        help="Reference backend: fa3 (main) or vsa_sm90_blk64 (a PR #5470 checkout)",
    )
    parser.add_argument(
        "--warm-l2",
        action="store_true",
        help="Warm-L2 regime of PR #5470 (reported separately)",
    )
    parser.add_argument("--repeat-iters", type=int, default=50)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        parser.error("Requires an SM90 GPU")
    cases = [case for case in CASES if args.case is None or case["uuid"] == args.case]
    if not cases:
        parser.error("Unknown --case")
    names = (args.baseline, args.candidate)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": args.seed,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "timing": "prepared GPU kernel activity, CUPTI, "
        + ("warm L2" if args.warm_l2 else "cold L2")
        + ", AB/BA",
        "cases": [],
    }
    for case in cases:
        mask, q, k, v, scale = make_inputs(case["axes"], args.seed)
        wrappers = {name: prepare(mask, name, scale) for name in names}
        outputs = {
            name: torch.empty(
                (q.shape[0] * q.shape[1], 1, 128), dtype=q.dtype, device=q.device
            )
            for name in wrappers
        }
        calls = {
            name: (
                lambda name=name: wrappers[name].run(
                    q, k, v, out=outputs[name], enable_pdl=False
                )
            )
            for name in wrappers
        }
        for call in calls.values():
            for _ in range(10):
                call()
        torch.cuda.synchronize()
        expected, actual = outputs[args.baseline], outputs[args.candidate]
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        max_error = float((actual.float() - expected.float()).abs().max())
        if max_error > 0.03:
            raise AssertionError(
                f"{case['uuid']}: maximum absolute error {max_error} > 0.03"
            )
        samples = {name: [] for name in wrappers}
        rounds = []
        for pair in range(args.pairs * 2):
            order = list(wrappers) if pair % 2 == 0 else list(reversed(wrappers))
            measured = {}
            for name in order:
                with warnings.catch_warnings():
                    # Do not silently change the timing boundary to host launch gaps.
                    warnings.filterwarnings(
                        "error", message=".*Falling back to CUDA events.*"
                    )
                    values = bench_gpu_time(
                        calls[name],
                        enable_cupti=True,
                        use_cuda_graph=False,
                        cold_l2_cache=not args.warm_l2,
                        dry_run_iters=10,
                        repeat_iters=args.repeat_iters,
                    )
                measured[name] = statistics.median(values)
                samples[name].append(measured[name])
            rounds.append({"order": order, "time_ms": measured})
        medians = {name: statistics.median(values) for name, values in samples.items()}
        row = {
            **case,
            "max_abs_error": max_error,
            "median_ms": medians,
            "speedup": medians[args.baseline] / medians[args.candidate],
            "rounds": rounds,
        }
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case['uuid']}: {args.baseline}={medians[args.baseline] * 1e3:.3f} us; "
            f"{args.candidate}={medians[args.candidate] * 1e3:.3f} us; {row['speedup']:.3f}x",
            flush=True,
        )
    report["latency_geomean_us"] = {
        backend: 1000
        * math.exp(
            statistics.mean(
                math.log(row["median_ms"][backend]) for row in report["cases"]
            )
        )
        for backend in names
    }
    report["speedup_geomean"] = math.exp(
        statistics.mean(math.log(row["speedup"]) for row in report["cases"])
    )
    report["speedup_min"] = min(row["speedup"] for row in report["cases"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"geomean speedup {args.candidate} vs {args.baseline}: {report['speedup_geomean']:.4f} (min {report['speedup_min']:.3f})"
    )


if __name__ == "__main__":
    main()
