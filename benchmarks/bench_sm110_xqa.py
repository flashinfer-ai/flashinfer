# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Seven native SM110 XQA rows using FlashInfer's CUPTI cold-L2 timer.

This reports native latency samples, not a source/export equivalence audit.
Compilation, allocation, and the FP32 oracle run outside measured replay.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import time
import warnings


def inputs_and_reference(shape, recipes):
    import torch

    tree = shape["head_dim"] == 512
    batch, queries, heads, kv_heads, dim, capacity = (
        shape[key]
        for key in (
            "batch",
            "query_tokens",
            "query_heads",
            "kv_heads",
            "head_dim",
            "capacity",
        )
    )
    recipe = recipes["tree" if tree else "decode"]
    generator = torch.Generator(device="cuda").manual_seed(recipe["seed"])

    def sample(size):
        if tree:
            return torch.empty(size, device="cuda", dtype=torch.float16).uniform_(
                recipe["minimum"], recipe["maximum"], generator=generator
            )
        return torch.randn(
            size, generator=generator, device="cuda", dtype=torch.float16
        )

    q = sample((batch, queries, heads, dim) if tree else (batch, heads, dim))
    kv = sample((batch, 2, kv_heads, capacity, dim))
    lengths = torch.full((batch,), capacity, device="cuda", dtype=torch.int32)
    mask = None
    if tree:
        if shape["mask"] != "causal" or queries > 31:
            raise ValueError(
                "this ledger benchmark requires the one-word causal draft mask"
            )
        mask = (
            torch.tensor(
                [[(1 << (row + 1)) - 1] for row in range(queries)],
                device="cuda",
                dtype=torch.int32,
            )
            .unsqueeze(0)
            .repeat(batch, 1, 1)
        )
    k_scale = v_scale = 1.0
    if shape["kv_dtype"] == "float8_e4m3fn":
        k_scale = float(kv[:, 0].float().abs().max()) / 448.0
        v_scale = float(kv[:, 1].float().abs().max()) / 448.0
        scales = torch.tensor([k_scale, v_scale], device="cuda").reshape(1, 2, 1, 1, 1)
        stored = (kv.float() / scales).to(torch.float8_e4m3fn)
        dense = stored.float() * scales
    elif shape["kv_dtype"] == "float16":
        stored, dense = kv, kv.float()
    else:
        raise ValueError("unsupported ledger cache dtype")
    pages = None
    if shape["page_size"]:
        size = shape["page_size"]
        count = capacity // size
        stored = stored.reshape(batch, 2, kv_heads, count, size, dim)
        stored = (
            stored.permute(0, 1, 3, 4, 2, 5)
            .contiguous()
            .reshape(batch * 2 * count, size, kv_heads, dim)
        )
        pages = torch.arange(
            batch * 2 * count, device="cuda", dtype=torch.int32
        ).reshape(batch, 2, count)

    outputs = []
    ratio = heads // kv_heads
    for request in range(batch):
        if tree:
            grouped_q = q[request].float().reshape(queries, kv_heads, ratio, dim)
            scores = torch.einsum(
                "qhrd,hsd->hrqs", grouped_q, dense[request, 0]
            ) / math.sqrt(dim)
            visible = torch.arange(capacity, device="cuda")[None, :] <= (
                capacity - queries + torch.arange(queries, device="cuda")[:, None]
            )
            scores.masked_fill_(~visible[None, None], -torch.inf)
            value = torch.einsum(
                "hrqs,hsd->qhrd", scores.softmax(-1), dense[request, 1]
            ).reshape(queries, heads, dim)
        else:
            grouped_q = q[request].float().reshape(kv_heads, ratio, dim)
            scores = torch.einsum(
                "hrd,hsd->hrs", grouped_q, dense[request, 0]
            ) / math.sqrt(dim)
            value = torch.einsum(
                "hrs,hsd->hrd", scores.softmax(-1), dense[request, 1]
            ).reshape(heads, dim)
        outputs.append(value)
    expected = torch.stack(outputs).to(torch.float16)
    kwargs = dict(
        mask=mask,
        page_table=pages,
        page_size=shape["page_size"],
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return (q, stored, lengths), kwargs, expected


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path, default=Path(__file__).with_name("sm110_xqa_shapes.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--warmup-ms",
        type=float,
        help="upstream timer warmup duration target in milliseconds (default: 250)",
    )
    warmup_group.add_argument(
        "--warmup-iters", type=int, help="explicit warmup count instead of duration"
    )
    parser.add_argument("--repeat-iters", type=int, default=256)
    args = parser.parse_args()
    if args.warmup_iters is not None:
        if args.warmup_iters < 1:
            parser.error("warmup iterations must be positive")
        warmup_options = {"dry_run_iters": args.warmup_iters}
        warmup_policy = {"warmup_iters": args.warmup_iters}
    else:
        duration = 250.0 if args.warmup_ms is None else args.warmup_ms
        if not math.isfinite(duration) or duration <= 0:
            parser.error("--warmup-ms must be finite and positive")
        warmup_options = {"dry_run_time_ms": duration}
        warmup_policy = {"warmup_ms": duration}
    if args.repeat_iters < 1:
        parser.error("measured iterations must be positive")

    # Preflight CUPTI before Torch initializes CUDA. Fail instead of silently
    # selecting an event timer when the profiling dependency is unavailable.
    from cupti import cupti

    cupti_version = version("cupti-python")
    if int(cupti_version.split(".")[0]) < 13:
        raise RuntimeError("this benchmark requires cupti-python >= 13")
    cupti.get_timestamp()

    import torch
    import flashinfer
    from flashinfer.experimental.sm110_xqa.jit import get_manifest, require_sm110
    from flashinfer.sm110_xqa import prepare
    from flashinfer.testing import bench_gpu_time

    started = time.monotonic()  # Physical turnaround only; never GPU timing.
    require_sm110(torch.device("cuda", torch.cuda.current_device()))
    manifest = get_manifest()
    ledger_bytes = args.ledger.read_bytes()
    ledger = json.loads(ledger_bytes)
    if (
        ledger["architecture"] != "sm_110a"
        or ledger["shape_count"] != 7
        or len(ledger["shapes"]) != 7
        or [row["name"] for row in ledger["shapes"]] != manifest["shape_denominator"]
        or hashlib.sha256(ledger_bytes).hexdigest() != manifest["shape_ledger_sha256"]
    ):
        raise ValueError("benchmark ledger must match the frozen seven-row manifest")
    if ledger["inputs"] != {
        "tree": {"distribution": "uniform", "minimum": -1.0, "maximum": 1.0, "seed": 0},
        "decode": {"distribution": "normal", "mean": 0.0, "std": 1.0, "seed": 0},
    }:
        raise ValueError("unsupported input recipe")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "architecture": "sm_110a",
        "device": properties.name,
        "multiprocessors": properties.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "cupti": cupti_version,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "timing": {
            "backend": "cupti",
            "cold_l2": True,
            "fallback": "error",
            "metric": "full callback GPU activity span",
            "cuda_graph": False,
            **warmup_policy,
            "repeat_iters": args.repeat_iters,
            "activity_identity_audit": False,
        },
        "rows": [],
        "complete": False,
    }
    for shape in ledger["shapes"]:
        row_started = time.monotonic()
        inputs, kwargs, expected = inputs_and_reference(shape, ledger["inputs"])
        plan = prepare(*inputs, **kwargs)
        torch.testing.assert_close(plan.run(), expected, atol=0.01, rtol=0.01)
        with warnings.catch_warnings():
            # The upstream CUPTI helper warns immediately before its fallback.
            # Turning that exact warning into an exception prevents execution
            # of the fallback without replacing or monkeypatching the timer.
            warnings.filterwarnings(
                "error",
                message=r"(?s).*Falling back to CUDA events for benchmarking\.",
                category=UserWarning,
            )
            samples = bench_gpu_time(
                plan.run,
                enable_cupti=True,
                cold_l2_cache=True,
                use_cuda_graph=False,
                **warmup_options,
                repeat_iters=args.repeat_iters,
            )
        if len(samples) != args.repeat_iters or any(
            not math.isfinite(value) or value <= 0 for value in samples
        ):
            raise RuntimeError("CUPTI returned an invalid sample set")
        torch.testing.assert_close(plan.run(), expected, atol=0.01, rtol=0.01)
        row = {
            "shape": shape,
            "route": plan.route,
            "correct": True,
            "median_us": 1000 * statistics.median(samples),
            "samples_ms": list(samples),
            "physical_seconds": time.monotonic() - row_started,
        }
        report["rows"].append(row)
        report.update(
            complete=len(report["rows"]) == 7,
            physical_seconds=time.monotonic() - started,
        )
        save(args.output, report)
        print(
            json.dumps(
                {key: value for key, value in row.items() if key != "samples_ms"}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
