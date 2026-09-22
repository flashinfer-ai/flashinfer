# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Fixed-32K sparse MLA prefill/decode shape suite (30 + 450 cases).

Run from the repository root. Both backends consume the same actual tensors.
Preparation, index selection, hashing and FP64 checking are outside timing.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import torch

from bench_attention_ts_sparse_mla import (
    Fixture,
    accuracy,
    make_backend,
    percentile,
    prepared_metadata_fingerprint,
)
from sparse_mla_bench_utils import ColdL2GraphBenchmark
from sparse_mla_model_fixtures import (
    chunked_reference,
    model_fixture,
    streaming_fingerprint,
)


def cases():
    result = []
    for phase, batches, queries in (
        ("prefill", [1], [8192]),
        ("decode", [1, 4, 16, 64, 256], [1, 4, 8]),
    ):
        for b, q, k, h, dtype in itertools.product(
            batches, queries, [512, 1024, 2048], [8, 16, 32, 64, 128], ["bf16", "fp8"]
        ):
            result.append(
                dict(
                    id=len(result),
                    phase=phase,
                    batch=b,
                    queries=q,
                    heads=h,
                    topk=k,
                    dtype=dtype,
                )
            )
    return result


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def provenance(args):
    prop = torch.cuda.get_device_properties(0)
    if (prop.major, prop.minor) not in ((10, 0), (10, 3)):
        raise RuntimeError("native sparse MLA requires SM100/SM103")
    names = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--",
                "flashinfer/attention/prims_ts",
                "flashinfer/prims_ts",
                "flashinfer/experimental/prims_ts_sparse_mla",
            ]
        )
        .decode()
        .splitlines()
    )
    digest = hashlib.sha256()
    for name in sorted(n for n in names if n.endswith(".py")):
        digest.update(name.encode() + b"\0" + Path(name).read_bytes())
    files = [
        Path(__file__),
        Path(__file__).with_name("sparse_mla_model_fixtures.py"),
        Path(__file__).with_name("sparse_mla_bench_utils.py"),
        Path(__file__).with_name("bench_attention_ts_sparse_mla.py"),
    ]
    from flashinfer.testing import sparse_mla_metadata

    files.append(Path(sparse_mla_metadata.__file__))
    return dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        source_tree_sha256=digest.hexdigest(),
        source_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--", "flashinfer"])
        ).hexdigest(),
        harness_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        gpu=prop.name,
        gpu_uuid=str(prop.uuid),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        sm_count=prop.multi_processor_count,
        l2_bytes=prop.L2_cache_size,
        compute_capability=[prop.major, prop.minor],
        job_id=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME"),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        driver=subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        )
        .decode()
        .strip(),
        protocol=(
            "identical-input/prepared-input/cold-L2/CUDA-Graph"
            if args.prepared
            else "identical-input/external-preparation/cold-L2/CUDA-Graph"
        ),
        timing_scope="prepared-attention"
        if args.prepared
        else "external-preparation+attention",
        eviction_multiplier=4,
        replays=args.replays,
        samples_per_replay=args.samples_per_replay,
        seed=args.seed,
        raw_kv_tokens=args.raw_kv_tokens,
        compression_ratio=args.compression_ratio,
        causal=not args.noncausal,
        swa_window=0 if args.no_swa else 128,
        swa_page=256,
        candidate_page=1,
        head_dim=512,
        reference="chunked full-output FP64; probability/output rounding bound for FP8",
        reference_chunk_rows=args.reference_chunk_rows,
        fixture="FlashMLA random-score top-k + random physical pages + sliding SWA",
        backends=args.backends.split(","),
        requested_cases=cases(),
    )


def run_case(case, args, source):
    started = time.perf_counter()
    fixture_data, layout = model_fixture(
        case["batch"],
        case["heads"],
        case["queries"],
        case["topk"],
        torch.bfloat16 if case["dtype"] == "bf16" else torch.float8_e4m3fn,
        raw_tokens=args.raw_kv_tokens,
        compression_ratio=args.compression_ratio,
        causal=not args.noncausal,
        seed=args.seed,
    )
    fixture = Fixture(**fixture_data)
    del fixture_data
    if args.no_swa:
        fixture.si.fill_(-1)
        fixture.sl.zero_()
        fixture.combined[:, :128].fill_(-1)
        fixture.combined_lengths.copy_(128 + fixture.cl.reshape(-1))
        if fixture.swa.dtype == torch.float8_e4m3fn:
            fixture.swa.view(torch.uint8).fill_(0x7F)
        else:
            fixture.swa.fill_(torch.nan)
        layout["source_mode"] = "single"
    fingerprint = streaming_fingerprint(fixture)
    result = dict(
        **case,
        **layout,
        source=source,
        fixture_id=fingerprint,
        backends={},
        complete=False,
    )
    case_path = (
        args.output
        / f"{case['id']:03d}-{case['phase']}-b{case['batch']}-q{case['queries']}-h{case['heads']}-k{case['topk']}-{case['dtype']}.json"
    )
    save(case_path, result)
    expected, _, bound = chunked_reference(
        fixture, chunk_rows=args.reference_chunk_rows
    )
    result["preparation_and_reference_s"] = time.perf_counter() - started
    runners, outputs = {}, {}
    for name in args.backends.split(","):
        try:
            fn, out, metadata = make_backend(
                name, fixture, prepared=args.prepared, single_source=args.no_swa
            )
            torch.cuda.synchronize()
            metrics = accuracy(out, expected, bound)
            runner = ColdL2GraphBenchmark(
                fn,
                device=fixture.query.device,
                samples_per_replay=args.samples_per_replay,
            )
            runner.sample()
            accuracy(out, expected, bound)
            runners[name], outputs[name] = runner, out
            result["backends"][name] = dict(
                status="ok", **metadata, **metrics, times_us=[]
            )
        except Exception as error:
            result["backends"][name] = dict(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            save(case_path, result)
            if any(
                t in str(error).lower()
                for t in ("illegal memory", "device-side assert", "unspecified launch")
            ):
                raise
    for iteration in range(args.replays):
        order = list(runners)
        if iteration % 2:
            order.reverse()
        for name in order:
            result["backends"][name]["times_us"].extend(runners[name].sample())
    for name in runners:
        entry = result["backends"][name]
        times = entry["times_us"]
        try:
            final_metrics = accuracy(outputs[name], expected, bound)
            if prepared_metadata_fingerprint(runners[name].fn) != entry.get(
                "prepared_metadata_sha256"
            ):
                raise AssertionError("backend mutated prepared input metadata")
        except Exception as error:
            # A backend can pass warmup and fail only after graph replays.
            # Keep that failure explicit and discard every measured sample.
            entry.clear()
            entry.update(
                status="failed",
                failure_stage="post_replay_accuracy",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            save(case_path, result)
            if any(
                t in str(error).lower()
                for t in ("illegal memory", "device-side assert", "unspecified launch")
            ):
                raise
        else:
            entry.update(
                median_us=statistics.median(times),
                p10_us=percentile(times, 0.1),
                p90_us=percentile(times, 0.9),
                **final_metrics,
            )
    if streaming_fingerprint(fixture) != fingerprint:
        raise AssertionError("backend mutated the shared fixture")
    result["complete"] = True
    result["elapsed_s"] = time.perf_counter() - started
    save(case_path, result)
    concise = {
        name: round(v["median_us"], 3) if v["status"] == "ok" else v["error"]
        for name, v in result["backends"].items()
    }
    print(f"DONE {case} {concise} elapsed={result['elapsed_s']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--indices", help="Optional comma-separated case IDs for preflight"
    )
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=480)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--raw-kv-tokens", type=int, default=32768)
    parser.add_argument("--compression-ratio", type=int, default=1)
    parser.add_argument("--noncausal", action="store_true")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument(
        "--prepared",
        action="store_true",
        default=True,
        help="Exclude external metadata preparation for both backends (default)",
    )
    timing.add_argument(
        "--include-preparation",
        dest="prepared",
        action="store_false",
        help="Measure explicit external preparation plus attention as a separate protocol",
    )
    parser.add_argument(
        "--no-swa",
        action="store_true",
        help="Prepared single-source trial; TRT retains its masked 128-column ABI segment",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backends", default="ts-auto,trtllm-gen")
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--samples-per-replay", type=int, default=4)
    parser.add_argument("--reference-chunk-rows", type=int, default=32)
    args = parser.parse_args()
    if args.no_swa and not args.prepared:
        parser.error("--no-swa requires --prepared")
    if min(args.replays, args.samples_per_replay, args.reference_chunk_rows) < 1:
        parser.error("replay/sample/reference sizes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = provenance(args)
    manifest_path = args.output / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "source_tree_sha256",
            "harness_sha256",
            "raw_kv_tokens",
            "compression_ratio",
            "causal",
            "seed",
            "backends",
            "replays",
            "samples_per_replay",
            "protocol",
            "swa_window",
        ):
            if previous[key] != manifest[key]:
                raise ValueError(f"resume protocol mismatch: {key}")
    save(manifest_path, manifest)
    source = {k: v for k, v in manifest.items() if k != "requested_cases"}
    requested_ids = set(map(int, args.indices.split(","))) if args.indices else None
    for case in cases():
        if not args.start <= case["id"] < args.stop:
            continue
        if args.phase != "all" and case["phase"] != args.phase:
            continue
        if requested_ids is not None and case["id"] not in requested_ids:
            continue
        previous = list(args.output.glob(f"{case['id']:03d}-*.json"))
        if (
            args.resume
            and previous
            and json.loads(previous[0].read_text()).get("complete")
        ):
            continue
        print(f"START {case}", flush=True)
        run_case(case, args, source)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
