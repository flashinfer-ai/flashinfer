#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Compare public Cake MegaMoE policies on exactly sixteen SM103a GPUs.

Run on four nodes with four GPUs each (one command per node):

    torchrun --nnodes=4 --nproc-per-node=4 --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR --master-port=29500 \
        benchmarks/bench_cake_mxfp8_megamoe_ep16.py --output results

Requires CUDA 13+, cupti-python and the experimental backend's dependencies.
The output directory must be new and shared across ranks. No CUDA Graphs or
event-timing fallback are used. Timing includes correlated device copies and
the complete fused-kernel/reducer span, but excludes setup and L2 flushing.
This is a two-policy benchmark, not a comparison against unmodified main.
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import timedelta
from functools import partial
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def gather(value, dist):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def decode_activities(windows, launches, activities, symbols):
    """Map GPU activities to host submissions; reject incomplete kernel pairs."""
    ordered = sorted(launches)
    starts = [record[0] for record in ordered]
    by_correlation = {}
    for record in activities:
        by_correlation.setdefault(record[3], []).append(record)
    samples = []
    for begin, end in windows:
        left = bisect.bisect_left(starts, begin)
        right = bisect.bisect_right(starts, end)
        correlations = {record[2] for record in ordered[left:right]}
        selected = [
            record
            for correlation in correlations
            for record in by_correlation.get(correlation, ())
        ]
        kernels = sorted(
            (record for record in selected if record[4] == "kernel"),
            key=lambda record: record[1],
        )
        require(
            [record[0] for record in kernels] == symbols, "CUPTI kernel pair differs"
        )
        require(
            all(record[2] > record[1] for record in selected), "invalid GPU timestamps"
        )
        samples.append(
            (max(r[2] for r in selected) - min(r[1] for r in selected)) / 1e6
        )
    require(len(samples) == len(windows) and samples, "incomplete CUPTI samples")
    return samples


def measure(fn, *, torch, cupti, flush, count, symbols):
    launches, activities, windows, callback_errors = [], [], [], []

    def buffer_requested():
        return 8 * 1024 * 1024, 0

    def buffer_completed(records):
        try:
            for record in records:
                if record.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                    activities.append(
                        [
                            record.name,
                            int(record.start),
                            int(record.end),
                            int(record.correlation_id),
                            "kernel",
                        ]
                    )
                elif record.kind in (
                    cupti.ActivityKind.MEMCPY,
                    cupti.ActivityKind.MEMSET,
                ):
                    activities.append(
                        [
                            str(record.kind),
                            int(record.start),
                            int(record.end),
                            int(record.correlation_id),
                            "copy",
                        ]
                    )
                elif record.kind in (
                    cupti.ActivityKind.RUNTIME,
                    cupti.ActivityKind.DRIVER,
                ):
                    launches.append(
                        [
                            int(record.start),
                            int(record.end),
                            int(record.correlation_id),
                        ]
                    )
        except Exception as error:
            callback_errors.append(repr(error))

    kinds = [
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ]
    enabled = []
    dropped = ctypes.c_size_t()
    torch.cuda.synchronize()
    cupti.activity_register_callbacks(buffer_requested, buffer_completed)
    try:
        for kind in kinds:
            cupti.activity_enable(kind)
            enabled.append(kind)
        for _ in range(count):
            flush.zero_()
            torch.cuda.synchronize()
            begin = int(cupti.get_timestamp())
            fn()
            end = int(cupti.get_timestamp())
            torch.cuda.synchronize()
            windows.append([begin, end])
        cupti.activity_flush_all(0)
        # Activity buffers use the global queue (null context / stream zero).
        # Fail closed if any activity, including a device copy, was lost.
        cupti.activity_get_num_dropped_records(0, 0, ctypes.addressof(dropped))
        require(dropped.value == 0, f"CUPTI dropped {dropped.value} activity records")
    finally:
        for kind in reversed(enabled):
            cupti.activity_disable(kind)
        cupti.finalize()
    require(not callback_errors, f"CUPTI callback failed: {callback_errors}")
    samples = decode_activities(windows, launches, activities, symbols)
    return {
        "samples_ms": samples,
        "windows": windows,
        "launches": launches,
        "activities": activities,
        "dropped_records": dropped.value,
    }


def make_routes(torch, tokens, profile, seed, rank):
    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    if profile == "balanced":
        return torch.randint(512, (tokens, 8), generator=generator, dtype=torch.int64)
    hot = torch.randint(128, (tokens, 4), generator=generator, dtype=torch.int64)
    other = torch.randint(128, 512, (tokens, 4), generator=generator, dtype=torch.int64)
    routes = torch.cat((hot, other), dim=1)
    for token in range(tokens):
        routes[token] = routes[token, torch.randperm(8, generator=generator)]
    return routes


def reference(torch, dist, hidden, ids, route_weights, transformed, source_oracle):
    def all_gather(tensor):
        parts = [torch.empty_like(tensor) for _ in range(16)]
        dist.all_gather(parts, tensor)
        return torch.cat(parts)

    def k_major(tensor):
        return (
            tensor
            if tensor.stride(1) == 1
            else tensor.transpose(1, 2).contiguous().transpose(1, 2)
        )

    tokens = hidden.shape[0]
    hidden, ids, route_weights = map(all_gather, (hidden, ids, route_weights))
    offset = dist.get_rank() * 32
    routes = ((ids >= offset) & (ids < offset + 32)).nonzero(as_tuple=False)
    terms = torch.zeros((16 * tokens, 8, 3072), dtype=torch.bfloat16, device="cuda")
    if routes.numel():
        token_ids, slots = routes[:, 0], routes[:, 1]
        fc1, fc2 = transformed
        terms[token_ids, slots] = source_oracle.compute_megamoe_reference_bf16_mxfp8(
            input_activation=hidden[token_ids].unsqueeze(0),
            input_topk_idx=(ids[token_ids, slots] - offset)
            .reshape(1, -1, 1)
            .contiguous(),
            input_topk_weights=route_weights[token_ids, slots]
            .reshape(1, -1, 1)
            .contiguous(),
            fc1_weight=k_major(fc1[0]).unsqueeze(0),
            fc1_weight_sf=fc1[1].view(source_oracle.Mxfp8ScaleDtype).unsqueeze(0),
            fc2_weight=k_major(fc2[0]).unsqueeze(0),
            fc2_weight_sf=fc2[1].view(source_oracle.Mxfp8ScaleDtype).unsqueeze(0),
            ref_compute_graph="deepgemm",
            fc2_output_dtype=torch.bfloat16,
            gate_up_clamp=None,
            apply_topk_in_fc1=True,
        )[0, :, 0, :]
    received = torch.empty_like(terms)
    dist.all_to_all_single(received, terms)
    received = received.reshape(16, tokens, 8, 3072)
    result = torch.zeros((tokens, 3072), dtype=torch.float32, device="cuda")
    for slot in range(8):
        for owner in range(16):
            result.add_(received[owner, :, slot].float())
    return result.to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rendezvous-file", type=Path)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--repeat-iters", type=int, default=1000)
    parser.add_argument("--groups", type=int, default=6)
    args = parser.parse_args()
    require(
        args.warmup_iters > 0 and args.repeat_iters > 0, "iterations must be positive"
    )
    require(
        args.groups > 0 and args.groups % 2 == 0, "groups must be positive and even"
    )
    # Import failure is intentional: there is no event-timing fallback.
    import torch
    import torch.distributed as dist
    from cupti import cupti
    from flashinfer.experimental.cake_mxfp8_megamoe_ep16 import jit
    from flashinfer.moe_ep import (
        CakeMxfp8MegaMoeEp16,
        MoEWeightPack,
        preprocess_cake_mxfp8_megamoe_ep16_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src import cutedsl_megamoe as source_oracle
    from flashinfer.testing.utils import get_l2_cache_size

    rank = int(os.environ["RANK"])
    require(int(os.environ["WORLD_SIZE"]) == 16, "requires exactly sixteen ranks")
    device = 0 if torch.cuda.device_count() == 1 else int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    require(torch.cuda.get_device_capability() == (10, 3), "requires SM103a")
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    init_method = (
        args.rendezvous_file.resolve().as_uri() if args.rendezvous_file else "env://"
    )
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=16,
        timeout=timedelta(seconds=1800),
        device_id=torch.device("cuda", device),
    )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    claim = {
        "rank": rank,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "warmup_iters": args.warmup_iters,
        "repeat_iters": args.repeat_iters,
        "groups": args.groups,
        "timing": "CUPTI, cold L2, copy-inclusive span",
    }
    write_json(root / f"rank-{rank}" / "claim.json", claim)
    claims = gather({k: v for k, v in claim.items() if k != "rank"}, dist)
    require(all(value == claims[0] for value in claims), "rank configurations differ")
    _, manifest = jit._read_manifest()
    sequences = {sequence["policy_id"]: sequence for sequence in manifest["sequences"]}
    flush = torch.empty(
        2 * get_l2_cache_size(torch.device("cuda", device)),
        dtype=torch.int8,
        device="cuda",
    )
    generator = torch.Generator(device="cuda").manual_seed(2026103401 + rank)
    w13 = torch.randn(
        (32, 10240, 3072), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    w13.mul_(3072**-0.5)
    w2 = torch.randn(
        (32, 3072, 5120), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    w2.mul_(5120**-0.5)
    transformed = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=5120,
        hidden_size=3072,
        kind="bf16_mxfp8_e4m3",
    )
    prepared = preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)
    official = [
        transformed[0][0].transpose(1, 2).contiguous().view(torch.uint8),
        transformed[0][1].contiguous().view(torch.uint8).reshape(-1, 128),
        transformed[1][0].transpose(1, 2).contiguous().view(torch.uint8),
        transformed[1][1].contiguous().view(torch.uint8).reshape(-1, 128),
    ]
    actual = [
        prepared.w13.view(torch.uint8),
        prepared.w13_scale.reshape(-1, 128),
        prepared.w2.view(torch.uint8),
        prepared.w2_scale.reshape(-1, 128),
    ]
    require(
        all(
            gather(
                all(torch.equal(a, b) for a, b in zip(official, actual, strict=True)),
                dist,
            )
        ),
        "independent preprocessing differs",
    )
    del official, actual
    rows = []
    # Keep sessions alive until all collective GPU work has finished.
    sessions = []
    for profile in ("balanced", "hot"):
        for index, tokens in enumerate((16, 32, 64)):
            label = f"{16 * tokens}-{profile}"
            if rank == 0:
                print(f"benchmark {label}", flush=True)
            generator = torch.Generator(device="cuda").manual_seed(
                2026090401 + index + rank
            )
            hidden = torch.randn(
                (tokens, 3072), dtype=torch.bfloat16, device="cuda", generator=generator
            )
            seed = 2026110401 + index + (50 if profile == "hot" else 0)
            ids = make_routes(torch, tokens, profile, seed, rank).to("cuda")
            counts = torch.bincount(ids.flatten(), minlength=512)
            dist.all_reduce(counts)
            require(int(counts.max()) <= 64, "routing exceeds expert capacity")
            generator = torch.Generator(device="cuda").manual_seed(seed + rank)
            route_weights = torch.softmax(
                torch.randn((tokens, 8), device="cuda", generator=generator), dim=-1
            )
            expected = reference(
                torch, dist, hidden, ids, route_weights, transformed, source_oracle
            )
            policies = {"default": ("mixed", "auto"), "n32": (32, "cta0")}
            functions, symbols, initial = {}, {}, {}
            for name, (tile, protocol) in policies.items():
                operand, routing, weights = (
                    hidden.clone(),
                    ids.clone(),
                    route_weights.clone(),
                )
                session = (
                    CakeMxfp8MegaMoeEp16(prepared, routing)
                    if name == "default"
                    else CakeMxfp8MegaMoeEp16(
                        prepared, routing, tile_n=tile, return_protocol=protocol
                    )
                )
                expected_policy = (
                    "mixed_" + ("all_cta" if tokens == 32 else "cta0")
                    if name == "default"
                    else "n32_cta0"
                )
                require(
                    session.policy_id == expected_policy,
                    "public policy selection changed",
                )
                sessions.append(session)
                functions[name] = partial(
                    session.run, operand, routing, weights, out=session.workspace_output
                )
                sequence = sequences[session.policy_id]
                symbols[name] = [sequence["fused_symbol"], sequence["reducer_symbol"]]
                replay_ok = True
                for replay in range(33):
                    output = functions[name]()
                    torch.cuda.synchronize()
                    if replay == 0:
                        initial[name] = output.clone()
                    replay_ok &= torch.equal(
                        output.view(torch.uint8), initial[name].view(torch.uint8)
                    )
                error = torch.linalg.vector_norm(
                    initial[name].float() - expected.float()
                )
                norm = torch.linalg.vector_norm(expected.float())
                relative_l2 = float(error / torch.clamp(norm, min=1e-30))
                correct = (
                    bool(torch.isfinite(initial[name]).all())
                    and torch.allclose(initial[name], expected, atol=1e-2, rtol=1e-2)
                    and relative_l2 <= 0.02
                    and replay_ok
                )
                require(all(gather(correct, dist)), "oracle or replay check failed")
                write_json(
                    root / f"rank-{rank}" / label / f"{name}-correctness.json",
                    {"passed": correct, "replays": 33, "relative_l2": relative_l2},
                )
            group_times = {name: [] for name in policies}
            for group in range(args.groups):
                order = list(policies) if group % 2 == 0 else list(reversed(policies))
                for name in order:
                    dist.barrier()
                    for _ in range(args.warmup_iters):
                        flush.zero_()
                        torch.cuda.synchronize()
                        functions[name]()
                        torch.cuda.synchronize()
                    timing = measure(
                        functions[name],
                        torch=torch,
                        cupti=cupti,
                        flush=flush,
                        count=args.repeat_iters,
                        symbols=symbols[name],
                    )
                    rank_medians = gather(statistics.median(timing["samples_ms"]), dist)
                    require(
                        all(
                            math.isfinite(value) and value > 0 for value in rank_medians
                        ),
                        "invalid timing",
                    )
                    group_times[name].append(max(rank_medians))
                    timing.update(rank_medians_ms=rank_medians, symbols=symbols[name])
                    write_json(
                        root / f"rank-{rank}" / label / f"{name}-group-{group}.json",
                        timing,
                    )
            for name, fn in functions.items():
                output = fn()
                torch.cuda.synchronize()
                require(
                    all(
                        gather(
                            torch.equal(
                                output.view(torch.uint8),
                                initial[name].view(torch.uint8),
                            ),
                            dist,
                        )
                    ),
                    "post-timing replay differs",
                )
            medians = {
                name: statistics.median(values) for name, values in group_times.items()
            }
            row = {
                "total_tokens": tokens * 16,
                "routing": profile,
                "groups_ms": group_times,
                "median_max_rank_ms": medians,
                "n32_over_default": medians["n32"] / medians["default"],
            }
            rows.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    write_json(root / f"rank-{rank}" / "result.json", {"claim": claim, "rows": rows})
    dist.barrier()
    dist.destroy_process_group()
    result_path = root / f"rank-{rank}" / "result.json"
    write_json(
        root / f"rank-{rank}" / "finalized.json",
        {
            "rank": rank,
            "main_return_code": 0,
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "after": "GPU synchronization, durable results, rank barrier and process-group destruction",
        },
    )
    # Standalone distributed benchmark: all work and durable outputs precede
    # this exit. Avoid extension finalizer ordering; exceptions never reach it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
