#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Complete distributed MoK validation on one peer-accessible GPU domain.

Launch one process per GPU with torchrun; WORLD_SIZE is 4, 8, 16, 32 or 64.
Defaults exercise 16,384 source tokens/rank, H=6144, I=2048, E=256, top-8;
``--experts 288`` selects the GLM-5.3-Flash layout (EP8 or EP32) and
``--swiglu-limit`` its clamped SwiGLU. Use --layout and --routing to run
matrix rows in independent processes. BF16 runs gate every element of all
nine outputs against the BF16 autograd reference; ``--mxfp8`` runs routed
experts natively in MXFP8 and gates the global relative L2 error of the
routed-dependent outputs against an FP32 reference plus the BF16 shared
expert weight gradients elementwise. All reference checks are outside the
benchmark and captured training graph.
"""

import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from flashinfer.mok import (
    create_mok_bf16_workspace,
    prepare_mok_bf16,
    quantize_mok_mxfp8_weights,
)
from mok_bf16_toy import RESULT_NAMES, TrainingIteration, error_report, reference

MXFP8_REL_L2 = 0.15


def source_counts(ep, layout, tokens):
    if layout == "fixed":
        return [tokens] * ep
    if layout == "near":
        # Distinct neighboring lengths with exactly the same total real input.
        return [tokens + 17 * (2 * rank - ep + 1) for rank in range(ep)]
    if layout == "strong":
        pattern = [
            0,
            32768,
            6144,
            26624,
            8192,
            24576,
            10240,
            22528,
            12288,
            20480,
            14336,
            18432,
            15360,
            17408,
            16001,
            16767,
        ]
        if tokens != 16384:
            raise ValueError("The strong layout is defined for 16384 tokens/rank")
        return (pattern * max(1, ep // len(pattern)))[:ep]
    if layout == "empty":
        return [0] * ep
    raise ValueError(layout)


def make_weights(rank, ep, hidden, intermediate, device, experts=256):
    def one(seed):
        torch.manual_seed(seed)
        return (
            (torch.randn(intermediate, hidden, device=device) / hidden**0.5).bfloat16(),
            (torch.randn(intermediate, hidden, device=device) / hidden**0.5).bfloat16(),
            (
                torch.randn(hidden, intermediate, device=device) / intermediate**0.5
            ).bfloat16(),
        )

    shared = one(33471)
    local_experts = experts // ep
    routed = [
        one(32471 + e) for e in range(rank * local_experts, (rank + 1) * local_experts)
    ]
    return (*shared, *(torch.stack([v[k] for v in routed]) for k in range(3)))


def make_data(counts, hidden, routing, generation, device, experts=256):
    total = sum(counts)
    x = torch.empty(total, hidden, device=device, dtype=torch.bfloat16)
    dy = torch.empty_like(x)
    ids = torch.empty(total, 8, device=device, dtype=torch.int64)
    scores = torch.empty(total, 8, device=device, dtype=torch.float32)
    offset = 0
    for source, count in enumerate(counts):
        torch.manual_seed(8291 + generation * 100000 + source * 1000000)
        rows = slice(offset, offset + count)
        x[rows].copy_(torch.randn(count, hidden, device=device).bfloat16())
        dy[rows].copy_(
            (torch.randn(count, hidden, device=device) / hidden**0.5).bfloat16()
        )
        # Sampling without replacement keeps eight distinct expert IDs/token.
        if routing == "uniform":
            chosen = torch.rand(count, experts, device=device).topk(8, dim=-1).indices
        else:
            # Four of the first quarter of experts and four of the rest.
            hot_n = experts // 4
            hot = torch.rand(count, hot_n, device=device).topk(4, dim=-1).indices
            cold = torch.rand(count, experts - hot_n, device=device).topk(4, dim=-1)
            chosen = torch.cat((hot, cold.indices + hot_n), dim=-1)
        ids[rows].copy_(chosen)
        s = torch.rand(count, 8, device=device) + 0.125
        scores[rows].copy_(s / s.sum(-1, keepdim=True) * 2.5)
        offset += count
    return dict(x=x, d_output=dy, expert_ids=ids, scores=scores)


def local_inputs(data, counts, rank):
    start = sum(counts[:rank])
    section = slice(start, start + counts[rank])
    return [
        data[key][section].clone() for key in ("x", "expert_ids", "scores", "d_output")
    ]


def snapshot(outputs, path=None):
    saved = {
        name: value.detach().cpu().clone()
        for name, value in zip(RESULT_NAMES, outputs, strict=True)
    }
    hashes = {
        name: hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, value in saved.items()
    }
    if path is not None:
        torch.save(saved, path)
    return hashes


def audit_routes(iteration, data, counts, rank, ep):
    storage = iteration.workspace.storage
    experts = iteration.weights[3].shape[0] * ep
    expected = torch.full_like(storage.all_gather_top_experts_buffer, -1)
    offset = 0
    for peer, count in enumerate(counts):
        expected[peer, :count].copy_(data["expert_ids"][offset : offset + count])
        offset += count
    assert torch.equal(expected, storage.all_gather_top_experts_buffer)
    histogram = torch.bincount(data["expert_ids"].flatten(), minlength=experts)
    local_experts = experts // ep
    own = histogram[rank * local_experts : (rank + 1) * local_experts]
    padded = ((own + 255) // 256) * 256
    schedule = iteration.schedule
    assert torch.equal(schedule.tokens_per_expert, padded.int())
    assert (
        schedule.num_tokens.item() == padded.sum().item() <= storage.schedule_capacity
    )
    valid = schedule.peer_rank >= 0
    peers = schedule.peer_rank[valid].long()
    indices = schedule.peer_token_idx[valid].long()
    assert valid.sum().item() == own.sum().item()
    limits = torch.tensor(counts, device=storage.device)[peers] * 8
    assert torch.all((indices >= 0) & (indices < limits)).item()
    # Only the active 256-row prefix past the real rows is padded and read;
    # rows beyond it (reserved capacity) are never touched.
    n = counts[rank]
    rows = max(256, (n + 255) // 256 * 256)
    for tail in (
        storage.d_y_buffer[n:rows],
        storage.combine_buffer[n * 8 : rows * 8],
        storage.d_x_routed_buffer[n * 8 : rows * 8],
        storage.d_router_weight_buffer[n:rows],
    ):
        assert torch.count_nonzero(tail).item() == 0
    assert torch.all(storage.router_weight_buffer[n:rows] == 1).item()
    if n == 0:
        assert all(torch.count_nonzero(t).item() == 0 for t in iteration.outputs[6:])
    dist.barrier()
    return dict(
        real_received_routes=int(own.sum()),
        padded_received_routes=int(padded.sum()),
        source_capacity=storage.num_local_tokens,
        schedule_capacity=storage.schedule_capacity,
    )


def check_peer_access(workspace, rank, ep):
    storage = workspace.storage
    storage.x_buffer.fill_(rank + 1)
    torch.cuda.synchronize()
    dist.barrier()
    for peer in range(ep):
        view = storage.x_buffer_handle.get_buffer(
            peer, storage.x_buffer.shape, dtype=storage.x_buffer.dtype
        )
        assert torch.all(view == peer + 1).item()
    assert len(storage.x_buffer_ptrs) == ep and all(storage.x_buffer_ptrs)
    assert storage.all_gather_top_experts_buffer_multicast_ptr
    assert storage.barrier_buffer_multicast_ptr
    torch.cuda.synchronize()
    dist.barrier()


def benchmark(iteration, groups):
    """Completed GPU-event timing of the full graph, with warm fixed inputs."""

    def measure(budget_ms):
        elapsed, repetitions = 0.0, 0
        while True:
            dist.barrier()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(8):
                iteration.run()
            end.record()
            end.synchronize()
            elapsed += start.elapsed_time(end)
            repetitions += 8
            minimum = torch.tensor(
                elapsed, dtype=torch.float64, device=iteration.x.device
            )
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            if minimum.item() >= budget_ms:
                break
        return elapsed, repetitions

    records = []
    for group in range(groups):
        warm_ms, warm_repetitions = measure(100.0)
        elapsed_ms, repetitions = measure(1000.0)
        local = dict(
            rank=dist.get_rank(),
            elapsed_ms=elapsed_ms,
            repetitions=repetitions,
            ms_per_iteration=elapsed_ms / repetitions,
            warmup_ms=warm_ms,
            warmup_repetitions=warm_repetitions,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, local)
        records.append(
            dict(
                group=group,
                critical_rank_ms=max(r["ms_per_iteration"] for r in ranks),
                ranks=ranks,
            )
        )
    return dict(
        timer="completed_cuda_events",
        cache_policy="warm_fixed_input",
        scope="complete_training_graph_including_copies_resets_schedule_and_recompute",
        groups=records,
    )


def run(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", device_id=device, timeout=datetime.timedelta(seconds=600)
    )
    rank, ep = dist.get_rank(), dist.get_world_size()
    if ep not in (4, 8, 16, 32, 64) or args.experts % ep:
        raise ValueError("Launch with 4, 8, 16, 32 or 64 ranks dividing the experts")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    counts = source_counts(ep, args.layout, args.tokens)
    if any(n < 0 for n in counts):
        raise ValueError("Source counts must be nonnegative")
    rank_dir = args.output / f"rank-{rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    limit = args.swiglu_limit
    functional = prepare_mok_bf16(
        ep_size=ep,
        local_experts=args.experts // ep,
        topk=8,
        clamped_swiglu=limit is not None,
    )
    functional.prepare(limit, args.mxfp8)
    config, workspace = create_mok_bf16_workspace(
        group=dist.group.WORLD,
        device=device,
        num_local_tokens=counts[rank],
        hidden_size=args.hidden,
        topk=8,
        fwd_num_comm_sms=40,
        bwd_num_comm_sms=40,
        minibatch_size=4096,
        macrobatch_size=393216,
        schedule_capacity_multiplier=3 / ep,
    )
    assert workspace.initial_source_counts == tuple(counts)
    check_peer_access(workspace, rank, ep)
    weights = make_weights(
        rank, ep, args.hidden, args.intermediate, device, args.experts
    )
    mxfp8_weights = quantize_mok_mxfp8_weights(*weights[3:]) if args.mxfp8 else None
    options = dict(
        functional=functional, swiglu_limit=limit, mxfp8_weights=mxfp8_weights
    )
    gate = dict(atol=1e-2, rtol=1e-2)

    def data_for(lengths, generation):
        return make_data(
            lengths, args.hidden, args.routing, generation, device, args.experts
        )

    def expected_for(source, lengths):
        # MXFP8 runs are gated against FP32; BF16 runs against the BF16 reference.
        bf16 = reference(source, weights, source_counts=lengths, swiglu_limit=limit)
        if not args.mxfp8:
            return bf16, None
        fp32 = reference(
            source, weights, fp32=True, source_counts=lengths, swiglu_limit=limit
        )
        return bf16, fp32

    data = data_for(counts, 0)
    first = TrainingIteration(
        config, workspace, *local_inputs(data, counts, rank), weights, **options
    )
    first.capture()
    reports = []

    def check(iteration, expected, source, lengths, label, save=False):
        actual = iteration.run()
        torch.cuda.synchronize()
        bf16, fp32 = expected
        errors = error_report(actual, bf16, gate)
        if fp32 is not None:
            # Routed-dependent outputs carry MXFP8 quantization error: gate their
            # global relative L2 against FP32; the BF16 shared expert stays strict.
            for name, a, b in zip(RESULT_NAMES[:6], actual[:6], fp32[:6], strict=True):
                stats = torch.zeros(2, dtype=torch.float64, device=a.device)
                stats[0] = (a.double() - b.double()).square().sum()
                stats[1] = b.double().square().sum()
                dist.all_reduce(stats)
                rel = (stats[0] / stats[1]).sqrt().item() if stats[1] else 0.0
                errors[name] = dict(
                    global_relative_l2_vs_fp32=rel,
                    bound=MXFP8_REL_L2,
                    pass_=rel < MXFP8_REL_L2,
                )
                errors[name]["pass"] = errors[name].pop("pass_")
        routes = audit_routes(iteration, source, lengths, rank, ep)
        assert actual[0].shape == actual[1].shape == (lengths[rank], args.hidden)
        assert actual[2].shape == (lengths[rank], 8)
        hashes = snapshot(
            actual, rank_dir / f"{label}.pt" if save and args.save_outputs else None
        )
        dist.barrier()
        reports.append(dict(label=label, errors=errors, routes=routes, hashes=hashes))
        (rank_dir / "checks.json").write_text(json.dumps(reports, indent=2) + "\n")
        if rank == 0:
            passed = all(e["pass"] for e in errors.values())
            print(
                f"{args.layout}/{args.routing}: {label} all nine pass={passed}",
                flush=True,
            )
        return hashes

    shift = 5 % ep or 1
    if args.sanitizer_smoke:
        # Numerics are qualified separately; instrument all recurring operations.
        rows = []
        for label, iteration, source, lengths in [("initial", first, data, counts)]:
            iteration.run()
            torch.cuda.synchronize()
            rows.append(
                dict(
                    label=label,
                    routes=audit_routes(iteration, source, lengths, rank, ep),
                )
            )
        changed_counts = counts[shift:] + counts[:shift]
        changed = data_for(changed_counts, 1)
        second = TrainingIteration(
            config,
            workspace,
            *local_inputs(changed, changed_counts, rank),
            weights,
            **options,
        )
        second.capture()
        second.run()
        torch.cuda.synchronize()
        rows.append(
            dict(
                label="changed-counts",
                routes=audit_routes(second, changed, changed_counts, rank, ep),
            )
        )
        first.run()
        torch.cuda.synchronize()
        rows.append(
            dict(
                label="earlier-graph",
                routes=audit_routes(first, data, counts, rank, ep),
            )
        )
        (rank_dir / "sanitizer-workload.json").write_text(
            json.dumps(
                dict(
                    status="PASS",
                    rank=rank,
                    ep=ep,
                    numerical_validation=False,
                    reports=rows,
                ),
                indent=2,
            )
            + "\n"
        )
        dist.barrier()
        dist.destroy_process_group()
        return

    expected = expected_for(data, counts)
    repeated = [
        check(first, expected, data, counts, f"fixed-{i}", save=True) for i in range(3)
    ]
    assert repeated[0] == repeated[1] == repeated[2]
    del expected
    torch.cuda.synchronize()
    # Keep only the original local inputs while checking a different count vector.
    del data
    gc.collect()
    changed_counts = counts[shift:] + counts[:shift]
    changed = data_for(changed_counts, 1)
    second = TrainingIteration(
        config,
        workspace,
        *local_inputs(changed, changed_counts, rank),
        weights,
        **options,
    )
    second.capture()
    expected = expected_for(changed, changed_counts)
    check(second, expected, changed, changed_counts, "changed-counts-and-inputs")
    del expected, changed
    gc.collect()
    data = data_for(counts, 0)
    expected = expected_for(data, counts)
    replayed = check(first, expected, data, counts, "earlier-graph")
    assert replayed == repeated[0]
    del expected, data
    gc.collect()
    updated = data_for(counts, 2)
    for dest, value in zip(
        (first.x, first.ids, first.scores, first.dy),
        local_inputs(updated, counts, rank),
        strict=True,
    ):
        dest.copy_(value)
    torch.cuda.synchronize()
    dist.barrier()
    expected = expected_for(updated, counts)
    check(first, expected, updated, counts, "same-shape-update")
    del expected, updated, second
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    # Reference allocations are excluded from reportable candidate peak memory.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    performance = benchmark(first, 3) if args.benchmark else None
    package = (
        Path(__file__).resolve().parents[1] / "flashinfer/experimental/cake_mok_bf16"
    )
    source_registry = json.loads((package / "sources.json").read_text())
    from flashinfer.experimental.cake_mok_bf16.jit import target_arch

    arch = target_arch(device)
    fused = [
        role + ("_mxfp8" if args.mxfp8 else "") + ("_clamped" if limit else "")
        for role in ("forward", "backward")
    ]
    passed = all(e["pass"] for row in reports for e in row["errors"].values())
    record = dict(
        status="PASS" if passed else "FAIL",
        rank=rank,
        ep=ep,
        layout=args.layout,
        routing=args.routing,
        num_local_tokens=counts[rank],
        nominal_tokens_per_rank=args.tokens,
        source_counts=counts,
        global_tokens=sum(counts),
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        topk=8,
        swiglu_limit=limit,
        mxfp8=args.mxfp8,
        target=arch,
        numerical_gate=gate,
        reports=reports,
        three_replays_bitwise_identical=True,
        earlier_graph_identical=True,
        performance=performance,
        fused_source_sha256={
            k: next(
                v["sha256"]
                for v in source_registry[k]["variants"]
                if arch in v["arches"]
            )
            for k in fused
        },
    )
    (rank_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    ranks = [None] * ep
    dist.all_gather_object(ranks, dict(rank=rank, status=record["status"]))
    if rank == 0:
        record["ranks"] = ranks
        record["reports"] = [
            dict(
                label=r["label"],
                errors={
                    k: {
                        a: b
                        for a, b in v.items()
                        if a.startswith("global_") or a == "pass"
                    }
                    for k, v in r["errors"].items()
                },
            )
            for r in reports
        ]
        (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    status=record["status"],
                    ep=ep,
                    layout=args.layout,
                    routing=args.routing,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()
    assert passed, (
        "Elementwise atol=1e-2 / rtol=1e-2 failed; inspect per-rank checks.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("fixed", "near", "strong", "empty"), default="fixed"
    )
    parser.add_argument(
        "--routing", choices=("uniform", "imbalanced"), default="uniform"
    )
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, choices=(256, 288), default=256)
    parser.add_argument("--swiglu-limit", type=float)
    parser.add_argument("--mxfp8", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument(
        "--sanitizer-smoke",
        action="store_true",
        help="Instrumentable graph/route checks; excludes reference and is not numerical acceptance",
    )
    run(parser.parse_args())
