"""Four/eight-rank SM90 MegaMoE CUDA-graph timing with optional source baselines.

The reference runner exports _module(format, family, plan, buffer) and
_BaselineBuffer(buffer, offsets), as in the source-regression runner supplied
with the two pinned source checkouts. Without it, this benchmarks FlashInfer.
Each graph queues an unmeasured NCCL GPU gate before the timing events. Raw
rank vectors are saved; per-trial rank MAX is reduced before medians/ratios.

``--redundant-slots S`` adds the MXFP4 replica-plan arm: the same tokens through
``S`` redundant expert slots per rank, planned by the in-kernel planner on every
call (plan and remap, then the kernel with the in-kernel weight copy), timed as
a whole and as the kernel alone. ``--routing-trace`` replays recorded routing
(a dict with ``layers``, a list of ``[tokens, topk]`` logical-id tensors, plus
``num_experts`` and ``num_topk``) instead of random routing, one case per layer.
Every replay of a case re-plans the same routing, so after the first plan the
slots already hold their experts and the timed kernel copies no weights: the
replica-plan arm is measured in the steady state of one chunk.
"""

import argparse
import datetime
import functools
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def _capture(stages, repeats):
    """Graph of `repeats` x `stages`; with one repeat an event follows every
    stage, so the stages can be timed apart."""
    gate = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_events = len(stages) + 1 if repeats == 1 else 2
    events = [
        torch.cuda.Event(enable_timing=True, external=True) for _ in range(num_events)
    ]
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        dist.all_reduce(gate)
        events[0].record()
        for _ in range(repeats):
            for i, stage in enumerate(stages):
                stage()
                if repeats == 1:
                    events[i + 1].record()
        if repeats != 1:
            events[-1].record()
    return graph, events, gate


def _load_trace(path, experts, topk):
    """Per-layer logical routing of a recorded trace, checked against the model shape."""
    trace = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if int(trace["num_experts"]) != experts or int(trace["num_topk"]) != topk:
        raise ValueError(
            f"trace is for {trace['num_experts']} experts / top-{trace['num_topk']}"
        )
    return trace["layers"]


def _trace_slice(rows, m, rank):
    """`m` consecutive tokens of one traced layer for `rank`, wrapping around."""
    total = rows.shape[0]
    start = (rank * m) % total
    return rows[(torch.arange(m) + start) % total].to(torch.int64).cuda()


def _worker(rank, args):
    """One rank: build every arm, capture it in a graph and time the interleaved replays."""
    from flashinfer import megamoe as api
    from flashinfer.jit import env
    from flashinfer.jit.megamoe import (
        gen_megamoe_sm90_nvfp4_module,
        gen_megamoe_sm90_nvfp4_split_module,
        gen_megamoe_sm90_mxfp4_module,
    )

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=args.ep,
        device_id=torch.device("cuda", rank),
        timeout=datetime.timedelta(minutes=15),
    )
    reference = None
    if args.reference_runner:
        spec = importlib.util.spec_from_file_location(
            "source_baseline", args.reference_runner
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    shapes = {
        "flash": (4096, 2048, 256, 6),
        "mimo": (6144, 2048, 384, 8),
        "pro": (7168, 3072, 384, 6),
    }
    cache_flush = (
        torch.empty(8 * 1024**3, dtype=torch.uint8, device="cuda")
        if args.cold_l2
        else None
    )
    results = []
    for shape_name in args.shapes:
        h, ih, experts, topk = shapes[shape_name]
        buffer = api.MegaMoESymmBuffer(
            dist.group.WORLD, experts, args.capacity, topk, h, ih
        )
        slots = slot_buffer = None
        if args.redundant_slots:
            slot_buffer = api.MegaMoESymmBuffer(
                dist.group.WORLD,
                experts + args.ep * args.redundant_slots,
                args.capacity,
                topk,
                h,
                ih,
            )
            slots = api.MegaMoEExpertSlots(slot_buffer, experts, args.redundant_slots)
        epr = experts // args.ep
        generator = torch.Generator(device="cuda").manual_seed(7919 + rank)
        layers = (
            _load_trace(args.routing_trace, experts, topk)
            if args.routing_trace
            else None
        )
        if layers is not None:
            step = max(1, len(layers) // args.trace_layers)
            cases = list(range(0, len(layers), step))[: args.trace_layers]
        else:
            cases = [None]
        for fmt in args.formats:
            gran = 16 if fmt == "nvfp4" else 32
            # Identical packed weights on both sides; packing is outside timing.
            raw = []
            for n, k in ((ih * 2, h), (h, ih)):
                p = torch.randint(
                    0,
                    256,
                    (epr, n, k // 2),
                    device="cuda",
                    dtype=torch.uint8,
                    generator=generator,
                )
                sf = torch.full(
                    (epr, n, k // gran),
                    16 if gran == 16 else 122,
                    dtype=torch.uint8,
                    device="cuda",
                )
                raw.append((p, sf))
            transform = (
                api.transform_nvfp4_weights_for_mega_moe
                if fmt == "nvfp4"
                else api.transform_mxfp4_weights_for_mega_moe
            )
            w1, w2 = transform(*raw)
            del raw
            if slots is not None and fmt == "mxfp4":
                slots.set_resident_weights(w1, w2)
            slot_ratios = {m: [] for m in args.tokens}
            for m, case in itertools.product(args.tokens, cases):
                x = (torch.randn(m, h, device="cuda", generator=generator) * 32).to(
                    torch.float8_e4m3fn
                )
                sf = torch.full((m, h // 128), 1 / 128, device="cuda")
                if case is not None:
                    idx = _trace_slice(layers[case], m, rank)
                else:
                    idx = (
                        torch.randn(m, experts, device="cuda", generator=generator)
                        .topk(topk, dim=-1)
                        .indices
                    )
                weights = torch.softmax(
                    torch.randn(m, topk, device="cuda", generator=generator), dim=-1
                )
                buffer.stage(x, sf, idx, weights)
                y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
                family = "split" if fmt == "nvfp4" and m > 256 else "fused"
                select = (
                    api.select_split_plan
                    if family == "split"
                    else (
                        api.select_fused_plan
                        if fmt == "nvfp4"
                        else api.select_mxfp4_fused_plan
                    )
                )
                plan = select(
                    buffer.num_sms, args.ep, experts, topk, h, ih, args.capacity, m
                )
                if fmt == "mxfp4":
                    spec = gen_megamoe_sm90_mxfp4_module(
                        plan,
                        buffer.num_sms,
                        args.ep,
                        args.capacity,
                        experts,
                        topk,
                        h,
                        ih,
                        10.0,
                        True,
                    )
                elif family == "split":
                    spec = gen_megamoe_sm90_nvfp4_split_module(
                        plan,
                        buffer.num_sms,
                        args.ep,
                        args.capacity,
                        experts,
                        topk,
                        h,
                        ih,
                        10.0,
                        True,
                    )
                else:
                    spec = gen_megamoe_sm90_nvfp4_module(
                        plan,
                        buffer.num_sms,
                        args.capacity,
                        experts,
                        topk,
                        h,
                        ih,
                        10.0,
                        True,
                        num_ranks=args.ep,
                    )
                module = spec.build_and_load()
                func = getattr(module, f"megamoe_sm90_{fmt}_{family}")

                run_new = functools.partial(
                    func,
                    y,
                    buffer.buffer,
                    buffer.handle.buffer_ptrs,
                    rank,
                    w1,
                    w2,
                    buffer._offsets,
                    m,
                    None,
                    None,
                    None,
                )

                run_new()
                torch.cuda.synchronize()
                assert torch.isfinite(y).all()
                runs = {"new": [run_new]}
                bb = None
                if reference is not None:
                    old = reference._module(fmt, family, plan, buffer)
                    offsets = (
                        list(old.baseline_layout())
                        if fmt == "mxfp4"
                        else list(old.megamoe_sm90_nvfp4_symm_buffer_layout())
                        if family == "fused"
                        else buffer._offsets
                    )
                    bb = reference._BaselineBuffer(buffer, offsets)
                    bb.stage(buffer, m)
                    old_y = torch.empty_like(y)
                    old_func = getattr(old, f"megamoe_sm90_{fmt}_{family}")

                    run_old = functools.partial(
                        old_func,
                        old_y,
                        bb.storage,
                        bb.handle.buffer_ptrs,
                        rank,
                        w1,
                        w2,
                        bb.offsets,
                        m,
                        None,
                        None,
                        None,
                    )

                    run_old()
                    torch.cuda.synchronize()
                    assert torch.isfinite(old_y).all()
                    runs["old"] = [run_old]
                if slots is not None and fmt == "mxfp4":
                    # The replica-plan arm: logical ids restored, planned and
                    # remapped in one launch, then the kernel with the in-kernel
                    # weight copy, all inside the graph.
                    slot_buffer.stage(x, sf, idx, weights)
                    slot_y = torch.empty_like(y)
                    slot_plan = api.select_mxfp4_fused_plan(
                        slot_buffer.num_sms,
                        args.ep,
                        slot_buffer.num_experts,
                        topk,
                        h,
                        ih,
                        args.capacity,
                        m,
                    )
                    slot_module = gen_megamoe_sm90_mxfp4_module(
                        slot_plan,
                        slot_buffer.num_sms,
                        args.ep,
                        args.capacity,
                        slot_buffer.num_experts,
                        topk,
                        h,
                        ih,
                        10.0,
                        True,
                        True,
                    ).build_and_load()
                    logical = idx.clone()

                    def prepare_slots():
                        slot_buffer.topk_idx[:m].copy_(logical)
                        slots.plan_and_remap(m)

                    run_slots = functools.partial(
                        slot_module.megamoe_sm90_mxfp4_fused_in_kernel_copy,
                        slot_y,
                        slot_buffer.buffer,
                        slot_buffer.handle.buffer_ptrs,
                        rank,
                        slots.l1_weights,
                        slots.l2_weights,
                        slot_buffer._offsets,
                        m,
                        None,
                        slots.l1_global_scales,
                        slots.l2_global_scales,
                        *slots._in_kernel_copy_args(),
                    )
                    prepare_slots()
                    run_slots()
                    torch.cuda.synchronize()
                    assert torch.isfinite(slot_y).all()
                    runs["slots"] = [prepare_slots, run_slots]
                # Freeze binary identity before any timed graph replay.
                if rank == 0:
                    binary = spec.jit_library_path
                    proof = {
                        "name": spec.name,
                        "library": str(binary),
                        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "sources": [str(p) for p in spec.sources],
                        "include_root": str(env.FLASHINFER_INCLUDE_DIR.resolve()),
                    }
                    if reference:
                        old_name = spec.name + "_original_github"
                        old_library = (
                            env.FLASHINFER_JIT_DIR / old_name / (old_name + ".so")
                        )
                        proof["baseline"] = {
                            "name": old_name,
                            "library": str(old_library),
                            "sha256": hashlib.sha256(
                                old_library.read_bytes()
                            ).hexdigest(),
                        }
                    print("JIT_READY_PROOF " + json.dumps(proof), flush=True)
                captures = {
                    name: _capture(stages, args.repeats)
                    for name, stages in runs.items()
                }
                for _ in range(3):
                    for graph, events, _ in captures.values():
                        graph.replay()
                        events[-1].synchronize()
                samples = []
                if reference:
                    order = ("old", "new", "new", "old")
                elif "slots" in runs:
                    order = ("new", "slots", "slots", "new")
                else:
                    order = ("new",)
                for round_idx in range(args.rounds):
                    for position, name in enumerate(order):
                        graph, events, _ = captures[name]
                        if cache_flush is not None:
                            cache_flush.zero_()
                        graph.replay()
                        events[-1].synchronize()
                        # With an event per stage the last stage is the kernel
                        # alone (the plan and remap of the replica-plan arm
                        # precede it); with two events the single interval
                        # spans every repeat of the kernel.
                        stage_us = [
                            events[i].elapsed_time(events[i + 1]) * 1000
                            for i in range(len(events) - 1)
                        ]
                        kernel_us = (
                            stage_us[-1] / args.repeats
                            if len(events) == 2
                            else stage_us[-1]
                        )
                        value = torch.tensor(
                            [sum(stage_us) / args.repeats, kernel_us],
                            dtype=torch.float64,
                            device="cuda",
                        )
                        gathered = [torch.empty_like(value) for _ in range(args.ep)]
                        dist.all_gather(gathered, value)
                        rank_us = [v[0].item() for v in gathered]
                        if rank == 0:
                            samples.append(
                                {
                                    "round": round_idx,
                                    "position": position,
                                    "variant": name,
                                    "rank_us": rank_us,
                                    "max_us": max(rank_us),
                                    "kernel_max_us": max(v[1].item() for v in gathered),
                                }
                            )
                if rank == 0:
                    medians = {
                        name: statistics.median(
                            s["max_us"] for s in samples if s["variant"] == name
                        )
                        for name in runs
                    }
                    kernel_medians = {
                        name: statistics.median(
                            s["kernel_max_us"] for s in samples if s["variant"] == name
                        )
                        for name in runs
                    }
                    ratio = medians["new"] / medians["old"] if reference else None
                    slots_ratio = (
                        {
                            "total": medians["new"] / medians["slots"],
                            "kernel": medians["new"] / kernel_medians["slots"],
                        }
                        if "slots" in runs
                        else None
                    )
                    if slots_ratio is not None:
                        slot_ratios[m].append(slots_ratio)
                    record = {
                        "shape": shape_name,
                        "hidden": h,
                        "intermediate": ih,
                        "experts": experts,
                        "topk": topk,
                        "tokens": m,
                        "format": fmt,
                        "family": family,
                        "case": "random" if case is None else f"layer{case}",
                        "plan": vars(plan),
                        "median_us": medians,
                        "kernel_median_us": kernel_medians,
                        "new_over_old": ratio,
                        "fixed_over_slots": slots_ratio,
                        "samples": samples,
                        "jit": proof,
                    }
                    results.append(record)
                    print(
                        f"BENCH {shape_name} {fmt} {family} M={m} {record['case']} "
                        f"{medians} kernel={kernel_medians} ratio={ratio} "
                        f"fixed_over_slots={slots_ratio}",
                        flush=True,
                    )
                    Path(args.output).write_text(
                        json.dumps(
                            {
                                "gpu": torch.cuda.get_device_name(),
                                "torch": torch.__version__,
                                "repeats": args.repeats,
                                "rounds": args.rounds,
                                "ep": args.ep,
                                "cold_l2": args.cold_l2,
                                "redundant_slots": args.redundant_slots,
                                "routing_trace": args.routing_trace,
                                "gate": "NCCL GPU all-reduce before timing events inside graph",
                                "results": results,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                if "slots" in runs:
                    del (
                        slot_module,
                        slot_plan,
                        logical,
                        prepare_slots,
                        run_slots,
                        slot_y,
                    )
                del captures, runs, y, graph, events, run_new
                if bb is not None:
                    del bb, old_y, run_old
                gc.collect()
                torch.cuda.synchronize()
            if rank == 0:
                for m, ratios in slot_ratios.items():
                    if ratios:
                        geomean = {
                            key: math.exp(
                                sum(math.log(r[key]) for r in ratios) / len(ratios)
                            )
                            for key in ("total", "kernel")
                        }
                        print(
                            f"GEOMEAN {shape_name} {fmt} M={m} cases={len(ratios)} "
                            f"fixed_over_slots={geomean}",
                            flush=True,
                        )
        if slots is not None:
            slots.destroy()
            slot_buffer.destroy()
        buffer.destroy()
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes", nargs="+", choices=["flash", "mimo", "pro"], default=["flash"]
    )
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[1, 24, 32, 64, 96, 384]
    )
    parser.add_argument("--ep", type=int, choices=[4, 8], default=8)
    parser.add_argument(
        "--formats", nargs="+", choices=["nvfp4", "mxfp4"], default=["nvfp4", "mxfp4"]
    )
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--capacity", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reference-runner", type=str)
    parser.add_argument(
        "--redundant-slots",
        type=int,
        default=0,
        help="MXFP4 only: also time the kernel with this many redundant expert "
        "slots per rank under the in-kernel planner",
    )
    parser.add_argument(
        "--routing-trace", type=str, help="recorded logical routing to replay"
    )
    parser.add_argument(
        "--trace-layers", type=int, default=4, help="traced layers to replay"
    )
    parser.add_argument("--output", type=str, default="megamoe_fp4_benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 1:
        parser.error("repeats and rounds must be positive")
    if args.trace_layers < 1:
        parser.error("trace layers must be positive")
    if args.cold_l2 and args.repeats != 1:
        parser.error("cold L2 timing requires --repeats 1")
    if args.redundant_slots and args.reference_runner:
        parser.error("--redundant-slots cannot be combined with --reference-runner")
    if args.redundant_slots and args.repeats != 1:
        parser.error(
            "--redundant-slots times the plan and the kernel apart; use --repeats 1"
        )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29650")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.ep)
