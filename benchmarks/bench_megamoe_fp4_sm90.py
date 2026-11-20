"""SM90 MegaMoE CUDA-graph timing with local spawn or multi-host torchrun.

Tokens are per rank by default; ``--token-scope total`` requires exact division
by EP. Global row/expert seeds keep a fixed total workload identical across EP
partitions. Generated routing is globally balanced by default; it is not a
historical R405 fixture. JSON retains every rank sample; optional CSV reports both the median
of per-sample rank maxima and the maximum of per-rank medians.

``--reference-runner`` accepts a single-host canonical-weight source adapter
exporting ``_module(format, family, plan, buffer)`` and ``_BaselineBuffer``.
``--redundant-slots`` adds a single-host replica-plan arm, with planning and
remapping inside the timed graph. A routing trace reuses the same layer during
replay, so this arm measures resident replicas after the first plan.
"""

import argparse
import csv
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

from megamoe_sm90_utils import make_input, make_weights, resolve_tokens


def _shapes(args):
    """Resolve either named model presets or one complete customer shape."""
    values = (args.hidden, args.intermediate, args.experts, args.topk)
    if any(value is not None for value in values):
        if not all(value is not None for value in values) or args.shapes:
            raise ValueError(
                "provide all of hidden/intermediate/experts/topk, without shapes"
            )
        return {"custom": values}
    presets = {
        "flash": (4096, 2048, 256, 6),
        "mimo": (6144, 2048, 384, 8),
        "pro": (7168, 3072, 384, 6),
    }
    return {name: presets[name] for name in (args.shapes or ["flash"])}


def _global_digest(local, count):
    """Combine actual per-row or per-expert byte hashes in global ID order."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    ordered = sorted(item for partition in gathered for item in partition)
    assert [item[0] for item in ordered] == list(range(count))
    return hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode()
    ).hexdigest()


def _make_weights(rank, ep, experts, hidden, intermediate, fmt, seed):
    """Pack local experts and record a partition-independent model identity."""
    group = 16 if fmt == "nvfp4" else 32
    raw, local = make_weights(
        rank * (experts // ep), experts // ep, hidden, intermediate, group, seed, "cuda"
    )
    return raw, {
        "sha256": _global_digest(local, experts),
        "kind": "global-expert-ordered source FP4 and scale bytes",
        "scale_group": group,
        "global_scales": [1.0, 1.0],
        "seed": seed,
    }


def _make_input(m, rank, hidden, experts, topk, seed, routes, routing):
    """Quantize the same global rows and routing under every EP partition."""
    tensors, local = make_input(
        m, rank, hidden, experts, topk, seed, "cuda", routes, routing=routing
    )
    identity = {
        "sha256": _global_digest(local, m * dist.get_world_size()),
        "kind": "global-row-ordered quantized input, scales, routes and routing weights",
        "quantizer": "FP8 E4M3FN with FP32 scale 1/128 per 128 elements",
        "seed": seed,
        "fixture": "generated_global_row_seed",
        "routing": "trace" if routes is not None else routing,
        "routing_source": "external_trace" if routes is not None else "generated",
    }
    return (*tensors, identity)


def _write_csv(path, results, args):
    """Write one summary row per shape, token count, routing case and variant."""
    rows = []
    for result in results:
        for variant, median in result["median_us"].items():
            rows.append(
                {
                    "shape": result["shape"],
                    "format": result["format"],
                    "weight_storage": result["implementation"]["weight_storage"],
                    "weight_tile_bytes": result["implementation"]["weight_tile_bytes"],
                    "ep": args.ep,
                    "nodes": args.nodes,
                    "hidden": result["hidden"],
                    "intermediate": result["intermediate"],
                    "experts": result["experts"],
                    "topk": result["topk"],
                    "token_scope": args.token_scope,
                    "requested_tokens": result["tokens"]
                    if args.token_scope == "per-rank"
                    else result["total_tokens"],
                    "tokens_per_rank": result["tokens_per_rank"],
                    "total_tokens": result["total_tokens"],
                    "capacity": args.capacity,
                    "case": result["case"],
                    "routing": result["input_identity"]["routing"],
                    "routing_source": result["input_identity"]["routing_source"],
                    "variant": variant,
                    "graph_calls": args.repeats,
                    "warmup_graphs": args.warmup_graphs,
                    "rounds": args.rounds,
                    "median_max_rank_us": median,
                    "max_rank_median_us": result["max_rank_median_us"][variant],
                    "input_sha256": result["input_identity"]["sha256"],
                    "source_weight_sha256": result["weight_identity"]["sha256"],
                    "seed": args.seed,
                }
            )
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    layers = trace["layers"]
    if not layers or any(
        not isinstance(rows, torch.Tensor)
        or rows.ndim != 2
        or rows.shape[0] == 0
        or rows.shape[1] != topk
        or rows.dtype not in (torch.int32, torch.int64)
        or bool(((rows < -1) | (rows >= experts)).any())
        for rows in layers
    ):
        raise ValueError(
            "trace layers must contain nonempty integer [tokens, topk] routing"
        )
    return layers


def _worker(rank, args):
    """One rank: build every arm, capture it in a graph and time the interleaved replays."""
    from flashinfer import megamoe as api

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=args.ep,
        device_id=torch.device("cuda", local_rank),
        timeout=datetime.timedelta(minutes=15),
    )
    reference = None
    if args.reference_runner:
        spec = importlib.util.spec_from_file_location(
            "source_baseline", args.reference_runner
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    shapes = _shapes(args)
    transport = api.MegaMoENcclTransport(
        dist.group.WORLD,
        gin_contexts=1 if args.nodes > 1 else 0,
        gin_signals=2 * (args.nodes - 1),
    )
    cache_flush = (
        torch.empty(8 * 1024**3, dtype=torch.uint8, device="cuda")
        if args.cold_l2
        else None
    )
    results = []
    for shape_name, (h, ih, experts, topk) in shapes.items():
        buffer = api.MegaMoESymmBuffer(
            dist.group.WORLD,
            experts,
            args.capacity,
            topk,
            h,
            ih,
            transport=transport,
            num_nodes=args.nodes,
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
            raw, weight_identity = _make_weights(
                rank, args.ep, experts, h, ih, fmt, args.seed
            )
            if fmt == "nvfp4":
                w1, w2 = api.transform_nvfp4_weights_for_mega_moe(*raw)
            else:
                w1, w2 = api.transform_mxfp4_weights_for_mega_moe(
                    *raw, storage=args.weight_storage
                )
            del raw
            if slots is not None and fmt == "mxfp4":
                slots.set_resident_weights(w1, w2)
            slot_ratios = {m: [] for m in args.tokens}
            for m, case in itertools.product(args.tokens, cases):
                x, sf, idx, weights, input_identity = _make_input(
                    m,
                    rank,
                    h,
                    experts,
                    topk,
                    args.seed,
                    None if case is None else layers[case],
                    args.routing,
                )
                buffer.stage(x, sf, idx, weights)
                y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
                family = "split" if fmt == "nvfp4" and m > 256 else "fused"
                run_api = api.mxfp4_mega_moe if fmt == "mxfp4" else api.nvfp4_mega_moe
                run_new = functools.partial(run_api, y, w1, w2, buffer)

                run_new()
                torch.cuda.synchronize()
                assert torch.isfinite(y).all()
                runs = {"new": [run_new]}
                bb = None
                plan = None
                if reference is not None:
                    select = (
                        api.select_split_plan
                        if family == "split"
                        else api.select_fused_plan
                        if fmt == "nvfp4"
                        else api.select_mxfp4_fused_plan
                    )
                    plan = select(
                        buffer.num_sms, args.ep, experts, topk, h, ih, args.capacity, m
                    )
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
                    logical = idx.clone()

                    def prepare_slots():
                        """Restore the logical routing, then plan and remap the redundant expert slots for ``m`` tokens."""
                        slot_buffer.topk_idx[:m].copy_(logical)
                        slots.plan_and_remap(m)

                    run_slots = functools.partial(
                        api.mxfp4_mega_moe,
                        slot_y,
                        slots.l1_weights,
                        slots.l2_weights,
                        slot_buffer,
                        l1_global_scales=slots.l1_global_scales,
                        l2_global_scales=slots.l2_global_scales,
                        expert_slots=slots,
                    )
                    prepare_slots()
                    run_slots()
                    torch.cuda.synchronize()
                    assert torch.isfinite(slot_y).all()
                    runs["slots"] = [prepare_slots, run_slots]
                implementation = {
                    "api": run_api.__name__,
                    "weight_storage": args.weight_storage
                    if fmt == "mxfp4"
                    else "nvfp4",
                    "weight_shape": [list(w1.shape), list(w2.shape)],
                    "weight_tile_bytes": w1.shape[-1] if fmt == "mxfp4" else None,
                }
                captures = {
                    name: _capture(stages, args.repeats)
                    for name, stages in runs.items()
                }
                for _ in range(args.warmup_graphs):
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
                    max_rank_medians = {
                        name: max(
                            statistics.median(
                                sample["rank_us"][r]
                                for sample in samples
                                if sample["variant"] == name
                            )
                            for r in range(args.ep)
                        )
                        for name in runs
                    }
                    record = {
                        "shape": shape_name,
                        "hidden": h,
                        "intermediate": ih,
                        "experts": experts,
                        "topk": topk,
                        "tokens": m,
                        "tokens_per_rank": m,
                        "total_tokens": m * args.ep,
                        "token_scope": args.token_scope,
                        "input_identity": input_identity,
                        "weight_identity": weight_identity,
                        "format": fmt,
                        "family": family,
                        "case": args.routing if case is None else f"layer{case}",
                        "reference_plan": vars(plan) if plan is not None else None,
                        "median_us": medians,
                        "max_rank_median_us": max_rank_medians,
                        "kernel_median_us": kernel_medians,
                        "new_over_old": ratio,
                        "fixed_over_slots": slots_ratio,
                        "samples": samples,
                        "implementation": implementation,
                    }
                    results.append(record)
                    if args.csv:
                        _write_csv(args.csv, results, args)
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
                                "warmup_graphs": args.warmup_graphs,
                                "rounds": args.rounds,
                                "ep": args.ep,
                                "nodes": args.nodes,
                                "seed": args.seed,
                                "token_scope": args.token_scope,
                                "cold_l2": args.cold_l2,
                                "redundant_slots": args.redundant_slots,
                                "routing_trace": args.routing_trace,
                                "routing": "trace"
                                if args.routing_trace
                                else args.routing,
                                "gate": "NCCL GPU all-reduce before timing events inside graph",
                                "results": results,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                if "slots" in runs:
                    del (
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
    transport.close()
    dist.destroy_process_group()


def _parse_args(argv=None):
    """Validate shapes and resolve torchrun topology before allocating CUDA storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", choices=["flash", "mimo", "pro"])
    parser.add_argument("--tokens", nargs="+", type=int)
    parser.add_argument(
        "--token-scope", choices=["per-rank", "total"], default="per-rank"
    )
    parser.add_argument("--ep", type=int, choices=[4, 8, 16])
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--seed", type=int, default=7919)
    parser.add_argument(
        "--weight-storage", choices=["mxfp4", "mxfp4_compact", "fp8"], default="mxfp4"
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["nvfp4", "mxfp4"], default=["mxfp4"]
    )
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--capacity", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup-graphs", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reference-runner", type=str)
    parser.add_argument(
        "--redundant-slots",
        type=int,
        default=0,
        help="MXFP4 only: also time the kernel with this many redundant expert "
        "slots per rank under the in-kernel planner",
    )
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument(
        "--routing", choices=["random", "balanced"], default="balanced"
    )
    routing.add_argument(
        "--routing-trace", type=str, help="recorded logical routing to replay"
    )
    parser.add_argument(
        "--trace-layers", type=int, default=4, help="traced layers to replay"
    )
    parser.add_argument("--output", type=str, default="megamoe_fp4_benchmark.json")
    parser.add_argument(
        "--csv", type=str, help="summary CSV; full rank samples remain in JSON"
    )
    args = parser.parse_args(argv)
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    world = int(os.environ["WORLD_SIZE"]) if launched else (args.ep or 8)
    if args.ep is not None and args.ep != world:
        parser.error("--ep must match torchrun WORLD_SIZE")
    args.ep = world
    local = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    if local < 1:
        parser.error("LOCAL_WORLD_SIZE must be positive")
    args.nodes = args.nodes if args.nodes is not None else world // local
    if world not in (4, 8, 16) or args.nodes < 1 or world % args.nodes:
        parser.error("EP must be 4, 8 or 16 and divide evenly across nodes")
    if launched and (local < 1 or world != local * args.nodes):
        parser.error("--nodes must match torchrun WORLD_SIZE / LOCAL_WORLD_SIZE")
    if not launched and args.nodes != 1:
        parser.error("use torchrun on each host for a multi-node benchmark")
    if args.seed < 0 or args.capacity < 1:
        parser.error("seed must be nonnegative and capacity positive")
    if args.tokens is None:
        args.tokens = (
            [1, 4, 8, 16, 32, 64]
            if args.token_scope == "per-rank"
            else [16, 64, 128, 256, 512, 1024]
        )
    try:
        args.tokens = resolve_tokens(
            args.tokens, args.token_scope, world, args.capacity
        )
        for h, ih, experts, topk in _shapes(args).values():
            if h <= 0 or h % 256 or ih <= 0 or ih % 128:
                raise ValueError(
                    "hidden must be a positive multiple of 256; intermediate of 128"
                )
            if experts < 1 or experts % world or not 0 < topk <= min(32, experts):
                raise ValueError(
                    "experts must divide by EP; topk must be in [1, min(32, experts)]"
                )
    except ValueError as error:
        parser.error(str(error))
    if "nvfp4" in args.formats and (world not in (4, 8) or args.nodes != 1):
        parser.error("NVFP4 currently requires four or eight ranks on one node")
    if args.redundant_slots < 0:
        parser.error("redundant slots must be nonnegative")
    if (args.reference_runner or args.redundant_slots) and (
        args.nodes != 1 or args.weight_storage != "mxfp4"
    ):
        parser.error(
            "source-baseline and replica-slot arms require one node and canonical weights"
        )
    if args.repeats < 1 or args.warmup_graphs < 1 or args.rounds < 1:
        parser.error("repeats, warmup graphs and rounds must be positive")
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
    return args


def _run(args):
    """Use torchrun's rank or spawn local workers for a single-node run."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _worker(int(os.environ["RANK"]), args)
        return
    if torch.cuda.device_count() < args.ep:
        raise ValueError(f"needs {args.ep} local GPUs, or launch with torchrun")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29650")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.ep)


if __name__ == "__main__":
    _run(_parse_args())
