"""Four/eight-rank SM90 MegaMoE CUDA-graph timing with optional source baselines.

The reference runner exports _module(format, family, plan, buffer) and
_BaselineBuffer(buffer, offsets), as in the source-regression runner supplied
with the two pinned source checkouts. Without it, this benchmarks FlashInfer.
Each graph queues an unmeasured NCCL GPU gate before the timing events. Raw
rank vectors are saved; per-trial rank MAX is reduced before medians/ratios.
"""

import argparse
import datetime
import functools
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def _capture(run, repeats):
    gate = torch.zeros(1, dtype=torch.int32, device="cuda")
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        dist.all_reduce(gate)
        start.record()
        for _ in range(repeats):
            run()
        end.record()
    return graph, start, end, gate


def _worker(rank, args):
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
        epr = experts // args.ep
        generator = torch.Generator(device="cuda").manual_seed(7919 + rank)
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
            for m in args.tokens:
                x = (torch.randn(m, h, device="cuda", generator=generator) * 32).to(
                    torch.float8_e4m3fn
                )
                sf = torch.full((m, h // 128), 1 / 128, device="cuda")
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
                runs = {"new": run_new}
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
                    runs["old"] = run_old
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
                    name: _capture(run, args.repeats) for name, run in runs.items()
                }
                for _ in range(3):
                    for graph, _, end, _ in captures.values():
                        graph.replay()
                        end.synchronize()
                samples = []
                order = ("old", "new", "new", "old") if reference else ("new",)
                for round_idx in range(args.rounds):
                    for position, name in enumerate(order):
                        graph, start, end, _ = captures[name]
                        if cache_flush is not None:
                            cache_flush.zero_()
                        graph.replay()
                        end.synchronize()
                        us = start.elapsed_time(end) * 1000 / args.repeats
                        value = torch.tensor([us], dtype=torch.float64, device="cuda")
                        gathered = [torch.empty_like(value) for _ in range(args.ep)]
                        dist.all_gather(gathered, value)
                        rank_us = [v.item() for v in gathered]
                        if rank == 0:
                            samples.append(
                                {
                                    "round": round_idx,
                                    "position": position,
                                    "variant": name,
                                    "rank_us": rank_us,
                                    "max_us": max(rank_us),
                                }
                            )
                if rank == 0:
                    medians = {
                        name: statistics.median(
                            s["max_us"] for s in samples if s["variant"] == name
                        )
                        for name in runs
                    }
                    ratio = medians["new"] / medians["old"] if reference else None
                    record = {
                        "shape": shape_name,
                        "hidden": h,
                        "intermediate": ih,
                        "experts": experts,
                        "topk": topk,
                        "tokens": m,
                        "format": fmt,
                        "family": family,
                        "plan": vars(plan),
                        "median_us": medians,
                        "new_over_old": ratio,
                        "samples": samples,
                        "jit": proof,
                    }
                    results.append(record)
                    print(
                        f"BENCH {shape_name} {fmt} {family} M={m} {medians} ratio={ratio}",
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
                                "gate": "NCCL GPU all-reduce before timing events inside graph",
                                "results": results,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                del captures, runs, y, graph, start, end, run_new
                if bb is not None:
                    del bb, old_y, run_old
                gc.collect()
                torch.cuda.synchronize()
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
    parser.add_argument("--output", type=str, default="megamoe_fp4_benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 1:
        parser.error("repeats and rounds must be positive")
    if args.cold_l2 and args.repeats != 1:
        parser.error("cold L2 timing requires --repeats 1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29650")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.ep)
