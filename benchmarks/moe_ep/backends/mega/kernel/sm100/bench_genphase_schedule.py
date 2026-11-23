# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare explicit, heuristic and cached GenPhase configs with graph replay.

Run each policy on every EP rank, for example from the repository root::

    torchrun --standalone --nproc_per_node=2 \
      benchmarks/moe_ep/backends/mega/kernel/sm100/bench_genphase_schedule.py \
      --policy heuristic --tile-n 128 --cluster-m 4 --output results/heuristic

Repeat with --policy original, then run both policies in reverse order to
control drift. --policy explicit --schedule-hint N overrides both defaults.
--auto-config --live-tokens N selects geometry from the host workload bucket.
--policy heuristic disables the cache; --policy auto requires an exact cache
hit from a prior tuning run and --weight-placement early (the public backend).
The defaults are H4096/I1024/E256/top-k8, capacity1024, MXFP8/BF16. Input
staging, quantization and host submission are outside the timed graphs.
CUDA events measure the full dispatch/main/reducer replay, with the maximum
rank time per sample. This is not a serving benchmark or an autotuner.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[6]


def collective_action(action, gather):
    """Propagate rank-local action failures so every EP rank exits the phase."""
    error = None
    try:
        action()
    except Exception as exc:
        error = repr(exc)
    errors = gather(error)
    if any(errors):
        raise RuntimeError(errors)


def atomic_json(path, data):
    """Publish a complete report by replacing its temporary file atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, default=SOURCE_ROOT)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--hidden", type=int, default=4096)
p.add_argument("--intermediate", type=int, default=1024)
p.add_argument("--experts", type=int, default=256)
p.add_argument("--topk", type=int, default=8)
p.add_argument("--capacity", type=int, default=1024)
p.add_argument("--active", default="0,17,65,1024")
p.add_argument("--calls", type=int, default=8)
p.add_argument("--samples", type=int, default=64)
p.add_argument("--schedule-hint", type=int)
p.add_argument(
    "--policy",
    choices=("original", "heuristic", "explicit", "auto"),
    default="heuristic",
)
p.add_argument("--tile-n", type=int, choices=(64, 128), default=128)
p.add_argument("--cluster-m", type=int, choices=(2, 4), default=4)
p.add_argument(
    "--auto-config", action="store_true", help="Leave tile and cluster to the backend"
)
p.add_argument(
    "--live-tokens",
    type=int,
    help="Host workload bucket used for configuration selection",
)
p.add_argument("--weight-placement", choices=("early", "late", "both"), default="both")
args = p.parse_args()
try:
    active_counts = tuple(int(value) for value in args.active.split(","))
except ValueError:
    p.error("--active must be a comma-separated list of integers")
if any(value < 0 or value > args.capacity for value in active_counts):
    p.error("--active values must be between zero and --capacity")
if args.policy == "heuristic":
    os.environ["FLASHINFER_MOE_EP_KNOB_CACHE"] = "off"
if args.policy == "auto":
    assert args.auto_config and args.live_tokens is not None
    assert args.weight_placement == "early", "Public backend uses early routing weights"

if args.policy == "explicit":
    assert args.schedule_hint is not None and args.schedule_hint > 0
else:
    assert args.schedule_hint is None, "--schedule-hint requires --policy explicit"
assert args.calls > 0 and args.samples > 0
policy = {
    "original": ("grouped", None),
    "heuristic": None,
    "auto": None,
    "explicit": ("grouped", args.schedule_hint),
}[args.policy]
assert (args.tile_n, args.cluster_m) in ((128, 4), (64, 4), (64, 2))
source = args.source.resolve()
sys.path.insert(0, str(source))
import torch
import torch.distributed as dist
from cuda.bindings import driver as cuda
import flashinfer
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

assert Path(flashinfer.__file__).resolve().is_relative_to(source)
rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
assert world in (1, 2, 4) and args.experts % world == 0
bootstrap = BootstrapConfig(rank=rank, world_size=world)
ensure_moe_ep_cuda_device(bootstrap)
assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
backend = create_mega_kernel(
    Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.topk,
        kernel_variant="genphase",
        mma_tiler_mnk=None if args.auto_config else (256, args.tile_n, 128),
        cluster_shape_mn=None if args.auto_config else (args.cluster_m, 1),
        schedule_policy=policy,
    )
)
backend.bind_ep_bootstrap(bootstrap)
runtime = bootstrap_moe_ep_runtime(bootstrap, backend.runtime_requirements(bootstrap))
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
torch.cuda.set_stream(stream)


def gather_errors(error):
    """Collect each EP rank's error before deciding whether to continue."""
    errors = [None] * world
    dist.all_gather_object(errors, error)
    return errors


def barrier():
    """Drain the benchmark stream before synchronizing EP ranks."""
    stream.synchronize()
    dist.barrier()


def assert_all(condition, context):
    """Fail on every EP rank if any rank's local correctness check fails."""
    failed = torch.tensor(not condition, device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    assert failed.item() == 0, context


def digest(tensor):
    """Hash contiguous tensor bytes for exact output comparisons between runs."""
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def save_report():
    """Write per-rank progress reports and propagate filesystem failures."""
    collective_action(
        lambda: atomic_json(args.output / f"rank-{rank}.json", report), gather_errors
    )


args.output.mkdir(parents=True, exist_ok=True)
report = dict(
    status="running",
    source=str(source),
    commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip(),
    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    hardware=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
    rank=rank,
    world=world,
    torch=torch.__version__,
    cuda=torch.version.cuda,
    geometry={
        k: getattr(args, k)
        for k in ("hidden", "intermediate", "experts", "topk", "capacity")
    },
    scope=f"EP{world} pre-staged MXFP8/BF16 native GenPhase dispatch+FC1/FC2+reducer graph replay. Native mirrored performance experiment after separate qualification. Excludes staging, quantization and host submission; no serving E2E claim.",
    correctness="Sampled independent native reference for dense GEMMs, then full-capacity ordered FP32 route oracle from dense native terms; exact bytes before/after replay. Cross-source snapshots require a separate comparison.",
    calls=args.calls,
    samples=args.samples,
    policy=args.policy,
    schedule_hint=args.schedule_hint,
    requested_tile_n=None if args.auto_config else args.tile_n,
    requested_cluster_m=None if args.auto_config else args.cluster_m,
    native_kernels_per_call=3,
    cases=[],
)
ws = graph = None
try:
    gen = torch.Generator(device="cuda").manual_seed(9187 + rank)
    e = args.experts // world
    w13 = (
        torch.randn(e, 2 * args.intermediate, args.hidden, generator=gen, device="cuda")
        / args.hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(e, args.hidden, args.intermediate, generator=gen, device="cuda")
        / args.intermediate**0.5
    ).bfloat16()
    x = torch.randn(args.capacity, args.hidden, generator=gen, device="cuda").bfloat16()
    weights = pkg.preprocess_genphase_weights(
        w13, w2, quant_kind="mxfp8_e4m3", intermediate_size=args.intermediate
    )
    qx, sx = pkg.quantize_genphase_mxfp8(x, torch.float8_e4m3fn)
    ids = torch.stack(
        [
            torch.randperm(args.experts, generator=gen, device="cuda")[: args.topk]
            for _ in range(args.capacity)
        ]
    ).to(torch.int32)
    scores = torch.linspace(
        0.125, 0.875, args.capacity * args.topk, device="cuda"
    ).reshape_as(ids)
    report["input_sha256"] = {
        name: digest(value)
        for name, value in dict(w13=w13, w2=w2, x=x, ids=ids, scores=scores).items()
    }
    del w13, w2, x
    save_report()
    for early in (
        (True, False)
        if args.weight_placement == "both"
        else (args.weight_placement == "early",)
    ):
        fleet = FleetParams(
            num_experts=args.experts,
            max_tokens_per_rank=args.capacity,
            token_hidden_size=args.hidden,
            tuning_live_tokens=args.live_tokens,
        )
        cfg = dataclasses.replace(
            backend._genphase_config(fleet), apply_routing_weights_before_fc2=early
        )
        report["resolved_schedule_policy"] = cfg.schedule_policy
        report["resolved_tile"] = cfg.resolved_mma_tiler_mnk
        report["resolved_cluster"] = cfg.cluster_shape_mn
        report["selection_live_tokens"] = args.live_tokens
        if args.policy == "auto":
            selected, selection_source = pkg.resolve_genphase_knobs(
                backend._genphase_base_config(fleet),
                sm_count=torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count,
                live_tokens=args.live_tokens,
            )
            assert selection_source == "cache", (
                "auto benchmark requires a previously tuned exact cache entry"
            )
            assert all(getattr(cfg, key) == value for key, value in selected.items())
            report["selection_source"] = selection_source
        elif args.auto_config:
            report["selection_source"] = "heuristic"

        ws = pkg.Sm100GenphaseSymmBuffer(cfg)
        plan = ws.kernel._mainloop
        report.setdefault("stage_plans", []).append(
            dict(
                early=early,
                weight_stages=plan.num_weight_ab_stages,
                token_stages=plan.num_token_ab_stages,
                asymmetric=plan.uses_asymmetric_ab_stages,
                mainloop_budget_bytes=plan.mainloop_smem_budget_bytes,
                accumulator_stages=plan.num_accumulator_pipeline_stages,
            )
        )
        ws.x[: args.capacity].copy_(qx)
        ws.x_sf[: args.capacity, : sx.shape[1]].copy_(sx)
        ws.topk_idx[: args.capacity].copy_(ids)
        ws.topk_weights[: args.capacity].copy_(scores)
        thunk = pkg.sm100_genphase_mega_launch_thunk(*weights, ws)
        thunk()
        barrier()
        names = list(ws._compiled.kernel_info)
        assert_all(
            len(names) == 3
            and sum("GenphaseTokenComm" in n for n in names) == 1
            and sum("BlockScaledSwapAbGenphaseMoeKernel" in n for n in names) == 1
            and sum("_reduce_bf16" in n for n in names) == 1,
            ("native kernel roles", names),
        )
        indices, expected = pkg.sampled_reference(
            ws, *weights, args.capacity, sample_size=32
        )
        error = pkg.output_error(ws.output_activation, indices, expected)
        assert_all(
            bool(torch.isfinite(torch.tensor(error))) and error < 0.02,
            ("dense reference", early, error),
        )
        terms = (
            ws._pre_reduced_activation.view(torch.bfloat16)[: args.capacity]
            .float()
            .clone()
        )
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.calls):
                thunk()
        status, _, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()), 0)
        assert_all(
            status == cuda.CUresult.CUDA_SUCCESS and count == 3 * args.calls,
            ("graph nodes", status, count),
        )
        for active in active_counts:
            for holes in (False, True):
                keep = torch.zeros(args.capacity, device="cuda", dtype=torch.bool)
                keep[
                    torch.arange(active, device="cuda")
                    * args.capacity
                    // max(active, 1)
                ] = True
                valid = keep[:, None].expand_as(ids).clone()
                if holes:
                    valid[:, 1::3] = False
                ws.topk_idx[: args.capacity].copy_(ids).masked_fill_(~valid, -1)
                ws.topk_weights[: args.capacity].copy_(scores).masked_fill_(
                    ~valid, float("nan")
                )
                control = None
                for k in range(args.topk):
                    value = torch.where(valid[:, k, None], terms[:, k], 0.0)
                    weight = (
                        torch.where(valid[:, k, None], scores[:, k, None], 0.0)
                        if not early
                        else torch.ones_like(value)
                    )
                    if control is None:
                        control = (
                            (value.double() * weight.double()).float()
                            if not early
                            else value
                        )
                    else:
                        control = (
                            control.double() + value.double() * weight.double()
                        ).float()
                expected = control.bfloat16()
                ws._pre_reduced_activation.fill_(255)
                ws.output_activation.fill_(float("nan"))
                barrier()
                for _ in range(16):
                    graph.replay()
                barrier()
                assert_all(
                    torch.equal(
                        ws.output_activation[: args.capacity].view(torch.uint8),
                        expected.view(torch.uint8),
                    ),
                    ("pre timing", early, active, holes),
                )
                events = [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(args.samples)
                ]
                for begin, end in events:
                    begin.record(stream)
                    graph.replay()
                    end.record(stream)
                stream.synchronize()
                timings = torch.tensor(
                    [a.elapsed_time(b) * 1000 / args.calls for a, b in events],
                    device="cuda",
                    dtype=torch.float64,
                )
                dist.all_reduce(timings, op=dist.ReduceOp.MAX)
                assert_all(
                    torch.equal(
                        ws.output_activation[: args.capacity].view(torch.uint8),
                        expected.view(torch.uint8),
                    ),
                    ("post timing", early, active, holes),
                )
                payload = dict(
                    output=ws.output_activation[: args.capacity].cpu(),
                    reference=expected.cpu(),
                )
                values = timings.cpu().tolist()
                report["cases"].append(
                    dict(
                        early=early,
                        active=active,
                        holes=holes,
                        graph_nodes=count,
                        dense_sample_error=error,
                        status="pass",
                        median_us=statistics.median(values),
                        samples_us=values,
                        output_sha256=digest(payload["output"]),
                        reference_sha256=digest(payload["reference"]),
                    )
                )
                save_report()
                print(rank, early, active, holes, statistics.median(values), flush=True)
        graph.reset()
        graph = None
        barrier()
        ws.destroy()
        ws = None
    report["status"] = "pass"
    save_report()
finally:
    if graph is not None:
        graph.reset()
    if ws is not None:
        ws.destroy()
    finalize_moe_ep_runtime(runtime)
