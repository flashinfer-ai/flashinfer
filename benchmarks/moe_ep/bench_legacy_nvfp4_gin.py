# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Paired legacy NVFP4 performance measurements; launch through torchrun.

Run each source in a fresh process. CUDA events measure the complete compiled
GPU pipeline (mega + separate reduce), not an isolated fused kernel. A separate
Kineto trace records fused kernels. Full calls use the public FI API; donor's
full adapter stages the same prequantized buffers and returns an owned output.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

_HOST_GROUP = None


def barrier():
    dist.barrier(group=_HOST_GROUP)


def tensor_record(tensor, *, transpose=False):
    value = tensor.transpose(-1, -2) if transpose else tensor
    data = value.view(torch.uint8).contiguous().cpu()
    return dict(
        data=data, dtype=str(tensor.dtype).removeprefix("torch."), transpose=transpose
    )


def tensor_load(record):
    value = record["data"].to("cuda").view(getattr(torch, record["dtype"]))
    return value.transpose(-1, -2) if record["transpose"] else value


def tensor_digest(tensor):
    transpose = tensor.ndim == 3 and tensor.stride(-1) != 1
    record = tensor_record(tensor, transpose=transpose)
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        sha256=hashlib.sha256(record["data"].numpy().tobytes()).hexdigest(),
    )


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=_HOST_GROUP)
    return values


def make_fi(args, tokens, *, prepare=False):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    hidden, intermediate, experts = (
        (7168, 1536, 48) if args.mode == "on" else (2048, 1024, 8)
    )
    config = dict(
        intermediate_size=intermediate,
        top_k=4,
        kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl",
    )
    if args.mode == "on":
        config.update(internode=True, ranks_per_node=2)
    bootstrap = BootstrapConfig(
        world_size=world,
        rank=rank,
        auto_bootstrap=args.mode == "off",
        device=torch.cuda.current_device(),
    )
    fixture = None
    path = args.root / "fixtures" / f"on-rank-{rank}.pt"
    if args.mode == "on" and not prepare:
        fixture = torch.load(path, weights_only=True)
        w1, s1, w2, s2 = [
            tensor_load(fixture[name]) for name in ("w1", "s1", "w2", "s2")
        ]
        transformed = ((w1, s1), (w2, s2))
        weights = MoEWeightPack(w13=w1, w2=w2)
        backend = MegaConfig(
            megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**config),
            quantize_input=False,
            preprocess_weights=False,
            transformed_weights=transformed,
        )
        inputs = MoEEpTensors(
            hidden_states=tensor_load(fixture["x"]),
            scales=tensor_load(fixture["sf"]),
            topk_ids=tensor_load(fixture["ids"]),
            topk_weights=tensor_load(fixture["scores"]),
        )
    else:
        gen = torch.Generator(device="cuda").manual_seed(101 + rank)
        weights = MoEWeightPack(
            w13=torch.randn(
                (experts // world, 2 * intermediate, hidden),
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01,
            w2=torch.randn(
                (experts // world, hidden, intermediate),
                dtype=torch.bfloat16,
                device="cuda",
                generator=gen,
            )
            * 0.01,
        )
        backend = MegaConfig(
            megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**config),
            quantize_input=True,
            preprocess_weights=True,
        )
        gen = torch.Generator(device="cuda").manual_seed(301 + rank)
        x = torch.randn(
            (tokens, hidden), dtype=torch.bfloat16, device="cuda", generator=gen
        )
        scores, ids = torch.rand((tokens, experts), device="cuda", generator=gen).topk(
            4, dim=-1
        )
        inputs = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=scores)
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=experts, max_tokens_per_rank=tokens, token_hidden_size=hidden
        ),
        weights=weights,
        backend=backend,
    )
    backend_class = type(layer._kernel).__module__ + "." + type(layer._kernel).__name__
    assert backend_class.endswith(
        ".nvfp4_nvfp4_bf16_cutedsl.backend.Nvfp4CutedslMegaKernelBackend"
    ), f"Expected legacy backend, got {backend_class}"
    first = layer.forward(inputs)
    torch.cuda.synchronize()
    workspace = layer._workspace
    fe = workspace._frontend
    mega_inputs = layer._kernel._mega_inputs(workspace, layer._transformed)
    kwargs = fe._build_mega_runtime_kwargs(mega_inputs, fe._mega)
    payload = {
        name: getattr(mega_inputs, name)
        for name in (
            "activation",
            "activation_sf",
            "topk_idx",
            "topk_weights",
            "fc1_weight",
            "fc1_weight_sf",
            "fc2_weight",
            "fc2_weight_sf",
            "fc1_alpha",
            "fc2_alpha",
            "fc1_norm_const",
        )
    }
    if prepare:
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
            ordered_nvfp4_gin_reference,
            snapshot_nvfp4_gin_terms,
        )

        shard = snapshot_nvfp4_gin_terms(workspace)
        all_values = gather((shard, workspace.topk_idx.cpu()))
        expected = ordered_nvfp4_gin_reference(
            [value[0] for value in all_values],
            torch.stack([value[1] for value in all_values]),
            ranks_per_node=2,
        )[rank]
        assert torch.equal(first.cpu().view(torch.int16), expected.view(torch.int16))
        fixture = {
            name: tensor_record(tensor, transpose=name in ("w1", "w2"))
            for name, tensor in dict(
                x=workspace.x,
                sf=workspace.x_sf,
                ids=workspace.topk_idx,
                scores=workspace.topk_weights,
                w1=layer._transformed[0][0],
                s1=layer._transformed[0][1],
                w2=layer._transformed[1][0],
                s2=layer._transformed[1][1],
            ).items()
        }
        fixture["expected"] = expected
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixture, path)
    expected = fixture["expected"] if fixture is not None else None
    return dict(
        raw=lambda: fe._mega.compiled(**kwargs),
        full=lambda: layer.forward(inputs),
        output=lambda: workspace.output_activation,
        first=first,
        expected=expected,
        payload=payload,
        close=layer.destroy,
        settings=dict(
            backend=backend_class,
            config=repr(fe.config),
            kernel=fe._mega.kernel.name(),
            source=str(__import__("flashinfer").__file__),
        ),
    )


def make_donor(args, tokens):
    # Do not import FlashInfer here: its top-level raw source aliases would
    # contaminate the fixed donor's common/src namespace.
    from pre_unification.moe_nvfp4_swapab import mega_runner as runner

    assert "/pre_unification/moe_nvfp4_swapab/" in str(Path(runner.__file__).resolve())

    rank, world = dist.get_rank(), dist.get_world_size()
    fixture = torch.load(
        args.root / "fixtures" / f"on-rank-{rank}.pt", weights_only=True
    )
    data = {
        name: tensor_load(fixture[name])
        for name in ("x", "sf", "ids", "scores", "w1", "s1", "w2", "s2")
    }
    impl = runner.ImplDesc(
        internode=True,
        enable_internode_relay=True,
        enable_nccl_gin=True,
        ranks_per_node=2,
        enable_static_expert_shape=True,
        epi_flag_batch=(1, 4),
        token_back_mode="reuse_dispatch_warps",
    )
    problem = runner.TokenCommProblemDesc(
        world_size=world,
        num_tokens_per_rank=tokens,
        num_topk=4,
        num_total_experts=48,
        hidden=7168,
        intermediate=3072,
    )
    signals, barriers = runner._resolve_nccl_gin_resource_counts(
        impl,
        world_size=world,
        num_experts_per_rank=12,
        max_relay_tokens_per_node=tokens,
        hidden_bytes=3584,
        sf_uint32_per_token=112,
        num_topk=4,
        combine_hidden=7168,
    )
    context = runner.create_nccl_gin_context(
        rank=rank,
        world_size=world,
        signal_count=signals,
        barrier_count=barriers,
        connection_type="rail",
    )
    runner.validate_nccl_gin_rail_topology(
        context.support, world_size=world, ranks_per_node=2
    )
    tester = runner.MegaMoETester(
        problem, impl, runner.MiscDesc(), rank=rank, nccl_gin_context=context
    )
    tester._ensure_kernel()
    tester.allocate_workspaces()
    tester.my_fc1_weight, tester.my_fc1_weight_sf = data["w1"], data["s1"]
    tester.my_fc2_weight, tester.my_fc2_weight_sf = data["w2"], data["s2"]
    tester.my_fc1_alpha = torch.ones(12, device="cuda", dtype=torch.float32)
    tester.my_fc2_alpha = torch.ones_like(tester.my_fc1_alpha)
    tester.my_fc1_norm_const = torch.ones_like(tester.my_fc1_alpha)
    copies = [
        (tester.my_activation, data["x"]),
        (tester.my_activation_sf, data["sf"]),
        (tester.my_topk_idx, data["ids"]),
        (tester.my_topk_weights, data["scores"]),
    ]
    for destination, source in copies:
        destination.copy_(source)
    kwargs = {}
    tester._launch_target_kernels_with_optional_torch_profiler = (
        lambda values: kwargs.update(values)
    )
    tester.run_kernel()  # Original donor compilation/argument lowering; no timing or source edits.
    raw = lambda: tester._compiled_kernel(**kwargs)

    def full():
        owned = torch.empty_like(tester.output_activation)
        for destination, source in copies:
            destination.copy_(source)
        raw()
        owned.copy_(tester.output_activation)
        return owned

    raw()
    torch.cuda.synchronize()
    payload = dict(
        activation=tester.my_activation,
        activation_sf=tester.my_activation_sf,
        topk_idx=tester.my_topk_idx,
        topk_weights=tester.my_topk_weights,
        fc1_weight=tester.my_fc1_weight,
        fc1_weight_sf=tester.my_fc1_weight_sf,
        fc2_weight=tester.my_fc2_weight,
        fc2_weight_sf=tester.my_fc2_weight_sf,
        fc1_alpha=tester.my_fc1_alpha,
        fc2_alpha=tester.my_fc2_alpha,
        fc1_norm_const=tester.my_fc1_norm_const,
    )
    return dict(
        raw=raw,
        full=full,
        output=lambda: tester.output_activation,
        first=tester.output_activation,
        expected=fixture["expected"],
        payload=payload,
        close=tester.close,
        settings=dict(
            kernel=tester._kernel.name(),
            source=runner.__file__,
            backend=type(tester._kernel).__module__
            + "."
            + type(tester._kernel).__name__,
        ),
    )


def check_output(value, expected):
    actual = value.detach().cpu()
    assert torch.isfinite(actual).all(), "Nonfinite benchmark output"
    if expected is not None:
        mismatch = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        assert mismatch == 0, f"Benchmark output bit mismatches: {mismatch}"
    return actual


def gpu_clock():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()


def profile_raw(adapter, evict, cache, prefix, calls=5):
    path = Path(str(prefix) + f"-trace-{cache}.json")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for iteration in range(calls):
            if cache == "cold":
                evict.fill_(iteration % 255)
            torch.cuda.synchronize()
            barrier()
            with torch.profiler.record_function("MEASURE_RAW_PIPELINE"):
                adapter["raw"]()
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = [
        dict(name=entry["name"], duration_us=entry["dur"], start_us=entry["ts"])
        for entry in json.loads(path.read_text())["traceEvents"]
        if entry.get("cat") == "kernel"
    ]
    return dict(trace=str(path), events=events, calls=calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("on", "off"), required=True)
    parser.add_argument("--impl", choices=("base", "port", "donor"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--skip-profiles", action="store_true")
    profiles.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    global _HOST_GROUP
    _HOST_GROUP = dist.new_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    import cutlass
    import nccl
    from cuda.bindings import driver as cuda

    error, l2_bytes = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        cuda.CUdevice(torch.cuda.current_device()),
    )
    assert int(error) == 0 and 2 * l2_bytes <= 512 * 1024 * 1024
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3)), (
        "Blackwell required"
    )
    if args.mode == "on":
        assert (
            world,
            int(os.environ["LOCAL_WORLD_SIZE"]),
            int(os.environ["GROUP_WORLD_SIZE"]),
        ) == (4, 2, 2)
    if args.impl == "donor":
        assert args.mode == "on"
    folder = (
        args.root
        / (
            "smoke"
            if args.smoke
            else "profiles"
            if args.profile_only
            else "measurements"
        )
        / args.mode
        / f"ep{world}"
        / f"round-{args.round}"
        / args.impl
    )
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f"rank-{rank}"
    manifest = dict(
        rank=rank,
        world=world,
        node=socket.gethostname(),
        job=os.environ.get("SLURM_JOB_ID"),
        impl=args.impl,
        mode=args.mode,
        round=args.round,
        gpu=str(props),
        sm_count=props.multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dsl=getattr(cutlass, "__version__", None),
        nccl4py=getattr(nccl, "__version__", None),
        nccl=torch.cuda.nccl.version(),
        l2_cache_bytes=l2_bytes,
        loaded_modules=dict(
            torch=torch.__file__, cutlass=cutlass.__file__, nccl=nccl.__file__
        ),
        clocks_before=gpu_clock(),
        protocol="eager; synchronous per-call; host Gloo rendezvous outside each timed sample",
        cache_flush_bytes=512 * 1024 * 1024,
        profile_only=args.profile_only,
        skip_profiles=args.skip_profiles,
        source_commit=os.environ.get("BENCH_SOURCE_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[],
    )
    Path(str(prefix) + "-manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.prepare:
        adapter = make_fi(args, 32, prepare=True)
        adapter["close"]()
        barrier()
        dist.destroy_process_group()
        print(f"[perf] fixture prepared rank={rank}", flush=True)
        return
    evict = torch.empty(manifest["cache_flush_bytes"], dtype=torch.uint8, device="cuda")
    with Path(str(prefix) + "-samples.jsonl").open("x") as stream:
        for tokens in (32,) if args.mode == "on" else (64, 1024):
            adapter = (
                make_donor(args, tokens)
                if args.impl == "donor"
                else make_fi(args, tokens)
            )
            expected = adapter["expected"]
            if args.mode == "off":
                path = args.root / "fixtures" / f"off-ep{world}-t{tokens}-rank{rank}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                if (
                    args.impl == "base"
                    and args.round == 0
                    and not args.smoke
                    and not args.profile_only
                ):
                    torch.save(adapter["first"].cpu(), path)
                if not args.smoke:
                    expected = torch.load(path, weights_only=True)
            check_output(adapter["first"], expected)
            case = dict(
                tokens=tokens,
                settings=adapter["settings"],
                validation="bitwise PASS" if expected is not None else "finite smoke",
                payload={
                    name: tensor_digest(value)
                    for name, value in adapter["payload"].items()
                },
            )
            manifest["cases"].append(case)
            Path(str(prefix) + "-manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
            for cache in ("warm", "cold"):
                for scope in (
                    () if args.profile_only else ("raw_pipeline", "full_operator")
                ):
                    launch = (
                        adapter["raw"] if scope == "raw_pipeline" else adapter["full"]
                    )
                    last = None
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    end.record()
                    end.synchronize()  # Materialize event handles outside timing.
                    for iteration in range(args.warmups + args.samples):
                        if cache == "cold":
                            evict.fill_((iteration + 1) % 255)
                        torch.cuda.synchronize()
                        barrier()
                        start.record()
                        begin = time.perf_counter_ns()
                        last = launch()
                        enqueue_us = (time.perf_counter_ns() - begin) / 1000
                        end.record()
                        end.synchronize()
                        wall_us = (time.perf_counter_ns() - begin) / 1000
                        gpu_us = start.elapsed_time(end) * 1000
                        values = gather(
                            dict(
                                rank=rank,
                                wall_us=wall_us,
                                gpu_us=gpu_us,
                                enqueue_us=enqueue_us,
                            )
                        )
                        if iteration >= args.warmups:
                            record = dict(
                                tokens=tokens,
                                cache=cache,
                                scope=scope,
                                iteration=iteration - args.warmups,
                                ranks=values,
                            )
                            stream.write(json.dumps(record) + "\n")
                            stream.flush()
                    check_output(
                        adapter["output"]() if scope == "raw_pipeline" else last,
                        expected,
                    )
                    print(
                        f"[perf] rank={rank} impl={args.impl} ep={world} t={tokens} round={args.round} {cache}/{scope} PASS",
                        flush=True,
                    )
                if (
                    args.profile_only or args.round == 0 and not args.skip_profiles
                ) and not args.smoke:
                    case.setdefault("profiles", {})[cache] = profile_raw(
                        adapter, evict, cache, Path(str(prefix) + f"-t{tokens}")
                    )
                    check_output(adapter["output"](), expected)
            adapter["close"]()
            barrier()
            case["complete"] = True
            Path(str(prefix) + "-manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
    manifest["clocks_after"] = gpu_clock()
    manifest["complete"] = True
    Path(str(prefix) + "-manifest.json").write_text(json.dumps(manifest, indent=2))
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
