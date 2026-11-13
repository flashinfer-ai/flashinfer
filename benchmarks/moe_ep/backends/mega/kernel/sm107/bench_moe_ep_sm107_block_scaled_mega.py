"""Native SM107 MegaMoE kernel and MoEEpLayer.forward latency benchmark.

Use --mode kernel|compute|forward and --execution eager|graph to select the span.
The primary metric is the maximum of per-rank p50 latencies, matching the
Blackwell autotuner's aggregation. JSONL also reports rank-zero p50,
preserving every sample, timing/cache
protocol, resolved knobs, shape, seed, environment, accuracy, and memory.
A sampled Torch oracle gates every result. --kernel-variant selects inference or
GenPhase; --prequantized prepares activation bytes outside the forward span.
--eager-completed-wall includes host dispatch and device completion. Without it,
eager timing retains its GPU-stream boundary. --testlist reads explicit workload
rows for the current EP degree and measures graph and completed eager forwards.

Example (four GPUs in one NVLink domain, CUTE_DSL_ARCH=sm_107a exported):
    torchrun --standalone --nproc_per_node=4 benchmarks/moe_ep/backends/mega/kernel/sm107/bench_moe_ep_sm107_block_scaled_mega.py \\
        --quant-kind all --routing both --mode forward --execution graph --no-l2-flush

See kernel_src/sm107/next_cutedsl_megamoe/TUNING.md for the complete
qualification matrix and the distinction between default, heuristic,
cached, and historically reported knob profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist

# DSv4 Pro EP4 problem (matches the upstream perf report).
HIDDEN = 7168
INTERMEDIATE = 3072
NUM_EXPERTS = 384
TOP_K = 6
WEIGHT_CHUNK_EXPERTS = 8  # bf16 generation + quantize peak-memory bound
SEED = 20260817
L2_FLUSH_BYTES = 300 * 1024 * 1024

# Upstream selected-best knobs per (routing, tokens/rank). Every winner uses
# mixed CGA (preferred 4x1, fallback 2x1), phase-interleave scheduling, atomic
# work IDs, FC2 bulk TMA stage 2, epi-warp token back, and separate top-k
# reduction; only tile / hint / epi flag batches / token-in flag batch vary.
WINNERS = {
    ("balanced", 1024): dict(tile=(256, 128, 256), hint=4, epi=(1, 4), tif=1),
    ("balanced", 2048): dict(tile=(256, 256, 256), hint=3, epi=(2, 4), tif=1),
    ("balanced", 4096): dict(tile=(256, 256, 256), hint=3, epi=(2, 4), tif=1),
    ("balanced", 8192): dict(tile=(256, 256, 256), hint=3, epi=(1, 4), tif=1),
    ("balanced", 16384): dict(tile=(256, 256, 256), hint=3, epi=(1, 4), tif=1),
    ("balanced", 32768): dict(tile=(256, 256, 256), hint=3, epi=(1, 4), tif=1),
    ("power_law", 1024): dict(tile=(256, 128, 256), hint=3, epi=(1, 4), tif=1),
    ("power_law", 2048): dict(tile=(256, 256, 256), hint=3, epi=(1, 4), tif=1),
    ("power_law", 4096): dict(tile=(256, 256, 256), hint=4, epi=(1, 4), tif=1),
    ("power_law", 8192): dict(tile=(256, 256, 256), hint=3, epi=(1, 4), tif=1),
    ("power_law", 16384): dict(tile=(256, 256, 256), hint=3, epi=(1, 4), tif=4),
    ("power_law", 32768): dict(tile=(256, 256, 256), hint=3, epi=(1, 4), tif=4),
}


def _topk_idx_balanced(generator, world, tokens, topk, experts, device):
    """Block-balanced routing (upstream tester/generate_inputs.py): each padded
    block of ``experts`` tokens sends exactly ``topk`` tokens to every expert."""
    padded = ((tokens + experts - 1) // experts) * experts
    blocks = padded // experts
    expert_perms = torch.rand(
        (world, blocks, experts), device=device, generator=generator
    ).argsort(dim=-1)
    topk_offsets = torch.rand(
        (world, blocks, experts), device=device, generator=generator
    ).argsort(dim=-1)[..., :topk]
    token_offsets = torch.arange(experts, device=device).view(1, 1, experts, 1)
    expert_indices = (token_offsets + topk_offsets.unsqueeze(2)) % experts
    src = expert_perms.unsqueeze(-1).expand(world, blocks, experts, topk)
    topk_blocks = torch.gather(src, 2, expert_indices)
    return topk_blocks.reshape(world, padded, topk)[:, :tokens, :]


def _topk_idx_power_law(generator, world, tokens, topk, experts, exponent, device):
    """Zipf-popularity routing (upstream): prob ~ 1/rank**exponent, Gumbel
    top-k draws distinct experts per token."""
    popularity = torch.randperm(experts, device=device, generator=generator)
    rank_freq = 1.0 / (
        torch.arange(1, experts + 1, device=device, dtype=torch.float64) ** exponent
    )
    probs = torch.empty(experts, dtype=torch.float64, device=device)
    probs[popularity] = rank_freq / rank_freq.sum()
    log_probs = torch.log(probs.clamp_min(1e-30))
    slots = world * tokens
    uniform = torch.rand(
        (slots, experts), device=device, dtype=torch.float64, generator=generator
    )
    gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-30)) + 1e-30)
    scores = log_probs.unsqueeze(0) + gumbel
    idx = scores.topk(topk, dim=-1).indices
    return idx.reshape(world, tokens, topk)


def _make_routing(world, tokens, routing, alpha, device="cuda"):
    if routing == "gaussian":
        # Independent rank seeds, unsorted top-k, and selected scores as weights.
        ids, weights = [], []
        for rank in range(world):
            generator = torch.Generator(device=device).manual_seed(SEED + 17 + rank)
            scores = torch.randn(
                tokens,
                NUM_EXPERTS,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            values, indices = torch.topk(scores, TOP_K, dim=-1, sorted=False)
            ids.append(indices)
            weights.append(values)
        return torch.stack(ids).to(torch.int32), torch.stack(weights)
    generator = torch.Generator(device=device).manual_seed(SEED + tokens)
    if routing == "balanced":
        topk_idx = _topk_idx_balanced(
            generator, world, tokens, TOP_K, NUM_EXPERTS, device
        )
    else:
        topk_idx = _topk_idx_power_law(
            generator, world, tokens, TOP_K, NUM_EXPERTS, alpha, device
        )
    topk_weights = (
        torch.rand((world, tokens, TOP_K), device=device, generator=generator) + 0.5
    )
    return topk_idx.to(torch.int32), topk_weights


def _local_transformed_weights(
    rank: int, world: int, quant_kind: str, input_profile: str = "rubin"
):
    """Random local expert slice, quantized chunk-wise to bound peak memory."""
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        concatenate_block_scaled_weights,
    )

    if quant_kind == "nvfp4":
        from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl import (
            weights as weights_mod,
        )

        extra = {}
    elif quant_kind in ("mxfp4", "mxfp4_mxfp8"):
        from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp4_bf16_cutedsl import (
            weights as weights_mod,
        )

        extra = {}
    else:
        from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl import (
            weights as weights_mod,
        )

        extra = {"kind": quant_kind}

    experts_per_rank = NUM_EXPERTS // world
    generator = torch.Generator(device="cuda").manual_seed(SEED + 7 * rank)
    if input_profile == "blackwell":
        # Preserve the historical full-bank RNG sequence; conversion still
        # proceeds in chunks. These allocations are outside the timed span.
        generator.manual_seed(SEED + 13 + rank)
        w13_bank = (
            torch.randn(
                experts_per_rank,
                2 * INTERMEDIATE,
                HIDDEN,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            / 15.0
        )
        w2_bank = (
            torch.randn(
                experts_per_rank,
                HIDDEN,
                INTERMEDIATE,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            / 15.0
        )
    fc1_parts, fc2_parts = [], []
    for begin in range(0, experts_per_rank, WEIGHT_CHUNK_EXPERTS):
        count = min(WEIGHT_CHUNK_EXPERTS, experts_per_rank - begin)
        w13 = (
            w13_bank[begin : begin + count]
            if input_profile == "blackwell"
            else (
                torch.randn(
                    count,
                    2 * INTERMEDIATE,
                    HIDDEN,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                * HIDDEN**-0.5
            ).to(torch.bfloat16)
        )
        w2 = (
            w2_bank[begin : begin + count]
            if input_profile == "blackwell"
            else (
                torch.randn(
                    count,
                    HIDDEN,
                    INTERMEDIATE,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                * INTERMEDIATE**-0.5
            ).to(torch.bfloat16)
        )
        (fc1_w, fc1_sf), (fc2_w, fc2_sf) = weights_mod.preprocess_mega_weights(
            MoEWeightPack(w13=w13, w2=w2),
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            **extra,
        )
        fc1_parts.append((fc1_w, fc1_sf.reshape(count, -1)))
        fc2_parts.append((fc2_w, fc2_sf.reshape(count, -1)))
        del w13, w2
        torch.cuda.empty_cache()
    return (
        concatenate_block_scaled_weights(fc1_parts),
        concatenate_block_scaled_weights(fc2_parts),
    )


def _l2_flush() -> torch.Tensor:
    """Match Blackwell's fresh random-write buffer before each start event."""
    return torch.randn(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")


def _make_compute_call(layer, transformed, x):
    """Reuse an owned output on already-staged inputs, matching Blackwell."""
    workspace = layer._workspace
    output = torch.empty_like(x)

    def invoke():
        return layer._kernel.compute(workspace, transformed, output=output)

    return invoke


def _timing_protocol(args):
    """Describe the measured work separately from cache and launch scheduling."""
    if args.mode == "kernel":
        allocation = "none"
    elif args.mode == "compute":
        allocation = "before_warmup"
    else:
        allocation = "during_capture" if args.execution == "graph" else "per_call"
    return dict(
        timing_method="perf_counter_completed_wall"
        if args.eager_completed_wall
        else "cuda_event",
        timed_span={
            "kernel": "prestaged_kernel_reset_reduce",
            "compute": "prestaged_backend_compute_and_output_copy",
            "forward": "prequantized_public_forward"
            if args.prequantized
            else "bf16_public_forward",
        }[args.mode],
        timing_scope=(
            "host_dispatch_and_completion"
            if args.eager_completed_wall
            else "gpu_graph_replay"
            if args.execution == "graph"
            else "gpu_stream"
        ),
        host_wall_time_measured=args.eager_completed_wall,
        input_staging_in_timed_span=args.mode == "forward",
        output_ownership="workspace_view" if args.mode == "kernel" else "owned",
        output_allocation=allocation,
        synchronization=(
            "barrier_before_batch_device_sync_after_each_call"
            if args.eager_completed_wall
            else "barrier_before_batch_no_inter_iteration_sync"
        ),
        cache_policy="consecutive_launches" if args.no_l2_flush else "l2_flushed",
        l2_flush_bytes=0 if args.no_l2_flush else L2_FLUSH_BYTES,
        l2_flush_method="none" if args.no_l2_flush else "per_iteration_fp32_randn",
        graph_replay_warmup=args.warmup if args.execution == "graph" else 0,
    )


def _summarize_samples(per_rank):
    """Report the primary and historical medians, retaining raw rank samples."""
    medians = [statistics.median(samples) for samples in per_rank]
    return {
        "primary_latency_statistic": "max_rank_p50_us",
        "max_rank_p50_us": max(medians),
        "p50_rank0_us": medians[0],
        "per_rank_samples_us": per_rank,
    }


def _selected_knobs(
    pkg,
    policy,
    tokens,
    capacity,
    routing,
    quant_kind,
    world,
    kernel_variant="inference",
):
    if policy == "default":
        return None
    if policy == "heuristic":
        return pkg.default_knobs(
            capacity, quant_kind=quant_kind, kernel_variant=kernel_variant
        )
    if policy == "cache":
        return "cache"
    if policy != "reported":
        value = json.loads(policy)
        if not isinstance(value, dict):
            raise ValueError("--knobs must name a policy or contain a JSON knob object")
        return {k: tuple(v) if isinstance(v, list) else v for k, v in value.items()}
    if world != 4 or (HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K) != (7168, 3072, 384, 6):
        raise ValueError(
            "the reported profile only applies to H7168/I3072/E384/K6 at EP4"
        )
    if (routing, tokens) not in WINNERS:
        raise ValueError(
            "no reported profile for this routing/token count; use default, heuristic, cache, or explicit knobs"
        )
    row = WINNERS[(routing, tokens)]
    return dict(
        mma_tiler_mnk=(
            row["tile"][0],
            row["tile"][1],
            256 if quant_kind in ("nvfp4", "mxfp4") else 128,
        ),
        cluster_shape_mn=(4, 1),
        fallback_cluster_shape_mn=(2, 1),
        schedule_policy=("phase_interleave", row["hint"]),
        work_id_mode="atomic_counter",
        fc2_use_bulk=True,
        fc2_tma_stages=2,
        epi_flag_batches=tuple(row["epi"]),
        token_in_flag_batch=row["tif"],
        token_back_mode="epi_warps",
        reduce_topk_in_kernel=False,
    )


def _bench_one(rank, world, tokens, capacity, routing, transformed, args, quant_kind):
    import dataclasses

    import flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe as pkg
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
        Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    selection = _selected_knobs(
        pkg,
        args.knobs,
        tokens,
        capacity,
        routing,
        quant_kind,
        world,
        args.kernel_variant,
    )
    if args.variant is not None and isinstance(selection, dict):
        selection = dict(selection, reduce_topk_in_kernel=args.variant == "ikr")
    common = dict(
        intermediate_size=INTERMEDIATE,
        top_k=TOP_K,
        knobs=selection,
        kernel_variant=args.kernel_variant,
        in_kernel_fc2_reduce=args.variant == "ikr",
        gate_up_clamp=10.0 if args.input_profile == "blackwell" else None,
    )
    cfg = (
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**common)
        if quant_kind == "nvfp4"
        else Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(**common)
        if quant_kind == "mxfp4"
        else Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(**common)
        if quant_kind == "mxfp4_mxfp8"
        else Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(kind=quant_kind, **common)
    )
    torch.cuda.reset_peak_memory_stats()
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world, rank=rank, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=capacity,
            token_hidden_size=HIDDEN,
        ),
        weights=None,
        backend=MegaConfig(
            megakernel=cfg,
            transformed_weights=transformed,
            quantize_input=not args.prequantized,
        ),
    )
    try:
        if args.input_profile == "blackwell":
            gen = torch.Generator(device="cuda").manual_seed(SEED + 7 + rank)
            x = (
                torch.randn(
                    tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen
                )
                / 10.0
            )
        else:
            gen = torch.Generator(device="cuda").manual_seed(SEED + 13 * rank + tokens)
            x = torch.randn(
                tokens, HIDDEN, device="cuda", dtype=torch.float32, generator=gen
            ).bfloat16()
        topk_idx, topk_weights = _make_routing(world, tokens, routing, args.alpha)
        if args.prequantized:
            if quant_kind == "nvfp4":
                xq, sf = pkg.quantize_nvfp4_block16(x)
            elif quant_kind == "mxfp4":
                xq, sf = pkg.quantize_mxfp4_block32(x)
            else:
                dtype = (
                    torch.float8_e5m2
                    if quant_kind == "mxfp8_e5m2"
                    else torch.float8_e4m3fn
                )
                xq, sf = pkg.quantize_mxfp8_block32(x, dtype)
            tensors = MoEEpTensors(xq, topk_idx[rank], topk_weights[rank], scales=sf)
        else:
            tensors = MoEEpTensors(x, topk_idx[rank], topk_weights[rank])
        dist.barrier()
        layer.forward(tensors)
        resolved_variant = (
            "ikr" if layer._workspace.config.reduce_topk_in_kernel else "bf16"
        )
        if args.variant is not None and resolved_variant != args.variant:
            raise RuntimeError(
                f"requested variant {args.variant}, but knobs resolved to {resolved_variant}; "
                "use a matching cache or explicit knobs"
            )
        indices, expected = pkg.sampled_reference(
            layer._workspace, *transformed, tokens
        )
        if args.mode == "kernel":
            invoke = pkg.sm107_block_scaled_mega_launch_thunk(
                *transformed, layer._workspace
            )
        elif args.mode == "compute":
            invoke = _make_compute_call(layer, transformed, x)
        else:

            def invoke():
                return layer.forward(tensors)

        torch.cuda.synchronize()
        for _ in range(args.warmup):
            invoke()
        torch.cuda.synchronize()
        if args.execution == "graph":
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = invoke()
            invoke = graph.replay
            # Warm up graph replay itself so upload/first-replay effects are
            # excluded along with compilation and capture.
            for _ in range(args.warmup):
                invoke()
            torch.cuda.synchronize()
        dist.barrier()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        wall_samples = []
        for i in range(args.iters):
            if not args.no_l2_flush:
                # Retain the allocation through the launch, as Blackwell does.
                flush_buffer = _l2_flush()
                if args.eager_completed_wall:
                    torch.cuda.synchronize()
            starts[i].record()
            wall_start = time.perf_counter_ns() if args.eager_completed_wall else 0
            output = invoke()
            if args.eager_completed_wall:
                torch.cuda.synchronize()
                wall_samples.append((time.perf_counter_ns() - wall_start) / 1000)
            stops[i].record()
        torch.cuda.synchronize()
        if not args.no_l2_flush:
            del flush_buffer
        samples = [
            a.elapsed_time(b) * 1000.0 for a, b in zip(starts, stops, strict=False)
        ]
        if args.eager_completed_wall:
            samples = wall_samples
        observed = (
            layer._workspace.output_activation[:tokens]
            if args.mode == "kernel"
            else graph_output
            if args.execution == "graph"
            else output
        )
        error = pkg.output_error(observed, indices, expected)
        error_tensor = torch.tensor(error, device="cuda", dtype=torch.float64)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
        if not math.isfinite(float(error_tensor)) or float(error_tensor) > (
            0.06 if quant_kind in ("nvfp4", "mxfp4") else 0.02
        ):
            raise RuntimeError(
                f"benchmark output failed sampled Torch oracle: rel_l2={float(error_tensor)}"
            )
        workspace = layer._workspace
        counts = torch.bincount(
            topk_idx.flatten().long(), minlength=NUM_EXPERTS
        ).float()
        result = dict(
            schema_version=3,
            case_id=args.case_id,
            kernel_variant=workspace.config.kernel_variant,
            prequantized=args.prequantized,
            rank=rank,
            routing=routing,
            alpha=args.alpha,
            seed=SEED,
            repetition=args.repetition,
            quant_kind=quant_kind,
            variant=resolved_variant,
            combine_dtype="bf16",
            input_profile=args.input_profile,
            tokens=tokens,
            capacity=capacity,
            hidden=HIDDEN,
            intermediate=INTERMEDIATE,
            num_experts=NUM_EXPERTS,
            topk=TOP_K,
            world_size=world,
            mode=args.mode,
            execution=args.execution,
            l2_flush=not args.no_l2_flush,
            warmup=args.warmup,
            iterations=args.iters,
            timing_protocol=_timing_protocol(args),
            knob_policy=args.knobs,
            resolved_config=dataclasses.asdict(workspace.config),
            relative_l2_max_rank=float(error_tensor),
            oracle_sample_rows=int(indices.numel()),
            routing_max_mean=float(counts.max() / counts.mean()),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            workspace_bytes={
                name: getattr(workspace, name).numel()
                * getattr(workspace, name).element_size()
                for name in (
                    "x",
                    "x_sf",
                    "topk_idx",
                    "topk_weights",
                    "output_activation",
                    "local_workspace",
                    "shared_workspace",
                )
            },
        )
        if args.execution == "graph":
            del observed, graph_output, invoke, graph
    except BaseException:
        # Failure is fatal to the distributed job. Do not enter a collective
        # free while a peer might be stuck in a kernel or have a CUDA fault.
        layer = None
        raise
    finally:
        if layer is not None:
            layer.destroy()
    return result, samples


def _environment():
    import datetime
    import importlib.metadata
    import flashinfer
    import subprocess

    def command(argv):
        try:
            return subprocess.check_output(
                argv, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {}
    for package in ("nvidia-cutlass-dsl", "cuda-python", "nvshmem4py-cu13"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_revision=os.environ.get("MEGAMOE_SOURCE_SHA")
        or command(["git", "rev-parse", "HEAD"]),
        flashinfer_source_path=flashinfer.__file__,
        git_status=command(["git", "status", "--porcelain"]),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(),
        compute_capability=torch.cuda.get_device_capability(),
        gpu_total_memory=torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory,
        driver=command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        versions=versions,
        environment={
            key: os.environ.get(key)
            for key in (
                "CUTE_DSL_ARCH",
                "NVSHMEM_SYMMETRIC_SIZE",
                "NVSHMEM_DISABLE_NVLS",
                "NVSHMEM_REMOTE_TRANSPORT",
                "NVSHMEM_DISABLE_P2P",
                "FLASHINFER_MOE_EP_KNOB_CACHE",
                "FLASHINFER_SM107_MEGA_CHECK_ROUTES",
                "FLASHINFER_SM107_FUSED_ROUTE_CHECK",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    )


def _testlist(args, world):
    """Expand an explicit workload list without changing standalone defaults."""
    if not args.testlist:
        return [args]
    with open(args.testlist) as stream:
        rows = json.load(stream)
    allowed = {
        "case_id",
        "world_size",
        "hidden",
        "intermediate",
        "num_experts",
        "topk",
        "tokens",
        "capacity",
        "quant_kind",
        "kernel_variant",
        "prequantized",
    }
    cases, ids = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != allowed:
            raise ValueError(
                "each testlist row must specify exactly " + ", ".join(sorted(allowed))
            )
        if not row["case_id"] or row["case_id"] in ids:
            raise ValueError("testlist case IDs must be nonempty and unique")
        ids.add(row["case_id"])
        if row["world_size"] != world:
            continue
        values = dict(vars(args))
        values.update({key: value for key, value in row.items() if key != "world_size"})
        values["tokens"] = str(values["tokens"])
        case = argparse.Namespace(**values)
        if any(
            type(getattr(case, name)) is not int
            for name in ("hidden", "intermediate", "num_experts", "topk", "capacity")
        ):
            raise ValueError("testlist dimensions and capacity must be integers")
        if (
            min(
                case.hidden,
                case.intermediate,
                case.num_experts,
                case.topk,
                int(case.tokens),
            )
            < 1
        ):
            raise ValueError("testlist dimensions must be positive")
        if (
            case.capacity < int(case.tokens)
            or case.num_experts % world
            or case.topk > case.num_experts
        ):
            raise ValueError("invalid capacity, expert partition or top-k")
        if (
            case.kernel_variant not in ("inference", "genphase")
            or case.quant_kind
            not in ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4", "mxfp4_mxfp8")
            or type(case.prequantized) is not bool
        ):
            raise ValueError("invalid kernel variant, format or input policy")
        if case.mode != "forward" or case.knobs not in ("default", "heuristic"):
            raise ValueError("testlists require forward timing and built-in knobs")
        if case.kernel_variant == "genphase" and (
            case.knobs != "heuristic" or case.capacity > 1024
        ):
            raise ValueError(
                "GenPhase testlists require heuristic knobs and capacity <= 1024"
            )
        cases.append(case)
    if not cases:
        raise ValueError(f"no testlist cases for EP{world}")
    return cases


def main():
    global HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, SEED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testlist", help="JSON testlist; select rows matching WORLD_SIZE"
    )
    parser.add_argument("--case-id", default="standalone")
    parser.add_argument(
        "--kernel-variant", choices=["inference", "genphase"], default="inference"
    )
    parser.add_argument("--prequantized", action="store_true")
    parser.add_argument(
        "--eager-completed-wall",
        action="store_true",
        help="time eager host dispatch through device completion for each call",
    )
    parser.add_argument("--tokens", default="8,64,512,1024,2048,4096,8192")
    parser.add_argument(
        "--capacity",
        type=int,
        help="fixed capacity >= every live token count; default: live size",
    )
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--topk", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--input-profile",
        choices=["rubin", "blackwell"],
        default="rubin",
        help="blackwell: historical BF16 activation/weight RNG and scales; use --seed 0",
    )
    parser.add_argument(
        "--variant",
        choices=["bf16", "ikr"],
        help="BF16 combine with separate reduction or in-kernel atomic reduction; "
        "overrides the reduction knob and verifies the resolved configuration",
    )
    parser.add_argument(
        "--quant-kind",
        default="nvfp4",
        choices=[
            "nvfp4",
            "mxfp8_e4m3",
            "mxfp8_e5m2",
            "mxfp4",
            "mxfp4_mxfp8",
            "both",
            "all",
        ],
    )
    parser.add_argument(
        "--mode", choices=["kernel", "compute", "forward"], default="kernel"
    )
    parser.add_argument("--execution", choices=["eager", "graph"], default="eager")
    parser.add_argument(
        "--knobs",
        default="default",
        help="default, heuristic, cache, reported, or a JSON knob object",
    )
    parser.add_argument("--no-l2-flush", action="store_true")
    parser.add_argument(
        "--routing",
        default="gaussian",
        choices=["gaussian", "balanced", "power_law", "both"],
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--repetition", type=int, default=1, help="label an independent fixed-seed run"
    )
    parser.add_argument("--output", default="bench_sm107_mega_results.jsonl")
    args = parser.parse_args()
    if args.eager_completed_wall and args.execution != "eager":
        parser.error("--eager-completed-wall requires --execution eager")
    if args.kernel_variant == "genphase" and (
        args.variant == "ikr" or args.knobs == "reported"
    ):
        parser.error("GenPhase does not use IKR or reported inference profiles")
    if args.knobs == "reported" and args.variant == "ikr":
        parser.error(
            "reported profiles use separate reduction; do not label an IKR override reported"
        )
    HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, SEED = (
        args.hidden,
        args.intermediate,
        args.num_experts,
        args.topk,
        args.seed,
    )
    token_list = [int(t) for t in args.tokens.split(",")]
    if min(token_list) < 1 or min(args.warmup, args.iters, args.repetition) < 1:
        parser.error("tokens, warmup, iters, and repetition must be positive")
    if args.capacity is not None and args.capacity < max(token_list):
        parser.error("--capacity must cover every live token count")
    if (
        not args.testlist
        and args.kernel_variant == "genphase"
        and (args.knobs == "default" or (args.capacity or max(token_list)) > 1024)
    ):
        parser.error(
            "GenPhase requires heuristic, cache, or explicit knobs and capacity <= 1024"
        )
    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    if world < 2:
        parser.error("run under torchrun with at least two EP ranks")
    if NUM_EXPERTS % world or not 1 <= TOP_K <= NUM_EXPERTS:
        parser.error(
            "num-experts must divide across ranks and topk must be in [1, num-experts]"
        )
    cases = _testlist(args, world)
    from flashinfer.moe_ep import (
        BootstrapConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.runtime import sm107_block_scaled_runtime_requirements
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        require_sm107_dsl,
    )

    bootstrap = BootstrapConfig(world_size=world, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    if torch.cuda.get_device_capability() != (10, 7):
        parser.error("this benchmark requires native SM107 GPUs")
    require_sm107_dsl()
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, sm107_block_scaled_runtime_requirements(bootstrap)
    )
    routings = ["balanced", "power_law"] if args.routing == "both" else [args.routing]
    kinds = (
        ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4", "mxfp4_mxfp8"]
        if args.quant_kind == "all"
        else ["nvfp4", "mxfp8_e4m3"]
        if args.quant_kind == "both"
        else [args.quant_kind]
    )
    environment = _environment()
    try:
        for case in cases:
            HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, SEED = (
                case.hidden,
                case.intermediate,
                case.num_experts,
                case.topk,
                case.seed,
            )
            # One weight bank per testlist row, reused by both timing boundaries.
            for kind in [case.quant_kind] if case.testlist else kinds:
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                transformed = _local_transformed_weights(
                    rank, world, kind, case.input_profile
                )
                torch.cuda.synchronize()
                preparation = dict(
                    elapsed_seconds=time.perf_counter() - start,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                for routing in routings:
                    for tokens in [int(t) for t in case.tokens.split(",")]:
                        executions = (
                            [("graph", False), ("eager", True)]
                            if case.testlist
                            else [(case.execution, case.eager_completed_wall)]
                        )
                        for case.execution, case.eager_completed_wall in executions:
                            result, samples = _bench_one(
                                rank,
                                world,
                                tokens,
                                case.capacity or tokens,
                                routing,
                                transformed,
                                case,
                                kind,
                            )
                            local = dict(
                                result=result,
                                samples_us=samples,
                                preprocessing=preparation,
                                environment=environment,
                            )
                            gathered = [None] * world
                            dist.all_gather_object(gathered, local)
                            if rank == 0:
                                summary = _summarize_samples(
                                    [r["samples_us"] for r in gathered]
                                )
                                summary["rank_max_median_us"] = statistics.median(
                                    max(values)
                                    for values in zip(
                                        *(r["samples_us"] for r in gathered),
                                        strict=True,
                                    )
                                )
                                summary["model_tflops_per_rank"] = (
                                    tokens
                                    * TOP_K
                                    * 6
                                    * HIDDEN
                                    * INTERMEDIATE
                                    / summary["max_rank_p50_us"]
                                    / 1e6
                                )
                                record = dict(result, **summary, per_rank=gathered)
                                with open(case.output, "a") as stream:
                                    stream.write(json.dumps(record) + "\n")
                                print(
                                    f"{case.case_id} {kind} EP{world} T={tokens}: "
                                    f"{case.execution} {summary['rank_max_median_us']:.2f} us",
                                    flush=True,
                                )
                            dist.barrier()
                del transformed
                torch.cuda.empty_cache()
    except BaseException:
        runtime = None
        raise
    finally:
        if runtime is not None:
            finalize_moe_ep_runtime(runtime)


if __name__ == "__main__":
    main()
