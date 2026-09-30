"""Native SM107 MegaMoE kernel and MoEEpLayer.forward latency benchmark.

Use --mode kernel|compute|forward and --execution eager|graph to select the span.
The primary metric is the maximum of per-rank p50 latencies, matching the
Blackwell autotuner's aggregation. JSONL also reports rank-zero p50,
preserving every sample, timing/cache
protocol, resolved knobs, shape, seed, environment, accuracy, and memory.
A sampled Torch oracle gates every result.

Example (four GPUs in one NVLink domain, CUTE_DSL_ARCH=sm_107a exported):
    torchrun --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm107_block_scaled_mega.py \\
        --quant-kind all --routing both --mode forward --execution graph --no-l2-flush

See kernel_src/sm107/next_cutedsl_megamoe/TUNING.md for the complete
qualification matrix and the distinction between default, heuristic,
cached, and historically reported knob profiles.
"""

from __future__ import annotations

import argparse
import json
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
        # bench_common.make_problem at moe_ep_benchmark ba9f8acb: independent
        # rank seeds, unsorted top-k, and the selected scores as routing weights.
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
        timing_method="cuda_event",
        timed_span={
            "kernel": "prestaged_kernel_reset_reduce",
            "compute": "prestaged_backend_compute_and_output_copy",
            "forward": "bf16_public_forward",
        }[args.mode],
        timing_scope="gpu_graph_replay" if args.execution == "graph" else "gpu_stream",
        host_wall_time_measured=False,
        input_staging_in_timed_span=args.mode == "forward",
        output_ownership="workspace_view" if args.mode == "kernel" else "owned",
        output_allocation=allocation,
        synchronization="barrier_before_batch_no_inter_iteration_sync",
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


def _selected_knobs(pkg, policy, tokens, capacity, routing, quant_kind, world):
    if policy == "default":
        return None
    if policy == "heuristic":
        return pkg.default_knobs(capacity, quant_kind=quant_kind)
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
            256 if quant_kind == "nvfp4" else 128,
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
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    selection = _selected_knobs(
        pkg, args.knobs, tokens, capacity, routing, quant_kind, world
    )
    if args.variant is not None and isinstance(selection, dict):
        selection = dict(selection, reduce_topk_in_kernel=args.variant == "ikr")
    common = dict(
        intermediate_size=INTERMEDIATE,
        top_k=TOP_K,
        knobs=selection,
        in_kernel_fc2_reduce=args.variant == "ikr",
        gate_up_clamp=10.0 if args.input_profile == "blackwell" else None,
    )
    cfg = (
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**common)
        if quant_kind == "nvfp4"
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
        backend=MegaConfig(megakernel=cfg, transformed_weights=transformed),
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
        for i in range(args.iters):
            if not args.no_l2_flush:
                # Retain the allocation through the launch, as Blackwell does.
                flush_buffer = _l2_flush()
            starts[i].record()
            output = invoke()
            stops[i].record()
        torch.cuda.synchronize()
        if not args.no_l2_flush:
            del flush_buffer
        samples = [
            a.elapsed_time(b) * 1000.0 for a, b in zip(starts, stops, strict=False)
        ]
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
        if float(error_tensor) > (0.06 if quant_kind == "nvfp4" else 0.02):
            raise RuntimeError(
                f"benchmark output failed sampled Torch oracle: rel_l2={float(error_tensor)}"
            )
        workspace = layer._workspace
        counts = torch.bincount(
            topk_idx.flatten().long(), minlength=NUM_EXPERTS
        ).float()
        result = dict(
            schema_version=3,
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
        git_revision=command(["git", "rev-parse", "HEAD"]),
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
                "CUDA_VISIBLE_DEVICES",
            )
        },
    )


def main():
    global HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, SEED

    parser = argparse.ArgumentParser(description=__doc__)
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
        choices=["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "both", "all"],
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
        ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"]
        if args.quant_kind == "all"
        else ["nvfp4", "mxfp8_e4m3"]
        if args.quant_kind == "both"
        else [args.quant_kind]
    )
    environment = _environment()
    try:
        for kind in kinds:
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            transformed = _local_transformed_weights(
                rank, world, kind, args.input_profile
            )
            torch.cuda.synchronize()
            preparation = dict(
                elapsed_seconds=time.perf_counter() - start,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            for routing in routings:
                for tokens in token_list:
                    result, samples = _bench_one(
                        rank,
                        world,
                        tokens,
                        args.capacity or tokens,
                        routing,
                        transformed,
                        args,
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
                        with open(args.output, "a") as stream:
                            stream.write(json.dumps(record) + "\n")
                        print(
                            f"{kind} EP{world} {routing} T={tokens}/{args.capacity or tokens} "
                            f"{args.mode}/{args.execution}: max(rank p50)={summary['max_rank_p50_us']:.2f} us, "
                            f"p50(rank 0)={summary['p50_rank0_us']:.2f} us, "
                            f"rel_l2={result['relative_l2_max_rank']:.5f}",
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
