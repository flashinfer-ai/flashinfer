"""4-GPU SM107 (Rubin) block-scaled mega kernel latency benchmark.

Reproduces the upstream cutedsl_megamoe Rubin TS4B perf-report protocol
(tested upstream at 47881ad2 / vendored 92dd334) on the flashinfer moe_ep
backends: DSv4-Pro EP4 shape (hidden 7168, MoE intermediate 3072, 384 total
experts, top-k 6, BF16 combine), NVFP4 and/or MXFP8 (--quant-kind; the
upstream baseline is NVFP4-only), balanced + power-law(0.8) routing,
5 warmup + 20 measured iterations, per-rank averages.

Timing covers ONLY the fused mega kernel launch (dispatch + FC1 + SwiGLU +
FC2 + combine) via the shim's steady-state launch thunk over pre-staged
inputs — the same span the upstream tester times. Input staging (torch
quantization fallback) is deliberately outside the timed region.

Run (whole node, 4 Rubin GPUs)::

    torchrun --nproc_per_node=4 benchmarks/bench_moe_ep_sm107_block_scaled_mega.py \
        --routing balanced --tokens 1024,2048,4096,8192,16384,32768

Results are appended as JSON lines to ``--output`` (rank 0 only) and printed
as a markdown table row per problem size. See
``flashinfer/moe_ep/kernel_src/next_cutedsl_megamoe/TUNING.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

# DSv4 Pro EP4 problem (matches the upstream perf report).
HIDDEN = 7168
INTERMEDIATE = 3072
NUM_EXPERTS = 384
TOP_K = 6
WEIGHT_CHUNK_EXPERTS = 8  # bf16 generation + quantize peak-memory bound
SEED = 20260817

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

# Upstream reference latencies (us) for the delta column, Rubin TS4B @ 47881ad2.
UPSTREAM_US = {
    ("balanced", 1024): 372.22,
    ("balanced", 2048): 410.48,
    ("balanced", 4096): 529.56,
    ("balanced", 8192): 800.48,
    ("balanced", 16384): 1484.16,
    ("balanced", 32768): 2960.92,
    ("power_law", 1024): 399.75,
    ("power_law", 2048): 474.56,
    ("power_law", 4096): 621.76,
    ("power_law", 8192): 1053.01,
    ("power_law", 16384): 2081.84,
    ("power_law", 32768): 4023.79,
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


def _make_routing(world, tokens, routing, alpha):
    generator = torch.Generator(device="cuda").manual_seed(SEED + tokens)
    if routing == "balanced":
        topk_idx = _topk_idx_balanced(
            generator, world, tokens, TOP_K, NUM_EXPERTS, "cuda"
        )
    else:
        topk_idx = _topk_idx_power_law(
            generator, world, tokens, TOP_K, NUM_EXPERTS, alpha, "cuda"
        )
    topk_weights = (
        torch.rand((world, tokens, TOP_K), device="cuda", generator=generator) + 0.5
    )
    return topk_idx.to(torch.int32), topk_weights


def _local_transformed_weights(rank: int, world: int, quant_kind: str):
    """Random local expert slice, quantized chunk-wise to bound peak memory."""
    from flashinfer.moe_ep import MoEWeightPack

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
    fc1_parts, fc1_sf_parts, fc2_parts, fc2_sf_parts = [], [], [], []
    for begin in range(0, experts_per_rank, WEIGHT_CHUNK_EXPERTS):
        count = min(WEIGHT_CHUNK_EXPERTS, experts_per_rank - begin)
        w13 = (
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
        w2 = (
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
        (fc1_w, fc1_sf), (fc2_w, fc2_sf) = weights_mod.preprocess_mega_weights(
            MoEWeightPack(w13=w13, w2=w2),
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            **extra,
        )
        fc1_parts.append(fc1_w)
        fc1_sf_parts.append(fc1_sf.reshape(count, -1))
        fc2_parts.append(fc2_w)
        fc2_sf_parts.append(fc2_sf.reshape(count, -1))
        del w13, w2
        torch.cuda.empty_cache()
    return (
        (torch.cat(fc1_parts), torch.cat(fc1_sf_parts)),
        (torch.cat(fc2_parts), torch.cat(fc2_sf_parts)),
    )


def _bench_one(
    rank: int,
    world: int,
    tokens: int,
    routing: str,
    alpha: float,
    transformed,
    warmup: int,
    iters: int,
    quant_kind: str = "nvfp4",
    knobs_override: Optional[dict] = None,
) -> Tuple[dict, list]:
    import flashinfer.moe_ep.kernel_src.next_cutedsl_megamoe as pkg
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    knobs = dict(WINNERS[(routing, tokens)])
    if knobs_override:
        knobs.update(knobs_override)
    # WINNERS tiles are the nvfp4 selections (2x-mode instruction K 128, tile
    # K 256). mxfp8's 2x-mode instruction K is 64, so the analogous tile K is
    # 128; M/N and the scheduler knobs carry over unchanged.
    tile = tuple(knobs["tile"])
    if quant_kind != "nvfp4":
        tile = (tile[0], tile[1], 128)
        knobs["tile"] = tile

    common = dict(
        intermediate_size=INTERMEDIATE,
        top_k=TOP_K,
        in_kernel_fc2_reduce=False,  # separate top-k reduction (upstream winners)
        schedule_policy=("phase_interleave", knobs["hint"]),
        work_id_mode="atomic_counter",
        fc2_use_bulk=True,
        fc2_tma_stages=2,
        epi_flag_batches=tuple(knobs["epi"]),
        token_in_flag_batch=knobs["tif"],
        mma_tiler_mnk=tile,
        cluster_shape_mn=(4, 1),
        fallback_cluster_shape_mn=(2, 1),
    )
    if quant_kind == "nvfp4":
        cfg = Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**common)
    else:
        cfg = Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(kind=quant_kind, **common)
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world, rank=rank, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=tokens,
            token_hidden_size=HIDDEN,
        ),
        weights=None,
        backend=MegaConfig(megakernel=cfg, transformed_weights=transformed),
    )
    try:
        gen = torch.Generator(device="cuda").manual_seed(SEED + 13 * rank + tokens)
        x = torch.randn(
            tokens, HIDDEN, device="cuda", dtype=torch.float32, generator=gen
        ).to(torch.bfloat16)
        topk_idx, topk_weights = _make_routing(world, tokens, routing, alpha)

        # First forward: stages inputs, compiles, and validates the pipeline.
        dist.barrier()
        y = layer.forward(
            MoEEpTensors(
                hidden_states=x,
                topk_ids=topk_idx[rank],
                topk_weights=topk_weights[rank],
            )
        )
        if not torch.isfinite(y.to(torch.float32)).all():
            raise RuntimeError("non-finite output from warmup forward")

        # Steady-state: relaunch the fused kernel over the staged inputs.
        thunk = pkg.sm107_block_scaled_mega_launch_thunk(
            layer._transformed[0], layer._transformed[1], layer._workspace
        )
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(warmup):
            thunk()
        torch.cuda.synchronize()
        dist.barrier()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            thunk()
            stops[i].record()
        torch.cuda.synchronize()
        samples_us = [starts[i].elapsed_time(stops[i]) * 1000.0 for i in range(iters)]
    finally:
        layer.destroy()

    avg = sum(samples_us) / len(samples_us)
    return (
        dict(
            routing=routing,
            tokens=tokens,
            rank=rank,
            avg_us=avg,
            min_us=min(samples_us),
            max_us=max(samples_us),
            knobs=knobs,
        ),
        samples_us,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument(
        "--quant-kind", default="nvfp4", choices=["nvfp4", "mxfp8_e4m3", "both"]
    )
    parser.add_argument(
        "--routing", default="balanced", choices=["balanced", "power_law", "both"]
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", default="bench_sm107_mega_results.jsonl")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        raise SystemExit("run under torchrun with >= 2 ranks (EP benchmark)")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.runtime import sm107_block_scaled_runtime_requirements

    bootstrap = BootstrapConfig(world_size=world, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, sm107_block_scaled_runtime_requirements(bootstrap)
    )

    token_list = [int(t) for t in args.tokens.split(",")]
    routings = ["balanced", "power_law"] if args.routing == "both" else [args.routing]
    kinds = ["nvfp4", "mxfp8_e4m3"] if args.quant_kind == "both" else [args.quant_kind]
    try:
        for quant_kind in kinds:
            transformed = _local_transformed_weights(rank, world, quant_kind)
            if rank == 0:
                print(
                    f"# sm107 {quant_kind} mega EP{world}: hidden={HIDDEN} "
                    f"inter={INTERMEDIATE} experts={NUM_EXPERTS} topk={TOP_K} "
                    f"warmup={args.warmup} iters={args.iters}",
                    flush=True,
                )
            for routing in routings:
                for tokens in token_list:
                    result, samples = _bench_one(
                        rank,
                        world,
                        tokens,
                        routing,
                        args.alpha,
                        transformed,
                        args.warmup,
                        args.iters,
                        quant_kind=quant_kind,
                    )
                    # Aggregate: mean of rank averages; min/max over every sample.
                    stats = torch.tensor(
                        [result["avg_us"], result["min_us"], result["max_us"]],
                        device="cuda",
                    )
                    gathered = [torch.zeros_like(stats) for _ in range(world)]
                    dist.all_gather(gathered, stats)
                    if rank == 0:
                        avg = sum(g[0].item() for g in gathered) / world
                        lo = min(g[1].item() for g in gathered)
                        hi = max(g[2].item() for g in gathered)
                        flops = tokens * TOP_K * 6 * HIDDEN * INTERMEDIATE
                        tflops = (
                            flops / (avg * 1e-6) / 1e12
                            if routing == "balanced"
                            else None
                        )
                        # The upstream report is NVFP4-only; mxfp8 has no baseline.
                        ref = (
                            UPSTREAM_US.get((routing, tokens))
                            if quant_kind == "nvfp4"
                            else None
                        )
                        delta = f"{(avg - ref) / ref * 100.0:+.2f}%" if ref else "n/a"
                        knobs = result["knobs"]
                        tflops_col = f"{tflops:.1f}" if tflops is not None else "-"
                        ref_col = f"{ref:.2f}" if ref is not None else "-"
                        detail = (
                            f"tile {'x'.join(map(str, knobs['tile']))}; "
                            f"hint {knobs['hint']}; "
                            f"epi {knobs['epi'][0]}x{knobs['epi'][1]}; "
                            f"tif {knobs['tif']}"
                        )
                        print(
                            f"| {quant_kind} | {routing} | {tokens} | {avg:.2f} | "
                            f"{lo:.2f}-{hi:.2f} | "
                            f"{tflops_col} | {ref_col} | {delta} | {detail} |",
                            flush=True,
                        )
                        with open(args.output, "a") as fh:
                            fh.write(
                                json.dumps(
                                    dict(
                                        result,
                                        quant_kind=quant_kind,
                                        rank_avgs=[g[0].item() for g in gathered],
                                        mean_us=avg,
                                        min_us=lo,
                                        max_us=hi,
                                        tflops=tflops,
                                        upstream_us=ref,
                                    )
                                )
                                + "\n"
                            )
                    dist.barrier()
            del transformed
            torch.cuda.empty_cache()
    finally:
        finalize_moe_ep_runtime(runtime)


if __name__ == "__main__":
    main()
