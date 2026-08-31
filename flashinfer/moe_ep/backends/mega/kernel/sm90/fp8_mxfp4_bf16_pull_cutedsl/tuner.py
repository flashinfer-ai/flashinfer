# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Offline tuner for the SM90 Humming MXFP4 x FP8 MegaMoE backend.

Both execution modes reuse the production weight preprocessor and
activation/routing staging code. Its deterministic data recipe matches the
canonical MXFP4 workload in ``benchmarks/bench_moe_ep_sm90_mega.py``. The
selected canonical routing profile is forwarded to candidate selection,
session construction, and the collective tuner; this module only owns CLI
orchestration and does not introduce a second benchmark/timer.

Green-Context split sessions bind compiled kernels and graph executables to
fixed workspace pointers, so their dedicated autotune wrapper constructs a
fresh split session per candidate and commits the winner.
"""

from __future__ import annotations

from typing import Any, Literal

from ...tuning import finish_sweep, run_tuning as _run_tuning
from ......sm90_routing import (
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    generate_sm90_routing_numpy,
    normalize_sm90_routing_profile,
    sm90_benchmark_mode_from_routing_profile,
    sm90_route_ids_sha256,
)


_MXFP4_SCALE_MODE: Literal["mxfp4_hybrid"] = "mxfp4_hybrid"
_MXFP4_WEIGHT_SEED = 0x4D584650
_ACTIVATION_SEED = 42
_ROUTING_SEED = 1234
_MXFP4_E8M0_MIN = 118
_MXFP4_E8M0_MAX_EXCLUSIVE = 124


def _balanced_routing(
    num_tokens: int,
    topk: int,
    num_experts: int,
    rank: int,
    world_size: int,
    device: Any,
    *,
    seed: int,
    routing_profile: str = SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
):
    """Return this rank's IDs from one canonical global routing profile."""

    import torch

    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("rank/world_size are inconsistent")
    routes = generate_sm90_routing_numpy(
        routing_profile=normalize_sm90_routing_profile(routing_profile),
        world_size=world_size,
        tokens=num_tokens,
        topk=topk,
        total_experts=num_experts,
        seed=seed,
    )
    return torch.from_numpy(routes[rank].astype("int64")).to(device)


def _raw_mxfp4_shapes(
    *, local_experts: int, hidden: int, intermediate: int
) -> dict[str, tuple[int, int, int]]:
    if local_experts <= 0:
        raise ValueError("local_experts must be positive")
    if hidden <= 0 or hidden % 128:
        raise ValueError("MXFP4 hidden must be a positive multiple of 128")
    if intermediate <= 0 or intermediate % 128:
        raise ValueError("MXFP4 intermediate must be a positive multiple of 128")
    return {
        "w13": (local_experts, 2 * intermediate, hidden // 2),
        "w13_scale": (local_experts, 2 * intermediate, hidden // 32),
        "w2": (local_experts, hidden, intermediate // 2),
        "w2_scale": (local_experts, hidden, intermediate // 32),
    }


def _create_canonical_inputs(
    args,
    rank: int,
    world_size: int,
    max_tokens: int,
    live_tokens: int,
    execution_mode: str,
    initial_tactic: dict[str, Any],
):
    """Build canonical raw weights, preprocess them, and stage live inputs."""
    import torch

    from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
        get_symm_buffer_for_hopper_mxfp4_mega_moe,
        get_symm_buffer_for_hopper_mxfp4_split_mega_moe,
    )
    from ......weights import PrequantizedMoEWeights
    from .staging import stage_mega_moe_inputs
    from .weights import preprocess_mega_weights

    if live_tokens < 0 or live_tokens > max_tokens:
        raise ValueError(f"live_tokens must be in [0, {max_tokens}], got {live_tokens}")
    if execution_mode not in ("fused", "split"):
        raise ValueError(f"unsupported MXFP4 execution mode {execution_mode!r}")
    if args.num_experts <= 0 or args.num_experts % world_size:
        raise ValueError("--num-experts must be positive and divisible by world size")

    device = torch.device("cuda", torch.cuda.current_device())
    local_experts = args.num_experts // world_size
    shapes = _raw_mxfp4_shapes(
        local_experts=local_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
    )
    weight_gen = torch.Generator(device=device).manual_seed(
        _MXFP4_WEIGHT_SEED + args.seed + rank
    )

    def _payload(name: str) -> torch.Tensor:
        return torch.randint(
            0,
            256,
            shapes[name],
            dtype=torch.uint8,
            device=device,
            generator=weight_gen,
        )

    def _exponent(name: str) -> torch.Tensor:
        return torch.randint(
            _MXFP4_E8M0_MIN,
            _MXFP4_E8M0_MAX_EXCLUSIVE,
            shapes[name],
            dtype=torch.uint8,
            device=device,
            generator=weight_gen,
        )

    raw = PrequantizedMoEWeights(
        w13=_payload("w13"),
        w2=_payload("w2"),
        w13_scale=_exponent("w13_scale"),
        w2_scale=_exponent("w2_scale"),
    )
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        raw,
        intermediate_size=args.intermediate,
        hidden_size=args.hidden,
    )
    del raw

    symm_buffer: Any = None
    try:
        # Explicit manifest tactics keep stale cache contents out of the
        # offline sweep. Split requires the complete tactic because geometry,
        # SM partition, graph variant and counter banks are session identity.
        common = (
            args.num_experts,
            max_tokens,
            args.topk,
            args.hidden,
            args.intermediate,
            rank,
            world_size,
        )
        if execution_mode == "fused":
            symm_buffer = get_symm_buffer_for_hopper_mxfp4_mega_moe(
                *common,
                fp8_scale_mode=_MXFP4_SCALE_MODE,
                knobs=initial_tactic,
                gate_up_clamp=args.gate_up_clamp,
                routing_profile=args.routing_profile,
            )
        else:
            symm_buffer = get_symm_buffer_for_hopper_mxfp4_split_mega_moe(
                *common,
                split_k1_mma_tiler_mnk=initial_tactic["k1_mma_tiler_mnk"],
                split_k2_mma_tiler_mnk=initial_tactic["k2_mma_tiler_mnk"],
                split_k1_cluster_shape_mnk=initial_tactic["k1_cluster_shape_mnk"],
                split_k2_cluster_shape_mnk=initial_tactic["k2_cluster_shape_mnk"],
                split_k1_group_hint=initial_tactic["k1_group_hint"],
                split_k2_group_hint=initial_tactic["k2_group_hint"],
                split_k1_num_sched_stages=initial_tactic["k1_num_sched_stages"],
                split_k2_num_sched_stages=initial_tactic["k2_num_sched_stages"],
                split_k1_sm_count=initial_tactic["k1_sm_count"],
                split_k2_sm_count=initial_tactic["k2_sm_count"],
                split_counter_epoch_banks=initial_tactic["counter_epoch_banks"],
                split_graph_variant=initial_tactic["graph_variant"],
                gate_up_clamp=args.gate_up_clamp,
                split_enable_iket=initial_tactic["enable_iket"],
                routing_profile=args.routing_profile,
            )

        activation_gen = torch.Generator(device=device).manual_seed(
            _ACTIVATION_SEED + args.seed + rank
        )
        hidden_states = torch.randn(
            live_tokens,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=activation_gen,
        )
        topk_idx = _balanced_routing(
            live_tokens,
            args.topk,
            args.num_experts,
            rank,
            world_size,
            device,
            seed=_ROUTING_SEED + args.seed,
            routing_profile=args.routing_profile,
        )
        topk_weights = torch.softmax(
            torch.randn(
                live_tokens,
                args.topk,
                dtype=torch.float32,
                device=device,
                generator=activation_gen,
            ),
            dim=-1,
        )
        stage_mega_moe_inputs(
            hidden_states,
            topk_weights,
            topk_idx,
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            quantize_input=True,
        )
        y = torch.empty(live_tokens, args.hidden, dtype=torch.bfloat16, device=device)
        return y, transformed_l1, transformed_l2, symm_buffer
    except BaseException:
        if symm_buffer is not None:
            symm_buffer.destroy()
        raise


def _candidate_union(
    pkg: Any, *, execution_mode: str, routing_profile: str
) -> list[dict[str, Any]]:
    """Read the only candidate source accepted by the MXFP4 CLI."""
    profile = normalize_sm90_routing_profile(routing_profile)
    candidates = pkg.hopper_mxfp4_candidates(
        execution_mode=execution_mode,
        routing_profile=profile,
    )
    if not candidates:
        raise RuntimeError(
            f"empty manifest-derived MXFP4 {execution_mode} candidate union"
        )
    return candidates


def _ordered_candidates(
    pkg: Any,
    *,
    execution_mode: str,
    max_tokens: int,
    hidden: int,
    intermediate: int,
    routing_profile: str,
) -> list[dict[str, Any]]:
    """Filter to legal tactics, preferring the bucket winner when legal."""
    profile = normalize_sm90_routing_profile(routing_profile)
    candidates = _candidate_union(
        pkg,
        execution_mode=execution_mode,
        routing_profile=profile,
    )
    default = pkg.hopper_mxfp4_default_tactic(
        max_tokens,
        execution_mode=execution_mode,
        routing_profile=profile,
    )
    if default not in candidates:
        raise RuntimeError(
            f"MXFP4 {execution_mode} default is absent from candidate union"
        )
    legal = [
        candidate
        for candidate in candidates
        if pkg.is_hopper_mxfp4_tactic_shape_compatible(
            candidate,
            execution_mode=execution_mode,
            hidden=hidden,
            intermediate=intermediate,
        )
    ]
    if not legal:
        raise RuntimeError(
            f"no manifest-derived MXFP4 {execution_mode} candidate supports "
            f"hidden={hidden}, intermediate={intermediate}"
        )
    if default not in legal:
        return legal
    return [default, *(candidate for candidate in legal if candidate != default)]


def tune_one(args, rank: int, world_size: int, max_tokens: int) -> dict:
    from ......kernel_src.sm90 import pull_style_cutedsl_megakernel as pkg
    from ......kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
        require_hopper_mxfp4_tuning_device,
    )

    # Embedded winners/unions are certified on standard 132-SM H200 only.
    # Explicit user tactics remain a separate production path.
    require_hopper_mxfp4_tuning_device()

    if args.seed != 0:
        raise SystemExit("SM90 MXFP4 tuning requires the canonical --seed 0")
    routing_profile = normalize_sm90_routing_profile(args.routing_profile)

    live_tokens = args.live_tokens if args.live_tokens is not None else max_tokens
    if live_tokens > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")

    mode = args.execution_mode
    candidates = _ordered_candidates(
        pkg,
        execution_mode=mode,
        max_tokens=max_tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        routing_profile=routing_profile,
    )
    symm_buffer: Any = None
    try:
        y, l1, l2, symm_buffer = _create_canonical_inputs(
            args,
            rank,
            world_size,
            max_tokens,
            live_tokens,
            mode,
            candidates[0],
        )
        if rank == 0:
            route_ids_sha256 = sm90_route_ids_sha256(
                generate_sm90_routing_numpy(
                    routing_profile=routing_profile,
                    world_size=world_size,
                    tokens=live_tokens,
                    topk=args.topk,
                    total_experts=args.num_experts,
                    seed=_ROUTING_SEED + args.seed,
                )
            )
            print(
                f"[moe_ep-tune] sm90_mxfp4 {mode} canonical data: "
                f"weight_seed=0x{_MXFP4_WEIGHT_SEED + args.seed:x}+rank "
                f"activation_seed={_ACTIVATION_SEED + args.seed}+rank "
                f"routing_seed={_ROUTING_SEED + args.seed} "
                f"routing_profile={routing_profile} "
                "routing_mode="
                f"{sm90_benchmark_mode_from_routing_profile(routing_profile)} "
                f"route_ids_sha256={route_ids_sha256}",
                flush=True,
            )
        tune_fn = (
            pkg.autotune_hopper_mxfp4_mega_moe
            if mode == "fused"
            else pkg.autotune_hopper_mxfp4_split_mega_moe
        )
        return finish_sweep(
            args,
            rank,
            max_tokens,
            live_tokens,
            symm_buffer,
            y,
            l1,
            l2,
            candidates,
            tune_fn,
            tune_kwargs={
                "gate_up_clamp": args.gate_up_clamp,
                "routing_profile": routing_profile,
            },
        )
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()


def run_tuning(args) -> int:
    from ......kernel_src.sm90 import pull_style_cutedsl_megakernel as pkg

    return _run_tuning(args, tune_one, pkg=pkg)


__all__ = ["run_tuning", "tune_one"]
