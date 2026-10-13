#!/usr/bin/env python3
"""Benchmark one-grid mixed K3/K4 Trellis against two serial tier launches."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import replace

import torch

from b12x.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    MixedTrellisRotations,
    build_ordered_maps,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    run_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


HIDDEN = 6144
INTERMEDIATE = 512
TOPK = 8
K3_EXPERTS = 192
K4_EXPERTS = 64
TOTAL_EXPERTS = K3_EXPERTS + K4_EXPERTS
TILE_CONFIG = (128, 128, 32, 512)


def _prepared(bits: int, experts: int, seed: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).to(torch.float16)

    return prepare_trellis256_moe_weights(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=TILE_CONFIG[1],
        fc2_tile_n=TILE_CONFIG[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        gate_suh=scales((experts, HIDDEN)),
        up_suh=scales((experts, HIDDEN)),
        intermediate_rotations=scales((experts, 3 * INTERMEDIATE)),
        down_svh=scales((experts, HIDDEN)),
        tile_config=TILE_CONFIG,
    )


def _routing(m: int, materialized: int, device: torch.device) -> torch.Tensor:
    rows = []
    generator = torch.Generator(device="cpu").manual_seed(20260730 + m)
    for _ in range(m):
        k3 = torch.randperm(materialized, generator=generator)[:6]
        k4 = K3_EXPERTS + torch.randperm(materialized, generator=generator)[:2]
        rows.append(torch.cat((k3, k4))[torch.randperm(TOPK, generator=generator)])
    return torch.stack(rows).to(dtype=torch.int32, device=device)


def _serial_state(m, tier, expert_map, props, device):
    buffers = make_w4a16_packed_buffers(
        tier,
        m=m,
        topk=TOPK,
        dtype=torch.float16,
        device=device,
        route_num_experts=TOTAL_EXPERTS,
        full_rotation=True,
        block_size_m=8,
    )
    launch = compile_w4a16_fused_moe(
        size_m=m,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=int(tier.num_experts),
        top_k=TOPK,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=int(buffers.block_expert_ids.numel()),
        element_dtype="fp16",
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj",
        trellis_bits=int(tier.trellis_bits),
        force_tile_config=TILE_CONFIG,
        intermediate_rotation=True,
        full_rotation=True,
        rotation_input_dtype="bf16",
    )
    topk_sum = compile_w4a16_topk_sum(
        m=m,
        topk=TOPK,
        hidden_size=HIDDEN,
        element_dtype="fp16",
        full_rotation=True,
        num_experts=int(tier.num_experts),
        route_num_experts=TOTAL_EXPERTS,
        route_ids_dtype=torch.int32,
        use_expert_map=True,
    )
    return buffers, launch, topk_sum, expert_map


def _run_serial(state, x, weights, ids):
    tier, buffers, launch, topk_sum, expert_map = state
    return run_w4a16_moe(
        x,
        tier,
        weights,
        ids,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        expert_map=expert_map,
        output_expert_map=expert_map,
        fused_launch=launch,
        topk_sum_launch=topk_sum,
        route_block_size_m=8,
        intermediate_rotation_scales=tier.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=tier.gate_suh,
        suh_up_table=tier.up_suh,
        svh_table=tier.down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    )


def _capture(fn, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _times(graph: torch.cuda.CUDAGraph, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def _bytes(tensors) -> int:
    storages: dict[tuple[int, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def _padded_rotations(tier0, tier1, device: torch.device) -> MixedTrellisRotations:
    intermediate = torch.ones(
        (TOTAL_EXPERTS, 3 * INTERMEDIATE), dtype=torch.float16, device=device
    )
    gate_suh = torch.ones((TOTAL_EXPERTS, HIDDEN), dtype=torch.float16, device=device)
    up_suh = torch.ones_like(gate_suh)
    down_svh = torch.ones_like(gate_suh)
    k3 = int(tier0.num_experts)
    k4 = int(tier1.num_experts)
    intermediate[:k3].copy_(tier0.intermediate_rotations)
    intermediate[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.intermediate_rotations)
    gate_suh[:k3].copy_(tier0.gate_suh)
    gate_suh[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.gate_suh)
    up_suh[:k3].copy_(tier0.up_suh)
    up_suh[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.up_suh)
    down_svh[:k3].copy_(tier0.down_svh)
    down_svh[K3_EXPERTS : K3_EXPERTS + k4].copy_(tier1.down_svh)
    return MixedTrellisRotations(intermediate, gate_suh, up_suh, down_svh)


def _pad_projection_major_w13(tier, compiled_experts: int):
    """Preserve the [projection, expert, ...] plane stride used by codegen."""
    materialized = int(tier.num_experts)
    bits = int(tier.trellis_bits)
    source = tier.w13.view(
        2,
        materialized,
        HIDDEN // 16,
        INTERMEDIATE // 16,
        8 * bits,
    )
    padded = torch.zeros(
        (
            2,
            int(compiled_experts),
            HIDDEN // 16,
            INTERMEDIATE // 16,
            8 * bits,
        ),
        dtype=torch.int32,
        device=tier.w13.device,
    )
    padded[:, :materialized].copy_(source)
    return replace(tier, w13=padded.view(-1), num_experts=int(compiled_experts))


def _pad_exact_tier_storage(tier, compiled_experts: int):
    """Materialize the full expert-sized storage contract used by ABI 5."""
    materialized = int(tier.num_experts)
    bits = int(tier.trellis_bits)
    padded = _pad_projection_major_w13(tier, compiled_experts)
    w2 = torch.zeros(
        (
            int(compiled_experts),
            INTERMEDIATE // 16,
            HIDDEN // 16,
            8 * bits,
        ),
        dtype=torch.int32,
        device=tier.w2.device,
    )
    w2[:materialized].copy_(tier.w2.view(materialized, *w2.shape[1:]))

    def pad_global(source: torch.Tensor) -> torch.Tensor:
        output = torch.ones(
            int(compiled_experts), dtype=source.dtype, device=source.device
        )
        output[:materialized].copy_(source)
        return output

    return replace(
        padded,
        w2=w2.view(-1),
        w13_global_scale=pad_global(tier.w13_global_scale),
        w2_global_scale=pad_global(tier.w2_global_scale),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--materialized-experts", type=int, default=16)
    parser.add_argument(
        "--variant",
        choices=("both", "serial", "serial-overlap", "mixed"),
        default="both",
    )
    parser.add_argument(
        "--separate-fc2",
        action="store_true",
        help="A/B control: replace the aliased FC2 output with separate storage",
    )
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    torch.manual_seed(20260730)
    materialized = int(args.materialized_experts)
    if materialized < 6 or materialized > K4_EXPERTS:
        raise ValueError(f"materialized-experts must be in [6, {K4_EXPERTS}]")
    tier0 = _prepared(3, materialized, 301, device)
    tier1 = _prepared(4, materialized, 401, device)
    mixed_tier0 = mixed_tier1 = rotations = None
    if args.variant in ("both", "mixed"):
        mixed_tier0 = _pad_exact_tier_storage(tier0, K3_EXPERTS)
        mixed_tier1 = _pad_exact_tier_storage(tier1, K4_EXPERTS)
        rotations = _padded_rotations(tier0, tier1, device)
    global_map, descriptor = build_ordered_maps(K3_EXPERTS, K4_EXPERTS, device=device)
    map0 = torch.cat(
        (
            torch.arange(materialized, dtype=torch.int32, device=device),
            torch.full(
                (TOTAL_EXPERTS - materialized,), -1, dtype=torch.int32, device=device
            ),
        )
    )
    map1 = torch.cat(
        (
            torch.full((K3_EXPERTS,), -1, dtype=torch.int32, device=device),
            torch.arange(materialized, dtype=torch.int32, device=device),
            torch.full(
                (K4_EXPERTS - materialized,), -1, dtype=torch.int32, device=device
            ),
        )
    )
    print(
        f"device={props.name} H={HIDDEN} I={INTERMEDIATE} topk={TOPK} "
        f"experts={K3_EXPERTS}xK3+{K4_EXPERTS}xK4"
    )
    for m in args.m:
        ids = _routing(m, materialized, device)
        weights = torch.softmax(
            torch.randn((m, TOPK), dtype=torch.float32, device=device), dim=-1
        )
        x = (torch.randn((m, HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)
        state0 = state1 = serial_output = serial_fn = None
        serial_med = serial_buffer_bytes = None
        if args.variant in ("both", "serial", "serial-overlap"):
            state0 = (tier0, *_serial_state(m, tier0, map0, props, device))
            state1 = (tier1, *_serial_state(m, tier1, map1, props, device))
            serial_output = torch.empty((m, HIDDEN), dtype=torch.float32, device=device)
            tier_streams = None
            if args.variant == "serial-overlap":
                tier_streams = (torch.cuda.Stream(), torch.cuda.Stream())

            def serial_fn(
                state0=state0,
                state1=state1,
                serial_output=serial_output,
                tier_streams=tier_streams,
                x=x,
                weights=weights,
                ids=ids,
            ):
                assert state0 is not None and state1 is not None
                assert serial_output is not None
                if tier_streams is None:
                    out0 = _run_serial(state0, x, weights, ids)
                    out1 = _run_serial(state1, x, weights, ids)
                else:
                    current = torch.cuda.current_stream()
                    tier_streams[0].wait_stream(current)
                    tier_streams[1].wait_stream(current)
                    with torch.cuda.stream(tier_streams[0]):
                        out0 = _run_serial(state0, x, weights, ids)
                    with torch.cuda.stream(tier_streams[1]):
                        out1 = _run_serial(state1, x, weights, ids)
                    current.wait_stream(tier_streams[0])
                    current.wait_stream(tier_streams[1])
                torch.add(out0, out1, out=serial_output)

            serial_fn()
            serial_graph = _capture(serial_fn, args.warmup)
            serial_med = statistics.median(_times(serial_graph, args.replays))
            serial_buffer_bytes = _bytes(vars(state0[1]).values()) + _bytes(
                vars(state1[1]).values()
            )

        launch = mixed_buffers = mixed_fn = None
        mixed_med = mixed_buffer_bytes = None
        if args.variant in ("both", "mixed"):
            route_slots = max_packed_route_slots(m * TOPK, 8, TOTAL_EXPERTS)
            launch = compile_mixed_trellis(
                size_m=m,
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                tier0_num_experts=K3_EXPERTS,
                tier1_num_experts=K4_EXPERTS,
                top_k=TOPK,
                max_m_blocks=(route_slots + 7) // 8,
                sms=int(props.multi_processor_count),
                max_shared_mem=int(props.shared_memory_per_block_optin),
                force_tile_config=TILE_CONFIG,
            )
            mixed_buffers = make_mixed_trellis_buffers(
                launch, device=device, sms=int(props.multi_processor_count)
            )
            if args.separate_fc2:
                mixed_buffers.fc2 = torch.empty_like(mixed_buffers.rotation_gate)

            def mixed_fn(
                mixed_tier0=mixed_tier0,
                mixed_tier1=mixed_tier1,
                rotations=rotations,
                launch=launch,
                mixed_buffers=mixed_buffers,
                x=x,
                weights=weights,
                ids=ids,
                global_map=global_map,
                descriptor=descriptor,
            ):
                assert mixed_tier0 is not None and mixed_tier1 is not None
                assert rotations is not None and launch is not None
                assert mixed_buffers is not None
                run_mixed_trellis(
                    x,
                    mixed_tier0,
                    mixed_tier1,
                    weights,
                    ids,
                    global_map,
                    descriptor,
                    rotations,
                    launch,
                    mixed_buffers,
                )

            mixed_fn()
            mixed_graph = _capture(mixed_fn, args.warmup)
            mixed_med = statistics.median(_times(mixed_graph, args.replays))
            mixed_buffer_bytes = _bytes(vars(mixed_buffers).values())

        torch.cuda.synchronize()
        if args.variant == "both":
            assert serial_output is not None and mixed_buffers is not None
            assert serial_med is not None and mixed_med is not None
            assert serial_buffer_bytes is not None and mixed_buffer_bytes is not None
            assert launch is not None
            rel = (
                serial_output - mixed_buffers.output[:m]
            ).norm() / serial_output.norm().clamp_min(1.0e-12)
            cosine = torch.nn.functional.cosine_similarity(
                serial_output.flatten(), mixed_buffers.output[:m].flatten(), dim=0
            )
            if float(rel) > 4.0e-3 or float(cosine) < 0.99999:
                raise RuntimeError(
                    f"correctness failed: rel={float(rel)} cos={float(cosine)}"
                )
            print(
                f"m={m:4d} serial={serial_med:9.2f}us mixed={mixed_med:9.2f}us "
                f"speedup={serial_med / mixed_med:6.3f}x rel={float(rel):.3e} "
                f"buffers={mixed_buffer_bytes / 2**20:.1f}/{serial_buffer_bytes / 2**20:.1f}MiB"
            )
        elif args.variant in ("serial", "serial-overlap"):
            assert serial_med is not None and serial_buffer_bytes is not None
            print(
                f"m={m:4d} variant={args.variant} time={serial_med:9.2f}us "
                f"buffers={serial_buffer_bytes / 2**20:.1f}MiB"
            )
        else:
            assert mixed_med is not None and mixed_buffer_bytes is not None
            assert launch is not None
            print(
                f"m={m:4d} variant=mixed time={mixed_med:9.2f}us "
                f"buffers={mixed_buffer_bytes / 2**20:.1f}MiB"
            )


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
