"""Parity tests for the two-tier hybrid W4A16 fused MoE kernel.

The hybrid kernel runs one grid over a heterogeneous-quantization layer
(tier 0 NVFP4 "packed" + tier 1 NF3 "nf3_2p1") using a global-expert
descriptor map. Its oracle is the production serial path: one single-tier
TC-decode launch per tier with the other tier's routes masked to -1 (the
contract test_nf3_tc_decode_production_shape_with_tier_mask already covers),
summed on the host. Identical prepared weights feed both paths, so any
difference beyond FC2 atomic-accumulation ordering is a defect.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.sm12x._lib.intrinsics import swizzle_block_scale
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.kernel import (
    build_w4a16_tier_local_map,
    compile_w4a16_fused_moe,
    compile_w4a16_fused_moe_hybrid,
    run_w4a16_moe,
    run_w4a16_moe_hybrid,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.prepare import (
    prepare_nf3_moe_weights,
    prepare_w4a16_modelopt_nvfp4_weights,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
)
from .test_w4a16_nf3 import _round_to_e4m3_scale

_DEVICE = torch.device("cuda")
_DTYPE = torch.bfloat16
_DEFAULT_MAX_SHARED_MEM = 101_376

_HIDDEN = 6144
_INTERMEDIATE = 512
_TOPK = 8
_CAPACITY_M = 4
_TILE_CONFIG = (64, 256, 64, 256)
_T0_EXPERTS = 4
_T1_EXPERTS = 6
_MAP_SLOTS = _T0_EXPERTS + _T1_EXPERTS
_T0_GLOBAL_IDS = (0, 3, 5, 9)
_T1_GLOBAL_IDS = (1, 2, 4, 6, 7, 8)


def _device_limits() -> tuple[int, int]:
    props = torch.cuda.get_device_properties(_DEVICE)
    return int(props.multi_processor_count), int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    scale = 0.05 + 0.2 * torch.rand(shape, device=_DEVICE)
    return scale.to(torch.float8_e4m3fn)


def _build_tier0_nvfp4():
    w13_rows = 2 * _INTERMEDIATE
    w13 = torch.randint(
        0,
        256,
        (_T0_EXPERTS, w13_rows, _HIDDEN // 2),
        dtype=torch.uint8,
        device=_DEVICE,
    )
    w2 = torch.randint(
        0,
        256,
        (_T0_EXPERTS, _HIDDEN, _INTERMEDIATE // 2),
        dtype=torch.uint8,
        device=_DEVICE,
    )
    w13_blockscale = swizzle_block_scale(
        _positive_fp8((_T0_EXPERTS, w13_rows, _HIDDEN // 16))
    )
    w2_blockscale = swizzle_block_scale(
        _positive_fp8((_T0_EXPERTS, _HIDDEN, _INTERMEDIATE // 16))
    )
    w13_global = (torch.rand(_T0_EXPERTS, device=_DEVICE) * 0.1 + 0.05).float()
    w2_global = (torch.rand(_T0_EXPERTS, device=_DEVICE) * 0.1 + 0.05).float()
    return prepare_w4a16_modelopt_nvfp4_weights(
        w13,
        w13_blockscale,
        w13_global,
        w2,
        w2_blockscale,
        w2_global,
        activation="silu",
        params_dtype=_DTYPE,
    )


def _build_tier1_nf3():
    w13_rows = 2 * _INTERMEDIATE
    w13_codes = torch.randint(
        0, 8, (_T1_EXPERTS, w13_rows, _HIDDEN), dtype=torch.int32, device=_DEVICE
    )
    w2_codes = torch.randint(
        0,
        8,
        (_T1_EXPERTS, _HIDDEN, _INTERMEDIATE),
        dtype=torch.int32,
        device=_DEVICE,
    )
    w13_scale = _round_to_e4m3_scale(
        0.01 + 0.24 * torch.rand(_T1_EXPERTS, w13_rows, _HIDDEN // 32, device=_DEVICE)
    )
    w2_scale = _round_to_e4m3_scale(
        0.01
        + 0.24 * torch.rand(_T1_EXPERTS, _HIDDEN, _INTERMEDIATE // 32, device=_DEVICE)
    )
    return prepare_nf3_moe_weights(
        w13_codes,
        w13_scale,
        w2_codes,
        w2_scale,
        activation="silu",
        fc1_tile_n=_TILE_CONFIG[1],
        fc2_tile_n=_TILE_CONFIG[3],
        params_dtype=_DTYPE,
    )


def _tier_local_ids(global_ids: torch.Tensor, tier_globals: tuple[int, ...]):
    """Map global route ids to tier-local ids, other-tier/invalid routes -> -1."""
    lookup = {gid: local for local, gid in enumerate(tier_globals)}
    local = torch.full_like(global_ids, -1)
    for row in range(global_ids.shape[0]):
        for col in range(global_ids.shape[1]):
            gid = int(global_ids[row, col])
            if gid in lookup:
                local[row, col] = lookup[gid]
    return local


def _serial_reference(
    x: torch.Tensor,
    prepared_t0,
    prepared_t1,
    topk_weights: torch.Tensor,
    global_ids: torch.Tensor,
) -> torch.Tensor:
    sms, max_shared_mem = _device_limits()
    m = x.shape[0]
    out = torch.zeros((m, _HIDDEN), dtype=_DTYPE, device=_DEVICE)
    for prepared, tier_globals in (
        (prepared_t0, _T0_GLOBAL_IDS),
        (prepared_t1, _T1_GLOBAL_IDS),
    ):
        local_ids = _tier_local_ids(global_ids, tier_globals)
        fused = compile_w4a16_fused_moe(
            size_m=_CAPACITY_M,
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            num_experts=int(prepared.num_experts),
            top_k=_TOPK,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=_CAPACITY_M * _TOPK,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout=prepared.weight_layout,
            scale_format=prepared.scale_format,
            w13_layout=getattr(prepared, "w13_layout", "packed"),
            direct_topk_routes=True,
            tc_decode_fused_sum=True,
            force_tile_config=_TILE_CONFIG,
        )
        buffers = make_w4a16_packed_buffers(
            prepared, m=_CAPACITY_M, topk=_TOPK, dtype=_DTYPE, device=_DEVICE
        )
        tier_out = run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            local_ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output[:m],
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            fused_launch=fused,
        )
        torch.cuda.synchronize()
        out += tier_out
    return out


def _run_hybrid(
    x: torch.Tensor,
    prepared_t0,
    prepared_t1,
    topk_weights: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    schedule_whole_tiles: bool,
) -> torch.Tensor:
    tier_local_map = build_w4a16_tier_local_map(
        _T0_GLOBAL_IDS,
        _T1_GLOBAL_IDS,
        map_slots=_MAP_SLOTS,
        device=_DEVICE,
    )
    buffers = make_w4a16_packed_buffers(
        prepared_t0, m=_CAPACITY_M, topk=_TOPK, dtype=_DTYPE, device=_DEVICE
    )
    m = x.shape[0]
    output = torch.full((m, _HIDDEN), float("nan"), dtype=_DTYPE, device=_DEVICE)
    result = run_w4a16_moe_hybrid(
        x,
        prepared_t0,
        prepared_t1,
        topk_weights,
        global_ids,
        tier_local_map,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=output,
        force_tile_config=_TILE_CONFIG,
        size_m=_CAPACITY_M,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        schedule_whole_tiles=schedule_whole_tiles,
    )
    torch.cuda.synchronize()
    return result


def _assert_parity(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    got = actual.float()
    ref = expected.float()
    assert torch.isfinite(got).all(), f"{label}: nonfinite hybrid output"
    denom = ref.abs().amax().clamp(min=1e-6)
    rel = (got - ref).abs().amax() / denom
    assert rel < 2e-2, f"{label}: rel err {float(rel):.5f} exceeds 2e-2"


@pytest.fixture(scope="module")
def hybrid_problem():
    torch.manual_seed(20260718)
    prepared_t0 = _build_tier0_nvfp4()
    prepared_t1 = _build_tier1_nf3()
    x = torch.randn(_CAPACITY_M, _HIDDEN, dtype=_DTYPE, device=_DEVICE) * 0.1
    topk_weights = torch.softmax(
        torch.randn(_CAPACITY_M, _TOPK, device=_DEVICE), dim=-1
    )
    return prepared_t0, prepared_t1, x, topk_weights


def _global_ids_case(case: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(hash(case) % (2**31))
    if case == "mixed":
        ids = torch.randint(
            0, _MAP_SLOTS, (_CAPACITY_M, _TOPK), dtype=torch.int32, generator=generator
        )
    elif case == "all_tier0":
        choices = torch.tensor(_T0_GLOBAL_IDS, dtype=torch.int32)
        idx = torch.randint(
            0, len(_T0_GLOBAL_IDS), (_CAPACITY_M, _TOPK), generator=generator
        )
        ids = choices[idx]
    elif case == "all_tier1":
        choices = torch.tensor(_T1_GLOBAL_IDS, dtype=torch.int32)
        idx = torch.randint(
            0, len(_T1_GLOBAL_IDS), (_CAPACITY_M, _TOPK), generator=generator
        )
        ids = choices[idx]
    elif case == "with_invalid":
        ids = torch.randint(
            0, _MAP_SLOTS, (_CAPACITY_M, _TOPK), dtype=torch.int32, generator=generator
        )
        ids[0, 1] = -1
        ids[1, 5] = -1
        ids[3, 0] = -1
    else:
        raise AssertionError(case)
    return ids.to(dtype=torch.int32, device=_DEVICE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", ["mixed", "all_tier0", "all_tier1", "with_invalid"])
def test_hybrid_matches_serial_two_launch(case: str, hybrid_problem) -> None:
    prepared_t0, prepared_t1, x, topk_weights = hybrid_problem
    global_ids = _global_ids_case(case)
    expected = _serial_reference(x, prepared_t0, prepared_t1, topk_weights, global_ids)
    actual = _run_hybrid(
        x,
        prepared_t0,
        prepared_t1,
        topk_weights,
        global_ids,
        schedule_whole_tiles=True,
    )
    _assert_parity(actual, expected, f"hybrid[{case}]")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hybrid_split_k_schedule_matches_whole_tiles(hybrid_problem) -> None:
    """The split-K tail schedule and the whole-tiles schedule are numerically
    interchangeable through the same route-map emit hooks."""
    prepared_t0, prepared_t1, x, topk_weights = hybrid_problem
    global_ids = _global_ids_case("mixed")
    expected = _serial_reference(x, prepared_t0, prepared_t1, topk_weights, global_ids)
    actual = _run_hybrid(
        x,
        prepared_t0,
        prepared_t1,
        topk_weights,
        global_ids,
        schedule_whole_tiles=False,
    )
    _assert_parity(actual, expected, "hybrid[split_k]")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hybrid_partial_batch(hybrid_problem) -> None:
    """m < capacity: trailing route slots are ignored, output rows match."""
    prepared_t0, prepared_t1, x, topk_weights = hybrid_problem
    global_ids = _global_ids_case("mixed")
    m = 1
    expected = _serial_reference(
        x[:m], prepared_t0, prepared_t1, topk_weights[:m], global_ids[:m]
    )
    actual = _run_hybrid(
        x[:m],
        prepared_t0,
        prepared_t1,
        topk_weights[:m],
        global_ids[:m],
        schedule_whole_tiles=True,
    )
    _assert_parity(actual, expected, "hybrid[m=1]")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hybrid_glm_geometry_admits_without_spills() -> None:
    """GLM-5.2 TP4 shard geometry: the combined two-decoder kernel must fit
    without local-memory spills, or the decode win silently evaporates.

    The exact shared-memory pin documents the production geometry; it should
    only move with a deliberate smem layout change. Register/local assertions
    are skipped when the compile came from the on-disk object cache (no
    introspection surface) -- a fresh compile covers them.
    """
    sms, max_shared_mem = _device_limits()
    hybrid = compile_w4a16_fused_moe_hybrid(
        size_m=4,
        hidden_size=6144,
        intermediate_size=512,
        tier0_num_experts=64,
        tier1_num_experts=192,
        top_k=8,
        activation="silu",
        map_slots=256,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=(64, 256, 64, 256),
    )
    assert hybrid.shared_memory_bytes == 45_184
    assert hybrid.blocks_per_sm == 1
    assert hybrid.cta_threads == 256
    if hybrid.registers_per_thread >= 0:
        assert 1 <= hybrid.registers_per_thread <= 255
        assert hybrid.local_memory_bytes == 0


def test_tier_local_map_builder_rejects_bad_maps() -> None:
    with pytest.raises(ValueError, match="mapped twice"):
        build_w4a16_tier_local_map((0, 1), (1, 2), map_slots=4)
    with pytest.raises(ValueError, match="outside"):
        build_w4a16_tier_local_map((0, 5), (1,), map_slots=4)
    table = build_w4a16_tier_local_map(_T0_GLOBAL_IDS, _T1_GLOBAL_IDS, map_slots=16)
    assert table.numel() == 16
    assert int(table[0]) == 0
    assert int(table[1]) == (1 << 8) | 0
    assert int(table[9]) == 3
    assert int(table[10]) == -1
