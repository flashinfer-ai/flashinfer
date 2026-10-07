"""Host-only selection rules for prepared fused-MoE variants and W4A16 launches."""

from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import pytest
import torch

from b12x.moe.fused_moe._preparation import (
    _FusedMoeCapacityState,
    _W4A16PrimaryLaunches,
    variant_for,
    _control_snapshot,
)


@pytest.mark.parametrize("mode", ["auto", "compact", "full"])
def test_trellis_decode_table_is_frozen_at_declaration(monkeypatch, mode):
    monkeypatch.setenv("B12X_TRELLIS_DECODE_TABLE", mode)
    controls = _control_snapshot()
    monkeypatch.setenv("B12X_TRELLIS_DECODE_TABLE", "invalid")
    assert controls["trellis_decode_table"] == mode
    with pytest.raises(ValueError, match="B12X_TRELLIS_DECODE_TABLE"):
        _control_snapshot()


@pytest.mark.parametrize(
    "rows,mode,expected,shared_limit",
    [
        (1, "auto", True, 101376),
        (8, "auto", True, 101376),
        (16, "auto", True, 101376),
        (17, "auto", False, 101376),
        (8, "compact", False, 101376),
        (8, "full", True, 101376),
        (8, "auto", False, 65536),
    ],
)
def test_trellis_direct_table_preserves_cooperative_residency(
    monkeypatch,
    rows,
    mode,
    expected,
    shared_limit,
):
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            multi_processor_count=188,
            shared_memory_per_block_optin=shared_limit,
        ),
    )
    monkeypatch.setenv("B12X_LUT_E4M3_SMEM", "1")
    kernel = W4A16FusedMoeKernel(
        size_m=rows,
        hidden_size=3584,
        intermediate_size=384,
        num_experts=512,
        top_k=16,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        fc1_tile_n=128,
        fc1_tile_k=128,
        fc2_tile_n=128,
        fc2_tile_k=128,
        moe_block_size=8,
        max_m_blocks=1024,
        element_dtype="fp16",
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj",
        trellis_bits=2,
        full_rotation=True,
        intermediate_rotation=True,
        intermediate_hadamard=True,
        trellis_decode_table=mode,
    )
    assert kernel.trellis_direct_lut == expected
    assert kernel.shared_words * 4 <= kernel.fc1.max_shared_mem
    if expected:
        assert kernel.blocks_per_sm == 1
        assert kernel.lut_e4m3_smem
        assert kernel.fc1.trellis_direct_lut and kernel.fc2.trellis_direct_lut


class _Variant:
    def __init__(self, count):
        self.count = count

    def bind(self, **kwargs):
        return (self.count, kwargs)


@pytest.mark.parametrize(
    "tokens,topk,experts,expected_unique",
    (
        (1, 6, 384, 6),
        (4, 6, 384, 14),
        (6, 6, 384, 22),
        (8, 6, 384, 29),
        (7, 3, 128, 13),
        (8, 6, 6, 6),
    ),
)
def test_shared_40_workload_reuses_experts_across_distinct_topk_rows(
    tokens,
    topk,
    experts,
    expected_unique,
):
    """Count reuse across the batch, never duplicate experts within a token."""
    from b12x.moe.fused_moe.workloads import make_routing_ids

    for seed in (42, 43, 44, 45):
        ids = make_routing_ids(tokens, topk, experts, seed=seed)
        assert ids.shape == (tokens, topk)
        assert ids.dtype == torch.int32
        assert ids.unique().numel() == expected_unique
        assert all(row.unique().numel() == topk for row in ids)
        assert ids.min() >= 0 and ids.max() < experts
        torch.testing.assert_close(
            ids,
            make_routing_ids(tokens, topk, experts, seed=seed),
        )


@pytest.mark.parametrize(
    "tokens,topk,experts,expected_unique",
    (
        (2, 6, 256, [12, 10, 7, 6, 6]),
        (6, 6, 256, [36, 29, 22, 14, 7]),
        (8, 6, 256, [48, 38, 29, 19, 10]),
        (6, 6, 6, [6, 6, 6, 6, 6]),
        (6, 6, 12, [12, 12, 12, 12, 7]),
    ),
)
def test_verifier_tuning_corpus_spans_sharing_without_duplicate_token_routes(
    tokens,
    topk,
    experts,
    expected_unique,
):
    """A fixed mean must not collapse every trial to the same expert count."""
    from b12x.moe.fused_moe.workloads import make_tuning_routes

    ids = make_tuning_routes(tokens, topk, experts, device="cpu")
    assert ids.shape == (5, tokens, topk)
    assert ids.dtype == torch.int32
    assert [pattern.unique().numel() for pattern in ids] == expected_unique
    assert all(row.unique().numel() == topk for pattern in ids for row in pattern)
    assert ids.min() >= 0 and ids.max() < experts
    torch.testing.assert_close(
        ids,
        make_tuning_routes(tokens, topk, experts, device="cpu"),
    )


@pytest.mark.parametrize("tokens", (1, 9, 128))
def test_nonverifier_tuning_routes_retain_cyclic_coverage(tokens):
    from b12x.moe.fused_moe.workloads import make_routing_ids, make_tuning_routes

    torch.testing.assert_close(
        make_tuning_routes(tokens, 6, 256, device="cpu"),
        make_routing_ids(tokens, 6, 256, workload="disjoint").unsqueeze(0),
    )


def test_variant_for_preserves_exact_counts_and_reuses_prefill_capacity():
    variants = {count: _Variant(count) for count in (1, 2, 4, 8, 125, 128)}
    assert variant_for(variants, 4) is variants[4]
    assert variant_for(variants, 125) is variants[125]
    for count in (3, 11, 31, 126, 128):
        assert variant_for(variants, count) is variants[128]
    with pytest.raises(ValueError, match="exceeds prepared MoE capacity 128"):
        variant_for(variants, 129)


def test_capacity_state_binds_the_planned_variant_with_the_live_activations():
    variants = {count: _Variant(count) for count in (4, 128)}
    state = _FusedMoeCapacityState(variants)
    activations = torch.empty(11, 16)
    count, kwargs = state.bind(a=activations, topk_ids=None)
    assert count == 128
    assert kwargs["a"] is activations
    with pytest.raises(TypeError):
        state.bind(a=torch.empty(11))
    with pytest.raises(ValueError, match="exceeds prepared MoE capacity 128"):
        state.bind(a=torch.empty(129, 16))


def test_a16_capacity_covers_unlisted_counts_without_crossing_cutoff():
    variants = {count: _Variant(count) for count in (4, 32, 64, 128)}
    state = _FusedMoeCapacityState(variants, a16_max_tokens=32)
    for rows, expected in (
        (1, 32),
        (4, 4),
        (17, 32),
        (31, 32),
        (32, 32),
        (33, 128),
        (64, 64),
        (127, 128),
    ):
        count, _ = state.bind(a=torch.empty(rows, 16))
        assert count == expected
    for rows in (0, 129):
        with pytest.raises(ValueError):
            state.bind(a=torch.empty(rows, 16))


@pytest.mark.parametrize(
    "capacity,live_rows,expected_namespace",
    (
        (8, 1, "dynamic_w4a8_decode"),
        (8, 8, "dynamic_w4a8_decode"),
        (16, 1, "dynamic"),
        (16, 8, "dynamic"),
        (16, 16, "dynamic"),
    ),
)
def test_repacked_grid_override_namespace_uses_prepared_capacity(
    monkeypatch,
    capacity,
    live_rows,
    expected_namespace,
):
    """A prefill-capacity plan must not adopt decode overrides for short tails."""
    from b12x.moe.fused_moe import _impl

    class NamespaceResolved(Exception):
        pass

    def resolve_grid(namespace, **_kwargs):
        assert namespace == expected_namespace
        raise NamespaceResolved

    monkeypatch.setattr(_impl, "_get_impl_mac", resolve_grid)
    monkeypatch.setattr(_impl, "_dynamic_work_source", lambda: "materialized_queue")
    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    # Stop at grid resolution, before any payload storage is accessed or any
    # CUDA operation runs. Unused kernel operands intentionally remain absent.
    arguments = dict.fromkeys(signature(_impl._launch_dynamic_flat).parameters)
    arguments.update(
        quant_mode="w4a8_mx",
        activation="silu",
        E=256,
        k=4096,
        n=1024,
        m=live_rows,
        num_topk=6,
        routed_rows=live_rows * 6,
        planned_num_tokens=capacity,
        planned_tile_m=16,
        w4a8_repacked=True,
        w4a8_n64_repacked=False,
        w13_sfb_rp=torch.empty(0, dtype=torch.uint8),
        deterministic_output=False,
        planned_direct_routing=False,
    )
    with pytest.raises(NamespaceResolved):
        _impl._launch_dynamic_flat(**arguments)


@pytest.mark.parametrize("tokens, expected", [(4, 4), (11, 128)])
def test_public_route_dispatch_selects_retained_capacity(monkeypatch, tokens, expected):
    from b12x.moe.fused_moe import api

    plan = object()
    binding = SimpleNamespace(plan=plan, hidden_states=torch.empty(tokens, 16))
    variants = {
        count: SimpleNamespace(route=lambda bound, count=count: (count, bound))
        for count in (4, 128)
    }
    monkeypatch.setattr(
        api, "require_prepared", lambda *_args: SimpleNamespace(variants=variants)
    )
    assert api.route(plan, binding=binding) == (expected, binding)


def _launches(*, direct, route_pack, route_mode="auto"):
    return _W4A16PrimaryLaunches(
        tokens=8,
        route_mode=route_mode,
        packed="packed",
        packed_mapped="packed_mapped",
        direct="direct" if direct else None,
        direct_mapped="direct_mapped" if direct else None,
        topk_sum="topk_sum",
        mapped_topk_sum="mapped_topk_sum",
        route_pack=route_pack,
    )


@pytest.mark.parametrize("route_mode", ("auto", "direct"))
def test_w4a16_select_preserves_exact_direct_and_dynamic_packed_routes(route_mode):
    launches = _launches(direct=True, route_pack="route_pack", route_mode=route_mode)
    assert launches.select(
        tokens=8,
        route_ids_dtype=torch.int32,
        has_route_map=False,
        activation_amax=None,
    ) == ("direct", "topk_sum", None)
    assert launches.select(
        tokens=8,
        route_ids_dtype=torch.int64,
        has_route_map=False,
        activation_amax=None,
    ) == ("packed", "topk_sum", "route_pack")
    assert launches.select(
        tokens=3,
        route_ids_dtype=torch.int32,
        has_route_map=False,
        activation_amax=None,
    ) == ("packed", "topk_sum", "route_pack")
    assert launches.select(
        tokens=3,
        route_ids_dtype=torch.int64,
        has_route_map=True,
        activation_amax=None,
    ) == ("packed_mapped", "topk_sum", "route_pack")
    with pytest.raises(RuntimeError, match="requested=9, prepared=8"):
        launches.select(
            tokens=9,
            route_ids_dtype=torch.int32,
            has_route_map=False,
            activation_amax=None,
        )


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_native_w4a16_selects_retained_micro_and_keeps_packed_boundary(dtype):
    packed = SimpleNamespace(
        small_m_direct_launches=tuple(
            SimpleNamespace(topk_ids_dtype=value)
            for value in (torch.int32, torch.int64)
        )
    )
    launches = replace(
        _launches(direct=False, route_pack="route_pack", route_mode="direct"),
        packed=packed,
    )
    exact = dict(
        tokens=8, route_ids_dtype=dtype, has_route_map=False, activation_amax=None
    )
    assert launches.select(**exact) == (packed, "topk_sum", None)
    assert launches.select(**{**exact, "tokens": 3}) == (
        packed,
        "topk_sum",
        "route_pack",
    )
    assert launches.select(**{**exact, "has_route_map": True}) == (
        "packed_mapped",
        "topk_sum",
        "route_pack",
    )
    assert launches.select(**{**exact, "activation_amax": object()}) == (
        packed,
        "topk_sum",
        "route_pack",
    )
    assert replace(launches, route_mode="packed").select(**exact) == (
        packed,
        "topk_sum",
        "route_pack",
    )


def test_native_launch_metadata_tracks_capacity_when_packed_program_is_cached(
    monkeypatch,
):
    from b12x._lib.compile_plan import ProgramKey, program_keys
    from b12x.moe._shared.kernels.w4a16 import kernel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(kernel, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(kernel, "_FUSED_CACHE", {})
    compiled = []

    def compile_fused(*args, **kwargs):
        program = SimpleNamespace(
            __b12x_programs__=(ProgramKey("cute", f"fused-{len(compiled)}", "fused"),)
        )
        compiled.append(program)
        return program

    def compile_direct(*, m, topk_ids_dtype, **kwargs):
        return SimpleNamespace(
            m=m,
            topk_ids_dtype=topk_ids_dtype,
            compiled=SimpleNamespace(
                __b12x_programs__=(
                    ProgramKey("cute", f"direct-{m}-{topk_ids_dtype}", "direct"),
                )
            ),
        )

    monkeypatch.setattr(kernel, "b12x_compile", compile_fused)
    monkeypatch.setattr(kernel, "_compile_w4a16_small_m_direct", compile_direct)
    packed_specializations = {}
    for m in (1, 2, 4, 8, 9, 8, 1):
        launch = kernel.compile_w4a16_fused_moe(
            size_m=m,
            hidden_size=256,
            intermediate_size=256,
            num_experts=8,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=32,
            sms=188,
            max_shared_mem=kernel._DEFAULT_MAX_SHARED_MEM,
            weight_layout="modelopt",
            force_tile_config=(128, 128, 128, 128),
        )
        assert launch.size_m == m
        assert launch.compiled is packed_specializations.setdefault(
            m == 1, launch.compiled
        )
        assert {
            (direct.m, direct.topk_ids_dtype)
            for direct in launch.small_m_direct_launches
        } == ({(m, torch.int32), (m, torch.int64)} if m <= 8 else set())
        expected_programs = program_keys(launch.compiled) + tuple(
            key
            for direct in launch.small_m_direct_launches
            for key in program_keys(direct.compiled)
        )
        assert program_keys(launch) == expected_programs
    assert len(compiled) == 2
