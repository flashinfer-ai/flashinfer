"""CPU checks for EP-specific configuration and JIT cache isolation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import flashinfer.megamoe as megamoe
from flashinfer.jit.megamoe import (
    get_megamoe_sm90_mxfp4_uri,
    get_megamoe_sm90_nvfp4_uri,
)
from flashinfer.megamoe import select_fused_plan
from flashinfer.megamoe.heuristics import (
    SMEM_CAPACITY,
    _fused_smem_bytes,
    select_mxfp4_fused_plan,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_sms", [78, 132])
@pytest.mark.parametrize(
    "experts,hidden,intermediate,topk",
    [(256, 4096, 2048, 6), (384, 6144, 2048, 8), (384, 7168, 3072, 6)],
)
@pytest.mark.parametrize("tokens", [1, 24, 64, 256])
def test_ep4_fused_plan_uses_legal_generic_pipeline(
    num_sms, experts, hidden, intermediate, topk, tokens
):
    plan = select_fused_plan(
        num_sms, 4, experts, topk, hidden, intermediate, 384, tokens
    )
    assert not plan.rs_swap_ab
    assert not plan.push_dispatch
    assert not plan.no_clean_barrier
    assert (experts // 4) % plan.num_experts_per_wave == 0
    if plan.block_m == 8:
        assert plan.num_stages in (3, 4)
    else:
        assert plan.num_stages == 3


@pytest.mark.cpu_only
def test_nvfp4_jit_cache_distinguishes_ep_width():
    plan = select_fused_plan(132, 8, 256, 6, 4096, 2048, 384, 1)
    args = (plan, 132, 384, 256, 6, 4096, 2048, 10.0, True)
    ep4 = get_megamoe_sm90_nvfp4_uri(*args, num_ranks=4)
    ep8 = get_megamoe_sm90_nvfp4_uri(*args, num_ranks=8)
    assert ep4 != ep8
    assert "_r4_" in ep4 and "_r8_" in ep8


@pytest.mark.cpu_only
@pytest.mark.parametrize("ep", [1, 2, 3, 16])
def test_nvfp4_planner_rejects_unvalidated_ep_width(ep):
    with pytest.raises(ValueError, match="4 or 8 ranks"):
        select_fused_plan(132, ep, 256, 6, 4096, 2048, 384, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "experts,hidden,intermediate,topk,push_max,rotated_from",
    [
        (256, 4096, 2048, 6, 32, None),
        (384, 6144, 2048, 8, 32, None),
        (384, 7168, 3072, 6, 64, 33),
    ],
)
def test_h200_ep8_counter_policy(
    experts, hidden, intermediate, topk, push_max, rotated_from
):
    for tokens in (1, 8, 24, 32, 33, 48, 64, 65, 128, 256):
        plan = select_fused_plan(
            132, 8, experts, topk, hidden, intermediate, 4096, tokens
        )
        assert plan.push_dispatch == (tokens <= push_max), tokens
        assert plan.no_clean_barrier == (
            rotated_from is not None and rotated_from <= tokens <= push_max
        ), tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_sms", [78, 132])
def test_counter_policy_limits(num_sms):
    # Unmeasured geometry (H7168 / E256 / K8, DeepSeek-V3): push up to M=32, no rotated pool.
    for tokens in (1, 16, 32, 33, 64):
        plan = select_fused_plan(num_sms, 8, 256, 8, 7168, 2048, 4096, tokens)
        assert plan.push_dispatch == (tokens <= 32), tokens
        assert not plan.no_clean_barrier
    # Forced counter dispatch is ignored outside the validated small-M plans.
    for tokens in (65, 128, 256):
        plan = select_fused_plan(
            num_sms,
            8,
            256,
            6,
            4096,
            2048,
            4096,
            tokens,
            force_push_dispatch=True,
            force_no_clean_barrier=True,
        )
        assert not plan.push_dispatch and not plan.no_clean_barrier, tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_sms", [78, 132])
@pytest.mark.parametrize(
    "hidden,intermediate,experts,topk",
    [
        (7168, 2048, 256, 8),
        (7168, 2048, 384, 8),
        (8192, 2048, 256, 8),
        (4096, 2048, 384, 8),
        (512, 256, 256, 6),
        (5120, 1536, 160, 8),  # GLM-4.6 / 4.7
        (4096, 1408, 128, 8),  # GLM-4.5-Air
        (2048, 768, 128, 8),  # Qwen3-30B-A3B
        (4096, 1024, 512, 10),  # Qwen3.5-397B-A17B
        (4096, 1024, 8192, 8),  # expert table alone overflows the BM64 pipeline
        (7168, 2048, 2048, 8),
    ],
)
def test_untuned_geometry_plans_are_legal(num_sms, hidden, intermediate, experts, topk):
    # Shapes outside the tuned set get the generic schedule (no H20 RS-only depth), a pipeline
    # depth that fits shared memory, and push only where the push row copy supports the hidden size.
    for tokens in range(1, 257):
        plan = select_fused_plan(
            num_sms, 8, experts, topk, hidden, intermediate, 4096, tokens
        )
        assert not plan.rs_swap_ab
        legal = {8: (3, 4), 16: (3,), 24: (3,), 64: (3,)}[plan.block_m]
        assert plan.num_stages in legal, (tokens, plan)
        assert _fused_smem_bytes(plan, num_sms, experts, hidden) <= SMEM_CAPACITY
        assert not plan.push_dispatch or (hidden % 1024 == 0 and tokens <= 32), tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "hidden,intermediate", [(4096, 2560), (4352, 2048), (4096, 1000)]
)
def test_unsupported_fused_shapes_raise(hidden, intermediate):
    with pytest.raises(ValueError):
        select_fused_plan(132, 8, 256, 6, hidden, intermediate, 4096, 1)


@pytest.mark.cpu_only
def test_push_needs_one_publishing_sm_per_local_expert():
    # Local expert e is published by SM e, so push needs experts per rank <= SMs.
    plan = select_fused_plan(78, 8, 1024, 8, 4096, 1024, 4096, 1)
    assert not plan.push_dispatch
    plan = select_fused_plan(132, 8, 1024, 8, 4096, 1024, 4096, 1)
    assert plan.push_dispatch


@pytest.mark.cpu_only
@pytest.mark.parametrize("ep", [8, 16])
@pytest.mark.parametrize(
    "tile_bytes,tokens",
    [(8448, m) for m in (1, 4, 8)] + [(16384, m) for m in (1, 4, 8, 16, 32, 64)],
)
def test_mxfp4_explicit_storage_plans(ep, tile_bytes, tokens):
    plan = select_mxfp4_fused_plan(
        78, ep, 256, 6, 4096, 2048, 64, tokens, weight_tile_bytes=tile_bytes
    )
    expected = (
        (8, 4 if tokens == 1 else 6)
        if tile_bytes == 8448 or tokens <= 16
        else {32: (16, 6), 64: (24, 5)}[tokens]
    )
    assert (plan.block_m, plan.num_stages) == expected
    assert plan.rs_swap_ab == (tile_bytes == 8448)
    assert plan.block_n == 256 and plan.num_experts_per_wave == 16
    assert plan.swap_ab and plan.use_interleaved_scheduler
    assert not plan.push_dispatch and not plan.no_clean_barrier


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tokens,tile_bytes,block_m,stages",
    [(2, 8448, 8, 4), (2, 16384, 8, 4), (128, 16384, 32, 4)],
)
def test_ep8_storage_plans_cover_equal_global_token_comparisons(
    tokens, tile_bytes, block_m, stages
):
    plan = select_mxfp4_fused_plan(
        78, 8, 256, 6, 4096, 2048, 128, tokens, weight_tile_bytes=tile_bytes
    )
    assert (plan.block_m, plan.num_stages) == (block_m, stages)
    assert plan.rs_swap_ab == (tile_bytes == 8448)


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_sms", [78, 132])
@pytest.mark.parametrize("ep", [8, 16])
@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 64, 128])
def test_mxfp4_canonical_keeps_generic_plans(num_sms, ep, tokens):
    args = (num_sms, ep, 256, 6, 4096, 2048, 128, tokens)
    assert select_mxfp4_fused_plan(*args) == select_mxfp4_fused_plan(
        *args, weight_tile_bytes=8704
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("tile_bytes", [8448, 16384])
@pytest.mark.parametrize(
    "overrides",
    [
        {"num_sms": 132},
        {"num_ranks": 4},
        {"num_experts": 384},
        {"hidden": 6144},
        {"intermediate_hidden": 3072},
        {"num_topk": 8},
        {"num_tokens": 2},
        {"num_logical_experts": 128},
    ],
)
def test_noncanonical_mxfp4_storage_never_uses_generic_fallback(tile_bytes, overrides):
    args = dict(
        num_sms=78,
        num_ranks=16,
        num_experts=256,
        num_topk=6,
        hidden=4096,
        intermediate_hidden=2048,
        num_max_tokens_per_rank=64,
        num_tokens=8,
        weight_tile_bytes=tile_bytes,
    )
    args.update(overrides)
    with pytest.raises(ValueError, match="noncanonical MXFP4 storage"):
        select_mxfp4_fused_plan(**args)


@pytest.mark.cpu_only
@pytest.mark.parametrize("ep,nodes", [(8, 1), (16, 2)])
def test_mxfp4_jit_cache_distinguishes_storage(ep, nodes):
    uris = []
    for tile_bytes in (8704, 8448, 16384):
        plan = select_mxfp4_fused_plan(
            78, ep, 256, 6, 4096, 2048, 64, 8, weight_tile_bytes=tile_bytes
        )
        uri = get_megamoe_sm90_mxfp4_uri(
            plan,
            78,
            ep,
            64,
            256,
            6,
            4096,
            2048,
            10.0,
            True,
            num_nodes=nodes,
            weight_tile_bytes=tile_bytes,
        )
        assert f"_wt{tile_bytes}" in uri and f"_r{ep}_" in uri
        uris.append(uri)
    assert len(set(uris)) == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize("tile_bytes,rs", [(8448, False), (16384, True)])
def test_mxfp4_jit_rejects_incompatible_math_storage(tile_bytes, rs):
    plan = select_mxfp4_fused_plan(78, 16, 256, 6, 4096, 2048, 64, 8)
    with pytest.raises(ValueError):
        get_megamoe_sm90_mxfp4_uri(
            replace(plan, rs_swap_ab=rs),
            78,
            16,
            64,
            256,
            6,
            4096,
            2048,
            10.0,
            True,
            weight_tile_bytes=tile_bytes,
        )


def _mxfp4_call_inputs(tokens, tile_bytes, ep=16, nodes=2):
    """Describe full-size operands without allocating CPU or GPU weight storage."""
    buffer = SimpleNamespace(
        buffer=torch.empty(0, device="meta"),
        num_sms=78,
        num_nodes=nodes,
        num_experts=256,
        num_topk=6,
        hidden=4096,
        intermediate_hidden=2048,
        num_max_tokens_per_rank=64,
        num_staged_tokens=64,
        group=SimpleNamespace(size=lambda: ep),
        _remote_args=("remote-arguments",),
    )
    y = torch.empty(tokens, 4096, dtype=torch.bfloat16, device="meta")
    w1 = torch.empty(256 // ep, 1024, tile_bytes, dtype=torch.uint8, device="meta")
    w2 = torch.empty(256 // ep, 512, tile_bytes, dtype=torch.uint8, device="meta")
    return y, w1, w2, buffer


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tile_bytes,tokens",
    [(8448, m) for m in (1, 4, 8)]
    + [(tile_bytes, m) for tile_bytes in (8704, 16384) for m in (1, 4, 8, 16, 32, 64)],
)
def test_mxfp4_public_api_routes_actual_weight_records(monkeypatch, tile_bytes, tokens):
    y, w1, w2, buffer = _mxfp4_call_inputs(tokens, tile_bytes)
    calls = []

    def record_call(factory, entry, plan, *args, **kwargs):
        calls.append((entry, plan, kwargs))
        return args[0]

    monkeypatch.setattr(megamoe, "_run_kernel", record_call)
    assert megamoe.mxfp4_mega_moe(y, w1, w2, buffer) is y
    entry, plan, kwargs = calls.pop()
    assert entry == "megamoe_sm90_mxfp4_fused"
    assert kwargs["extra_jit_args"] == (False, 2, tile_bytes)
    assert kwargs["extra_kernel_args"] == buffer._remote_args
    if tile_bytes != 8704:
        assert plan.rs_swap_ab == (tile_bytes == 8448)


@pytest.mark.cpu_only
def test_mxfp4_public_api_rejects_mixed_or_malformed_weight_records():
    y, w1, w2, buffer = _mxfp4_call_inputs(8, 8448)
    with pytest.raises(ValueError, match="matching 3-D records"):
        megamoe.mxfp4_mega_moe(y, w1, w2[..., :8192], buffer)
    with pytest.raises(ValueError, match="weights must have shape"):
        megamoe.mxfp4_mega_moe(y, w1[:-1], w2, buffer)


@pytest.mark.cpu_only
@pytest.mark.parametrize("nodes,tile_bytes", [(2, 8704), (1, 8448), (1, 16384)])
def test_mxfp4_slots_fail_before_accessing_unsupported_slot_state(nodes, tile_bytes):
    y, w1, w2, buffer = _mxfp4_call_inputs(8, tile_bytes, nodes=nodes)
    with pytest.raises(ValueError, match="single-node canonical"):
        megamoe.mxfp4_mega_moe(y, w1, w2, buffer, expert_slots=object())


@pytest.mark.cpu_only
def test_mxfp4_single_node_canonical_slots_keep_existing_entry(monkeypatch):
    y, _, _, buffer = _mxfp4_call_inputs(8, 8704, ep=8, nodes=1)
    buffer.num_experts = 320
    w1 = torch.empty(40, 1024, 8704, dtype=torch.uint8, device="meta")
    w2 = torch.empty(40, 512, 8704, dtype=torch.uint8, device="meta")
    scales = torch.empty(40, dtype=torch.float32, device="meta")
    checks, calls = [], []
    slots = SimpleNamespace(
        num_experts=256,
        l1_global_scales=scales,
        l2_global_scales=scales,
        _check_kernel_operands=lambda *args: checks.append(args),
        _in_kernel_copy_args=lambda: ("copy-arguments",),
    )

    def record_call(factory, entry, plan, *args, **kwargs):
        calls.append((entry, kwargs))
        return args[0]

    monkeypatch.setattr(megamoe, "_run_kernel", record_call)
    assert megamoe.mxfp4_mega_moe(y, w1, w2, buffer, expert_slots=slots) is y
    assert len(checks) == 1
    entry, kwargs = calls.pop()
    assert entry == "megamoe_sm90_mxfp4_fused_in_kernel_copy"
    assert kwargs["extra_jit_args"] == (True, 1, 8704)
    assert kwargs["extra_kernel_args"] == buffer._remote_args + ("copy-arguments",)


@pytest.mark.cpu_only
def test_nvfp4_rejects_cross_node_buffer_before_kernel_selection():
    y, w1, w2, buffer = _mxfp4_call_inputs(8, 8704, ep=8)
    with pytest.raises(ValueError, match="single node"):
        megamoe.nvfp4_mega_moe(y, w1, w2, buffer)


def _mxfp4_source_weights():
    """Construct complete small tiles in the documented source layout."""
    return tuple(
        (
            torch.zeros(1, 256, k // 2, dtype=torch.uint8),
            torch.full((1, 256, k // 32), 127, dtype=torch.uint8),
        )
        for k in (256, 128)
    )


@pytest.mark.cpu_only
def test_mxfp4_preparation_rejects_unknown_storage():
    with pytest.raises(ValueError, match="unsupported MXFP4 storage"):
        megamoe.transform_mxfp4_weights_for_mega_moe(
            *_mxfp4_source_weights(), storage="unknown"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("storage", ["mxfp4", "mxfp4_compact", "fp8"])
@pytest.mark.parametrize("scale", [0, 121, 134, 255])
@pytest.mark.parametrize("code_byte", [0x10, 0x01])
def test_mxfp4_nonzero_groups_reject_out_of_range_scales(storage, scale, code_byte):
    weights = _mxfp4_source_weights()
    weights[0][0][0, 0, 16] = code_byte
    weights[0][1][0, 0, 1] = scale
    with pytest.raises(ValueError, match=r"nonzero MXFP4 groups.*\[122, 133\]"):
        megamoe.transform_mxfp4_weights_for_mega_moe(*weights, storage=storage)


@pytest.mark.cpu_only
@pytest.mark.parametrize("storage", ["mxfp4", "mxfp4_compact", "fp8"])
def test_mxfp4_zero_group_scales_are_unused(storage):
    weights = _mxfp4_source_weights()
    for packed, scales in weights:
        packed.fill_(0x88)
        scales.copy_(torch.arange(scales.numel()).reshape_as(scales).to(torch.uint8))
    # A nonzero code in group zero does not make group one's scale live.
    weights[0][0][0, 0, 0] = 0x10
    weights[0][1][0, 0, 0] = 122
    weights[0][1][0, 0, 1] = 255
    if storage == "mxfp4":
        outputs = megamoe.transform_mxfp4_weights_for_mega_moe(*weights)
        assert [tuple(t.shape) for t in outputs] == [(1, 4, 8704), (1, 2, 8704)]
    else:
        with pytest.raises(ValueError, match="requires CUDA inputs"):
            megamoe.transform_mxfp4_weights_for_mega_moe(*weights, storage=storage)


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [122, 133])
def test_mxfp4_scale_domain_includes_both_endpoints(scale):
    weights = _mxfp4_source_weights()
    for packed, scales in weights:
        packed.fill_(0xFF)
        scales.fill_(scale)
    outputs = megamoe.transform_mxfp4_weights_for_mega_moe(*weights)
    assert all(t.is_contiguous() for t in outputs)


@pytest.mark.cpu_only
def test_expert_slots_reject_cross_node_buffer_before_workspace_creation(monkeypatch):
    monkeypatch.setattr(
        megamoe.eplb,
        "_get_jit_module",
        lambda *args: pytest.fail("unsupported slots must not build a workspace"),
    )
    with pytest.raises(ValueError, match="requires a single node"):
        megamoe.MegaMoEExpertSlots(SimpleNamespace(num_nodes=2), 256, 4)


def _cpu_replica_plan():
    """Create CPU-backed tables for the actual plan-installation method."""
    slots = megamoe.MegaMoEExpertSlots.__new__(megamoe.MegaMoEExpertSlots)
    slots.num_ranks, slots.num_experts = 4, 8
    slots.num_resident_experts, slots.num_redundant_slots, slots.rank = 2, 2, 1
    slots.slot_expert = torch.full((4, 2), -1, dtype=torch.int32)
    slots.copies = torch.empty(8, 4, 2, dtype=torch.int32)
    slots._fill_resident_copies()
    slots.expert_base = torch.full((8,), 11, dtype=torch.int32)
    slots.local_counter = torch.full((8,), 7, dtype=torch.int32)
    slots.copy_list = torch.full((2, 2), -1, dtype=torch.int32)
    slots.copy_count = torch.zeros(1, dtype=torch.int32)
    slots.copy_done = torch.full((2, 2), 9, dtype=torch.int32)
    table = torch.tensor([[2, 4], [0, 4], [0, 2], [0, 2]], dtype=torch.int64)
    copies = slots.copies.to(torch.int64)
    copies[0] = torch.tensor([[0, 0], [1, 0], [2, 2], [3, 2]])
    # The device planner writes only the first sentinel, leaving later data stale.
    copies[2] = torch.tensor([[1, 2], [0, 3], [-1, -1], [77, -42]])
    copies[4] = torch.tensor([[2, 0], [1, 0], [-1, 0], [99, -99]])
    base = torch.arange(8, dtype=torch.int64)
    return slots, table, copies, base


@pytest.mark.cpu_only
def test_replica_plan_accepts_zero_shares_full_rows_and_stale_sentinel_tail():
    slots, table, copies, base = _cpu_replica_plan()
    assert slots.set_replica_plan(table, copies, base) == 2
    assert torch.equal(slots.slot_expert, table.to(torch.int32))
    assert torch.equal(slots.copies, copies.to(torch.int32))
    assert torch.equal(slots.expert_base, base.to(torch.int32))
    assert not slots.local_counter.any()
    assert slots.copy_list.tolist() == [[0, 0], [1, 4]]
    assert slots.copy_count.item() == 2
    assert not slots.copy_done.any()
    assert slots.set_replica_plan(table, copies, base) == 2
    slots.copy_count.zero_()
    assert slots.set_replica_plan(table, copies, base) == 0


@pytest.mark.cpu_only
def test_replica_plan_default_tables_restore_resident_only_routing():
    slots, table, copies, base = _cpu_replica_plan()
    slots.set_replica_plan(table, copies, base)
    assert slots.set_replica_plan(torch.full((4, 2), -1)) == 0
    assert slots.slot_expert.eq(-1).all()
    assert slots.copies[:, 0, 0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert slots.copies[:, 0, 1].eq(torch.iinfo(torch.int32).max).all()
    assert slots.copies[:, 1:].eq(-1).all()
    assert not slots.expert_base.any() and not slots.copy_count.any()


@pytest.mark.cpu_only
def test_replica_plan_preserves_int32_boundary_values_and_flat_inputs():
    slots, table, copies, base = _cpu_replica_plan()
    base[0], base[1] = torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
    copies[2, 1, 1] = torch.iinfo(torch.int32).max
    slots.set_replica_plan(table.flatten().tolist(), copies.flatten(), base)
    assert torch.equal(slots.expert_base, base.to(torch.int32))
    assert slots.copies[2, 1, 1].item() == torch.iinfo(torch.int32).max


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "invalid",
    [
        "slot_shape",
        "slot_range",
        "slot_resident",
        "slot_duplicate",
        "slot_float",
        "slot_overflow",
        "copies_shape",
        "copies_float",
        "copies_rank_range",
        "copies_negative_rank",
        "copies_wrong_resident",
        "copies_missing_slot",
        "copies_negative_bound",
        "copies_decreasing_bounds",
        "copies_rank_overflow",
        "copies_bound_overflow",
        "base_shape",
        "base_float",
        "base_overflow",
    ],
)
def test_replica_plan_rejects_all_bad_inputs_before_mutating_tables(invalid):
    slots, table, copies, base = _cpu_replica_plan()
    if invalid == "slot_shape":
        table = table[:-1]
    elif invalid == "slot_range":
        table[1, 0] = 8
    elif invalid == "slot_resident":
        table[1, 0] = 2
    elif invalid == "slot_duplicate":
        table[1, 1] = table[1, 0]
    elif invalid == "slot_float":
        table = table.to(torch.float32)
    elif invalid == "slot_overflow":
        table[1, 0] = 2**32
    elif invalid == "copies_shape":
        copies = copies[:-1]
    elif invalid == "copies_float":
        copies = copies.to(torch.float32)
    elif invalid == "copies_rank_range":
        copies[0, 1, 0] = 4
    elif invalid == "copies_negative_rank":
        copies[0, 1, 0] = -2
    elif invalid == "copies_wrong_resident":
        copies[0, 0, 0] = 1
    elif invalid == "copies_missing_slot":
        table[2, 0] = 1
    elif invalid == "copies_negative_bound":
        copies[0, 0, 1] = -1
    elif invalid == "copies_decreasing_bounds":
        copies[0, 0, 1], copies[0, 1, 1] = 3, 2
    elif invalid == "copies_rank_overflow":
        copies[0, 1, 0] = 2**32 + 1
    elif invalid == "copies_bound_overflow":
        copies[0, 0, 1] = 2**31
    elif invalid == "base_shape":
        base = base[:-1]
    elif invalid == "base_float":
        base = base.to(torch.float32)
    elif invalid == "base_overflow":
        base[0] = 2**31
    names = (
        "slot_expert",
        "copies",
        "expert_base",
        "local_counter",
        "copy_list",
        "copy_count",
        "copy_done",
    )
    before = {name: getattr(slots, name).clone() for name in names}
    with pytest.raises(ValueError):
        slots.set_replica_plan(table, copies, base)
    for name in names:
        assert torch.equal(getattr(slots, name), before[name]), name


@pytest.mark.cpu_only
@pytest.mark.parametrize("ep", [8, 16])
@pytest.mark.parametrize("tokens", [16, 32, 64, 128])
def test_compact_prefill_rejected_before_kernel_dispatch(monkeypatch, ep, tokens):
    y, w1, w2, buffer = _mxfp4_call_inputs(tokens, 8448, ep=ep, nodes=ep // 8)
    buffer.num_max_tokens_per_rank = buffer.num_staged_tokens = 128
    monkeypatch.setattr(
        megamoe,
        "_run_kernel",
        lambda *a, **kw: pytest.fail("unsupported compact plan launched"),
    )
    with pytest.raises(ValueError, match="8448-byte records"):
        megamoe.mxfp4_mega_moe(y, w1, w2, buffer)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "hidden,intermediate,message",
    [
        (4096, 2176, r"hidden >= 2 \* intermediate_hidden"),
        (16640, 8320, "intermediate_hidden <= 8192"),
    ],
)
def test_mxfp4_canonical_rejects_scheduler_shape_limits(hidden, intermediate, message):
    with pytest.raises(ValueError, match=message):
        select_mxfp4_fused_plan(78, 16, 256, 6, hidden, intermediate, 2, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("hidden,intermediate,topk", [(512, 256, 32), (16384, 8192, 6)])
def test_mxfp4_canonical_scheduler_boundaries_reach_jit(hidden, intermediate, topk):
    plan = select_mxfp4_fused_plan(78, 16, 256, topk, hidden, intermediate, 2, 1)
    assert plan.use_interleaved_scheduler and plan.block_n == 256
    uri = get_megamoe_sm90_mxfp4_uri(
        plan, 78, 16, 2, 256, topk, hidden, intermediate, 10.0, True, num_nodes=2
    )
    assert f"_k{topk}_h{hidden}_i{intermediate}_cap2_" in uri
    assert uri.endswith("_wt8704_nodes2")
