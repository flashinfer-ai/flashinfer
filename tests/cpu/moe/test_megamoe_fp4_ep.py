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
@pytest.mark.parametrize("tile_bytes", [8448, 16384])
@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 64])
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
@pytest.mark.parametrize("tile_bytes", [8704, 8448, 16384])
@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 64])
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
