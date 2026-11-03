"""Complete routes 18 (m512, c340sr) and 19 (m128, c336sr): the self-routing up kernels (no alignment launch) equal the prepared
route-12 / route-10 paths bit for bit, on the fixture routing and on a skewed routing with > 32 pairs per expert. With the K-256 W2
sidecars, both routes are also FP4-close to the torch oracle on the contract fixture, with and without a BF16 seed."""

import importlib.util
import pathlib

import pytest
import torch

# pytest collects this directory without putting it on sys.path (importlib mode): load the sibling test module by file path.
_SIBLING = pathlib.Path(__file__).with_name("test_alphamoe_nvfp4_sm100.py")
_SPEC = importlib.util.spec_from_file_location(
    "alphamoe_nvfp4_sm100_test_helpers", _SIBLING
)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)
_complete_route_case = _HELPERS._complete_route_case
_complete_route_workspaces = _HELPERS._complete_route_workspaces
_skip_if_not_supported = _HELPERS._skip_if_not_supported
_make_case = _HELPERS._make_case
_pair_experts = _HELPERS._pair_experts
_reference = _HELPERS._reference
_assert_nvfp4_close = _HELPERS._assert_nvfp4_close

K256_KEYS = ("w2_data_prepared_k256", "w2_scale_prepared_k256")
# m -> (self-routing route, prepared reference route selected when the two K-256 sidecars are dropped)
ROUTES = {512: (18, 12), 128: (19, 10)}
# m -> number of leading tokens routed to a permutation of experts 0..top_k-1 (each of those experts then holds >= SKEW_TOKENS pairs)
SKEW_TOKENS = {512: 150, 128: 70}


def _route_of(api, case, seed_out):
    return api._alphamoe_complete_route_id(
        case["hidden_states"],
        case["hidden_states_scale"],
        case["gemm1_weights"],
        case["gemm1_weights_scale"],
        case["gemm2_weights_scale"],
        case["topk_ids"],
        seed_out,
        case["top_k"],
        case["block_m"],
        case["w1_scale_prepared"],
        case["w1_data_prepared"],
        w1_scale_prepared_interleaved=case.get("w1_scale_prepared_interleaved"),
        w2_scale_prepared=case.get("w2_scale_prepared"),
        w2_data_prepared=case.get("w2_data_prepared"),
        w2_data_prepared_k256=case.get("w2_data_prepared_k256"),
        w2_scale_prepared_k256=case.get("w2_scale_prepared_k256"),
    )


def _skewed_topk_ids(topk_ids, skew_tokens):
    """First `skew_tokens` tokens: a fresh permutation of experts 0..top_k-1 each (distinct ids per token, deterministic)."""
    m, top_k = topk_ids.shape
    assert 0 < skew_tokens <= m
    g = torch.Generator(device="cpu").manual_seed(7000 + m)
    perms = torch.stack(
        [torch.randperm(top_k, generator=g) for _ in range(skew_tokens)]
    ).to(torch.int32)
    ids = topk_ids.clone()
    ids[:skew_tokens] = perms.to(ids.device)
    return ids.contiguous()


def _assert_selfroute_matches_prepared(
    api, case, seed_out, workspaces, route, reference_route
):
    for key in K256_KEYS:
        assert case.get(key) is not None, f"fixture lacks {key}"
    reference_case = {k: v for k, v in case.items() if k not in K256_KEYS}
    assert _route_of(api, case, seed_out) == route
    assert _route_of(api, reference_case, seed_out) == reference_route

    direct = torch.full_like(seed_out, float("nan"))
    api.alphamoe_nvfp4_routed_moe(out=direct, accumulate=False, **workspaces(), **case)
    reference = torch.full_like(seed_out, float("nan"))
    api.alphamoe_nvfp4_routed_moe(
        out=reference, accumulate=False, **workspaces(), **reference_case
    )
    torch.cuda.synchronize()
    assert torch.isfinite(direct.float()).all()
    assert torch.equal(direct, reference), (
        f"route {route} differs from the prepared route-{reference_route} path (accumulate=False)"
    )

    seeded = seed_out.clone()
    api.alphamoe_nvfp4_routed_moe(out=seeded, **workspaces(), **case)
    seeded_reference = seed_out.clone()
    api.alphamoe_nvfp4_routed_moe(
        out=seeded_reference, **workspaces(), **reference_case
    )
    torch.cuda.synchronize()
    assert torch.equal(seeded, seeded_reference), (
        f"route {route} differs from the prepared route-{reference_route} path (accumulate=True)"
    )


@pytest.mark.parametrize("m", [512, 128], ids=["m512", "m128"])
def test_alphamoe_nvfp4_selfroute_route_matches_prepared(m):
    """Route 18 (c340sr) / 19 (c336sr) equal the prepared route-12 / route-10 path bit for bit, in both accumulate forms."""
    _skip_if_not_supported()
    from flashinfer.fused_moe import alphamoe_nvfp4_sm100 as api

    route, reference_route = ROUTES[m]
    case, seed_out, workspaces = _complete_route_case(m, 4020 + m, prepared_data=True)
    _assert_selfroute_matches_prepared(
        api, case, seed_out, workspaces, route, reference_route
    )


@pytest.mark.parametrize("m", [512, 128], ids=["m512", "m128"])
def test_alphamoe_nvfp4_selfroute_skewed_routing_matches_prepared(m):
    """Same, with a skewed routing: experts 0..top_k-1 hold > 32 pairs each (multi-owner bins -> the sibling-window path of the
    self-routing prologue), the remaining experts keep the fixture's routing (empty experts included on m128)."""
    _skip_if_not_supported()
    from flashinfer.fused_moe import alphamoe_nvfp4_sm100 as api

    route, reference_route = ROUTES[m]
    case, seed_out, workspaces = _complete_route_case(m, 4020 + m, prepared_data=True)
    case = dict(case)
    case["topk_ids"] = _skewed_topk_ids(case["topk_ids"], SKEW_TOKENS[m])
    counts = torch.bincount(
        case["topk_ids"].reshape(-1).long(), minlength=case["gemm1_weights"].shape[0]
    )
    assert int(counts.max()) > 32, int(counts.max())
    assert int((counts[: case["top_k"]] >= SKEW_TOKENS[m]).sum()) == case["top_k"], (
        counts[: case["top_k"]].tolist()
    )
    _assert_selfroute_matches_prepared(
        api, case, seed_out, workspaces, route, reference_route
    )


def _oracle_case(api, m, seed):
    """Contract fixture (controlled FP4 scale exponents) at the complete-route geometry, with every prepared operand.

    Routed scaling 0.25 keeps the error of the oracle's 32 BF16 additions per output element (8 routes x 4 intermediate
    blocks) well inside the contract tolerance.
    """
    case = _make_case(m, 1024, 6144, 256, 8, 8, 0.8, 0.25, seed)
    topk_ids = (
        _pair_experts(case).reshape(m, case["top_k"]).to(torch.int32).contiguous()
    )
    kwargs = dict(
        hidden_states=case["x"],
        hidden_states_scale=case["x_scale"],
        gemm1_weights=case["w1"],
        gemm1_weights_scale=case["w1_scale"],
        gemm2_weights=case["w2"],
        gemm2_weights_scale=case["w2_scale"],
        output1_scale_gate_scalar=case["output1_scale_gate_scalar"],
        output1_scale_scalar=case["output1_scale_scalar"],
        output2_scale_scalar=case["output2_scale_scalar"],
        topk_weights=case["topk_weights"],
        topk_ids=topk_ids,
        top_k=case["top_k"],
        block_m=case["block_m"],
        routed_scaling_factor=case["scaling_factor"],
        w1_scale_prepared=api.prepare_nvfp4_w1_scales(case["w1_scale"]),
        w2_scale_prepared=api.prepare_nvfp4_w2_scales(case["w2_scale"]),
        w1_data_prepared=api.prepare_nvfp4_w1_data(case["w1"]),
        w2_data_prepared=api.prepare_nvfp4_w2_data(case["w2"]),
        w2_data_prepared_k256=api.prepare_nvfp4_w2_data_k256(case["w2"]),
        w2_scale_prepared_k256=api.prepare_nvfp4_w2_scales_k256(case["w2_scale"]),
        w1_scale_prepared_interleaved=api.prepare_nvfp4_w1_scales_interleaved(
            case["w1_scale"]
        ),
    )
    return case, kwargs


@pytest.mark.parametrize("m", [512, 128], ids=["m512", "m128"])
def test_alphamoe_nvfp4_selfroute_route_matches_oracle(m):
    """Route 18 / 19 with the K-256 W2 sidecars: FP4-close to the torch oracle, written directly (accumulate=False) and added to
    a nonzero BF16 seed (accumulate=True)."""
    _skip_if_not_supported()
    from flashinfer.fused_moe import alphamoe_nvfp4_sm100 as api

    route, _ = ROUTES[m]
    case, kwargs = _oracle_case(api, m, 28150 + m)
    generator = torch.Generator(device="cuda").manual_seed(28160 + m)
    seed_out = (
        torch.randn(
            (m, case["K"]), dtype=torch.float32, device="cuda", generator=generator
        )
        * 4.0
    ).to(torch.bfloat16)
    assert _route_of(api, kwargs, seed_out) == route

    direct = torch.full_like(seed_out, float("nan"))
    api.alphamoe_nvfp4_routed_moe(
        out=direct, accumulate=False, **_complete_route_workspaces(m), **kwargs
    )
    seeded = seed_out.clone()
    api.alphamoe_nvfp4_routed_moe(out=seeded, **_complete_route_workspaces(m), **kwargs)
    torch.cuda.synchronize()

    assert torch.isfinite(direct.float()).all()
    _assert_nvfp4_close(
        direct, _reference(case), f"route {route} vs torch oracle (accumulate=False)"
    )
    _assert_nvfp4_close(
        seeded,
        _reference(case, out_init=seed_out),
        f"route {route} vs torch oracle (seeded)",
    )
