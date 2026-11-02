"""Source geometry (N = 256, K = 7168, E = 256, top-8, block_m 8) at M = 8: the routed entry takes the route-9 fast chain
(in-kernel-aligned prepared-W1 up, persistent BF16-route down, no-seed finalize) when the caller owns every route-9 carrier
(prepared W1 scales, prepared W1 data, interleaved W1 scales, prepared W2 scales).

Inputs come from this suite's contract fixture (`_make_case`: controlled FP4 scale exponents, the `decode_m8_n256_e256_k7168_top8`
coordinate) and the independent torch oracle `_reference` of test_alphamoe_nvfp4_sm100.py.

Asserted: route 9 is selected (route 0 without the data/interleaved carriers, and for every other token count); the route-9 output is
finite and within the FP4 tolerance (atol 1.0 / rtol 0.1) of the torch oracle, as is the generic path; the max |difference| between the
route-9 output and the generic path is at most 2 BF16 ulp at the generic output's max magnitude; two consecutive route-9 calls are bitwise
identical (the chain is deterministic: BF16 per-route partials and a fixed-order finalize).

Not asserted: bitwise equality with the generic path. The generic M = 8 path for this geometry (split-K up + split-K merge down with fp32
atomics into the accumulator) is run-to-run nondeterministic in its own output (two runs of the same build on the same inputs differ in
a few elements), so a bitwise gate against it is not meaningful. The deterministic route-9 chain is held to the bound the source rows
carry against the reference kernels (<= 2 BF16 ulp, FP4 tolerance). With uniformly random FP4 bytes and no scale control
(the `_complete_route_case` recipe) the two kernels disagree on about 0.9 % of the elements at |out| ~ 6e4 (each within 1 BF16 ulp
of the oracle); that regime has no oracle here and is outside the contract fixture, so it is not tested.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "_source_route_contract_base",
    Path(__file__).with_name("test_alphamoe_nvfp4_sm100.py"),
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

_M, _N, _K, _E, _TOP_K, _BLOCK_M = 8, 256, 7168, 256, 8, 8
_BALANCEDNESS, _SCALING = 0.8, 2.5


def _case_and_ids(seed: int, m: int = _M):
    case = _base._make_case(
        m, _N, _K, _E, _TOP_K, _BLOCK_M, _BALANCEDNESS, _SCALING, seed
    )
    pair_expert = _base._pair_experts(case)
    topk_ids = pair_expert.reshape(m, _TOP_K).to(torch.int32).contiguous()
    return case, topk_ids


def _routed_kwargs(case, topk_ids):
    return dict(
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
    )


def _workspaces(case):
    return dict(
        sorted_token_ids=torch.empty_like(case["sorted_token_ids"]),
        expert_ids=torch.empty_like(case["expert_ids"]),
        num_tokens_post_padded=torch.empty_like(case["num_tokens_post_padded"]),
        cumsum_buffer=torch.empty((_E + 2,), dtype=torch.int32, device="cuda"),
    )


def _carriers(api, case):
    return dict(
        w1_scale_prepared=api.prepare_nvfp4_w1_scales(case["w1_scale"]),
        w2_scale_prepared=api.prepare_nvfp4_w2_scales(case["w2_scale"]),
        w1_data_prepared=api.prepare_nvfp4_w1_data(case["w1"]),
        w1_scale_prepared_interleaved=api.prepare_nvfp4_w1_scales_interleaved(
            case["w1_scale"]
        ),
    )


def _route_id(api, kw, carriers, out, *, with_data_carriers: bool):
    return api._alphamoe_complete_route_id(
        kw["hidden_states"],
        kw["hidden_states_scale"],
        kw["gemm1_weights"],
        kw["gemm1_weights_scale"],
        kw["gemm2_weights_scale"],
        kw["topk_ids"],
        out,
        kw["top_k"],
        kw["block_m"],
        carriers["w1_scale_prepared"],
        carriers["w1_data_prepared"] if with_data_carriers else None,
        w1_scale_prepared_interleaved=carriers["w1_scale_prepared_interleaved"]
        if with_data_carriers
        else None,
        w2_scale_prepared=carriers["w2_scale_prepared"],
    )


@pytest.mark.parametrize("seed", [28104, 28144])
def test_source_m8_route9_matches_oracle_and_generic(seed):
    """Route 9 on the source geometry: FP4-close to the torch oracle, <= 2 BF16 ulp from the generic path, deterministic."""
    _base._skip_if_not_supported()
    from flashinfer.fused_moe import alphamoe_nvfp4_sm100 as api

    case, topk_ids = _case_and_ids(seed)
    kw = _routed_kwargs(case, topk_ids)
    carriers = _carriers(api, case)
    probe = torch.empty((_M, _K), dtype=torch.bfloat16, device="cuda")
    assert _route_id(api, kw, carriers, probe, with_data_carriers=True) == 9
    assert _route_id(api, kw, carriers, probe, with_data_carriers=False) == 0

    direct = torch.full((_M, _K), float("nan"), dtype=torch.bfloat16, device="cuda")
    api.alphamoe_nvfp4_routed_moe(
        out=direct, accumulate=False, **_workspaces(case), **kw, **carriers
    )
    generic_carriers = {
        k: carriers[k] for k in ("w1_scale_prepared", "w2_scale_prepared")
    }
    generic = torch.zeros((_M, _K), dtype=torch.bfloat16, device="cuda")
    api.alphamoe_nvfp4_routed_moe(
        out=generic, accumulate=False, **_workspaces(case), **kw, **generic_carriers
    )
    torch.cuda.synchronize()
    expected = _base._reference(case)

    assert torch.isfinite(direct.float()).all(), (
        "route-9 source output has non-finite values"
    )
    _base._assert_nvfp4_close(
        direct, expected, f"source route 9 vs torch oracle (seed {seed})"
    )
    _base._assert_nvfp4_close(
        generic, expected, f"source generic path vs torch oracle (seed {seed})"
    )

    generic32, direct32 = generic.float(), direct.float()
    generic_max = generic32.abs().max().item()
    assert generic_max > 0.0
    bf16_ulp = 2.0 ** (
        math.floor(math.log2(generic_max)) - 7
    )  # BF16 keeps 7 stored significand bits
    max_diff = (direct32 - generic32).abs().max().item()
    assert max_diff <= 2.0 * bf16_ulp, (
        f"route 9 differs from the generic path by {max_diff} > 2 ulp ({bf16_ulp} @ |generic| max {generic_max}); "
        f"elements differing: {int((direct32 != generic32).sum())}/{direct32.numel()}"
    )

    again = torch.full_like(direct, float("nan"))
    api.alphamoe_nvfp4_routed_moe(
        out=again, accumulate=False, **_workspaces(case), **kw, **carriers
    )
    torch.cuda.synchronize()
    assert torch.equal(again, direct), (
        "the route-9 source chain must be deterministic run to run"
    )


@pytest.mark.parametrize("m", [1, 4, 16, 128], ids=["m1", "m4", "m16", "m128"])
def test_source_other_token_counts_keep_the_generic_route(m):
    """Only the eight-token source shape is dispatched to route 9; every other token count keeps route 0 (selector only, no launch)."""
    _base._skip_if_not_supported()
    from flashinfer.fused_moe import alphamoe_nvfp4_sm100 as api

    case, topk_ids = _case_and_ids(4300 + m, m=m)
    kw = _routed_kwargs(case, topk_ids)
    carriers = _carriers(api, case)
    probe = torch.empty((m, _K), dtype=torch.bfloat16, device="cuda")
    assert _route_id(api, kw, carriers, probe, with_data_carriers=True) == 0
