# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""GPU (SM100): the FROST MTP verify form (fused speculative conv, per-token state layout) against the pure-torch
reference of the whole fused chain, at the Qwen3.8-27B geometry over requests in {1, 2, 4} and T in {4, 6}.

Tolerances as in ``test_frost_decode.py`` (the ones the kernels were validated with against the unfused chain). Each
shape also runs under the 148-SM (B200) launch policy: on a smaller part the device policy takes the register-capped
two-CTAs-per-SM path everywhere, the B200 policy the one-CTA-per-SM path a B200 runs below one wave.
"""

import pytest
import torch

from flashinfer.qwen_3_5_gdn_block import (
    Qwen35GdnGeometry,
    prepare_qwen_3_5_gdn_block,
    qwen_3_5_gdn_block_verify,
)
from tests.experimental.qwen_3_5_gdn_block.reference import (
    QWEN38_27B,
    make_case,
    quant_fp8,
    reference,
    same_bytes,
)

GEOMETRY = Qwen35GdnGeometry(**QWEN38_27B)
B200_SMS = 148


def _tensors(case, ssm, conv):
    return dict(
        qkv=case.qkv,
        z=case.z,
        hidden=case.hidden,
        ba_weight=case.ba_weight,
        A_log=case.A_log,
        dt_bias=case.dt_bias,
        ssm_state=ssm,
        state_indices=case.state_indices,
        cu_seqlens=case.cu_seqlens,
        num_accepted_tokens=case.num_accepted_tokens,
        norm_weight=case.norm_weight,
        out_scale=case.out_scale,
        conv_state=conv,
        conv_weight=case.conv_weight,
    )


def _outputs(rows, device):
    hv, dv = GEOMETRY.num_v_heads, GEOMETRY.head_v_dim
    return (
        torch.empty(rows, hv, dv, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(rows, hv, dv, dtype=torch.bfloat16, device=device),
    )


def _check(case, q, out, ssm, conv):
    ref_out, ref_q, ref_ssm, ref_conv, live = reference(case, "verify")
    diff = (out.float() - ref_out.float()).abs()
    bad = diff > ref_out.float().abs() * 2**-6 + 1e-3
    assert not bad.any(), (int(bad.sum()), float(diff.max()))
    assert same_bytes(q[live], ref_q[live]) > 0.999
    assert torch.equal(
        q.view(torch.uint8), quant_fp8(out, case.out_scale).view(torch.uint8)
    )
    assert not q[~live].view(torch.uint8).any() and not out[~live].any()
    assert torch.equal(conv, ref_conv)
    idx = case.state_indices
    written = torch.zeros(ssm.shape[0], dtype=torch.bool, device=ssm.device)
    written[idx[idx > 0].long()] = True
    assert torch.equal(ssm[~written], case.ssm_state[~written])
    err = float((ssm[written] - ref_ssm[written]).abs().max())
    assert err < 1e-4, err


@pytest.mark.parametrize("sm_policy", ["device", "b200"])
@pytest.mark.parametrize("T", [4, 6])
@pytest.mark.parametrize("requests", [1, 2, 4])
def test_verify_matches_reference(frost_device, requests, T, sm_policy):
    case = make_case(
        requests, T, seed=100 * requests + T, device=frost_device, form="verify"
    )
    rows = requests * T
    plan = None
    if sm_policy == "b200":
        plan = prepare_qwen_3_5_gdn_block(
            "verify",
            geometry=GEOMETRY,
            device=frost_device,
            requests=requests,
            tokens_per_request=T,
        )
        plan._runner.kernel.sms = (
            B200_SMS  # the wrapper's launch policy input, before the first compile
        )
    ssm, conv = case.ssm_state.clone(), case.conv_state.clone()
    q, out = _outputs(rows, frost_device)
    ret = qwen_3_5_gdn_block_verify(
        **_tensors(case, ssm, conv), geometry=GEOMETRY, out=q, out_bf16=out, plan=plan
    )
    assert ret is q
    torch.cuda.synchronize()
    _check(case, q, out, ssm, conv)


def test_verify_cuda_graph_replay_matches_eager(frost_device):
    requests, T = 2, 4
    rows = requests * T
    case = make_case(requests, T, seed=5, device=frost_device, form="verify")
    plan = prepare_qwen_3_5_gdn_block(
        "verify",
        geometry=GEOMETRY,
        device=frost_device,
        requests=requests,
        tokens_per_request=T,
    )
    ssm_a, conv_a = case.ssm_state.clone(), case.conv_state.clone()
    q_a, out_a = _outputs(rows, frost_device)
    plan.run(**_tensors(case, ssm_a, conv_a), out=q_a, out_bf16=out_a)

    ssm_b, conv_b = case.ssm_state.clone(), case.conv_state.clone()
    q_b, out_b = _outputs(rows, frost_device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(**_tensors(case, ssm_b, conv_b), out=q_b, out_bf16=out_b)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q_a.view(torch.uint8), q_b.view(torch.uint8))
    assert torch.equal(out_a, out_b)
    assert torch.equal(ssm_a, ssm_b) and torch.equal(conv_a, conv_b)
    _check(case, q_b, out_b, ssm_b, conv_b)


def test_verify_declines_a_conv_pool_it_cannot_fuse(frost_device):
    """A conv pool stored ``[slots, C, T + 2]`` (channels not innermost) is a typed decline before any launch."""
    case = make_case(
        1, 4, seed=3, device=frost_device, form="verify", conv_dim_first=True
    )
    with pytest.raises(NotImplementedError, match="channel-contiguous"):
        qwen_3_5_gdn_block_verify(
            **_tensors(case, case.ssm_state.clone(), case.conv_state.clone()),
            geometry=GEOMETRY,
        )
