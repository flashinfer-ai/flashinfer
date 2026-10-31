# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""GPU (SM100): the FROST decode form against the pure-torch reference of the whole fused chain.

Tolerances are the ones the kernels were validated with against the unfused op chain: FP8 bytes agree on more than
99.9% of the live outputs, the BF16 normed value within 2^-6 relative + 1e-3 (two BF16 roundings after FP32 sums in
another order), the conv state bitwise, written SSM slots within 1e-4 of the FP64 recurrence and every other slot
untouched; padding rows are zero. The FP8 output is also checked bitwise against the static quantization of the
kernel's own BF16 value.
"""

import pytest
import torch

from flashinfer.qwen_3_5_gdn_block import (
    Qwen35GdnGeometry,
    prepare_qwen_3_5_gdn_block,
    qwen_3_5_gdn_block_decode,
)
from tests.experimental.qwen_3_5_gdn_block.reference import (
    QWEN38_27B,
    make_case,
    quant_fp8,
    reference,
    same_bytes,
)

GEOMETRY = Qwen35GdnGeometry(**QWEN38_27B)
ROWS = [1, 2, 3, 4, 8, 16, 64]


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
    ref_out, ref_q, ref_ssm, ref_conv, live = reference(case, "decode")
    diff = (out.float() - ref_out.float()).abs()
    bad = diff > ref_out.float().abs() * 2**-6 + 1e-3
    assert not bad.any(), (int(bad.sum()), float(diff.max()))
    assert same_bytes(q[live], ref_q[live]) > 0.999
    assert torch.equal(
        q.view(torch.uint8), quant_fp8(out, case.out_scale).view(torch.uint8)
    )
    assert not q[~live].view(torch.uint8).any() and not out[~live].any()
    assert torch.equal(conv, ref_conv)
    written = torch.zeros(ssm.shape[0], dtype=torch.bool, device=ssm.device)
    written[case.state_indices[case.state_indices > 0].long()] = True
    assert torch.equal(ssm[~written], case.ssm_state[~written])
    err = float((ssm[written] - ref_ssm[written]).abs().max())
    assert err < 1e-4, err


@pytest.mark.parametrize("rows", ROWS)
def test_decode_matches_reference(frost_device, rows):
    case = make_case(rows, 1, seed=rows, device=frost_device, form="decode")
    ssm, conv = case.ssm_state.clone(), case.conv_state.clone()
    q, out = _outputs(rows, frost_device)
    ret = qwen_3_5_gdn_block_decode(
        **_tensors(case, ssm, conv), geometry=GEOMETRY, out=q, out_bf16=out
    )
    assert ret is q
    torch.cuda.synchronize()
    _check(case, q, out, ssm, conv)


def test_decode_fp32_params_and_dim_first_conv_pool(frost_device):
    """FP32 dt_bias and norm weight (another plan), a conv pool stored ``[slots, C, 3]``, the plan's own FP8 output."""
    case = make_case(
        4,
        1,
        seed=7,
        device=frost_device,
        form="decode",
        param_dtypes=(torch.float32, torch.float32, torch.float32),
        conv_dim_first=True,
    )
    ssm, conv = case.ssm_state.clone(), case.conv_state.clone()
    _, out = _outputs(4, frost_device)
    q = qwen_3_5_gdn_block_decode(
        **_tensors(case, ssm, conv), geometry=GEOMETRY, out_bf16=out
    )
    torch.cuda.synchronize()
    _check(case, q, out, ssm, conv)


def test_decode_cuda_graph_replay_matches_eager(frost_device):
    rows = 4
    case = make_case(rows, 1, seed=11, device=frost_device, form="decode")
    plan = prepare_qwen_3_5_gdn_block(
        "decode", geometry=GEOMETRY, device=frost_device, requests=rows
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
