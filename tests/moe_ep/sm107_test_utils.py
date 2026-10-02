"""MoE references using canonical weight layouts and the shim's quantizers.

The reference evaluates activation and scaling math without the production
weight transform. Quantization and unpacking helpers are shared with the shim.
"""

import torch

from flashinfer.moe_ep import PrequantizedMoEWeights
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
    scale_to_f32,
    unpack_fp4_to_f32,
)


def quantize(tensor, kind, norm=1.0):
    if tensor.numel() == 0:
        nvfp4 = kind == "nvfp4"
        dtype = (
            torch.uint8
            if nvfp4
            else torch.float8_e4m3fn
            if kind == "mxfp8_e4m3"
            else torch.float8_e5m2
        )
        return (
            torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] // (2 if nvfp4 else 1)),
                device=tensor.device,
                dtype=dtype,
            ),
            torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] // (16 if nvfp4 else 32)),
                device=tensor.device,
                dtype=torch.float8_e4m3fn if nvfp4 else torch.float8_e8m0fnu,
            ),
        )
    if kind == "nvfp4":
        return quantize_nvfp4_block16(tensor.float(), norm)
    dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    return quantize_mxfp8_block32(tensor.float(), dtype)


def weight_pack(w13, w2, kind, *, scaled=False):
    """Return canonical packed weights and matching global-scale corrections."""
    experts = w13.shape[0]
    if scaled:
        n1 = torch.linspace(1.1, 2.3, experts, device=w13.device)
        n2 = torch.linspace(0.7, 1.9, experts, device=w13.device)
        intermediate_norm = torch.linspace(1.3, 2.7, experts, device=w13.device)
        input_norm = 1.7
        scalars = dict(
            fc1_alpha=(input_norm * n1).reciprocal(),
            fc2_alpha=(intermediate_norm * n2).reciprocal(),
            fc1_norm_const=intermediate_norm,
        )
    else:
        n1 = n2 = torch.ones(experts, device=w13.device)
        input_norm, scalars = 1.0, {}
    q1, sf1 = quantize(w13, kind, n1[:, None, None])
    q2, sf2 = quantize(w2, kind, n2[:, None, None])
    return PrequantizedMoEWeights(q1, q2, sf1, sf2), input_norm, scalars


def canonical_reference(
    xq,
    xsf,
    ids,
    scores,
    pack,
    kind,
    *,
    situ=False,
    early=True,
    scalars=None,
):
    """FP32 math on canonical weights, independent of the kernel's layout transform."""
    scalars = {} if scalars is None else scalars
    if xq.shape[0] == 0:
        return torch.empty(
            (0, pack.w2.shape[1]), device=xq.device, dtype=torch.bfloat16
        )
    vec = 16 if kind == "nvfp4" else 32

    def dequant(data, scale):
        values = unpack_fp4_to_f32(data) if kind == "nvfp4" else data.float()
        return values * scale_to_f32(scale).repeat_interleave(vec, dim=-1)

    x = dequant(xq, xsf)
    w13 = dequant(pack.w13, pack.w13_scale)
    w2 = dequant(pack.w2, pack.w2_scale)
    result = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for e in range(w13.shape[0]):
        row, slot = (ids == e).nonzero(as_tuple=True)
        if not row.numel():
            continue
        fc1 = x[row] @ w13[e].T
        if "fc1_alpha" in scalars:
            fc1 *= scalars["fc1_alpha"][e]
        gate, up = fc1.chunk(2, dim=-1)
        if situ:
            act = (
                1.25
                * torch.tanh(gate / 1.25)
                * torch.sigmoid(gate)
                * 0.75
                * torch.tanh(up / 0.75)
            )
        else:
            act = up * gate * torch.sigmoid(gate)
        if early:
            act *= scores[row, slot, None]
        norm = scalars["fc1_norm_const"][e] if "fc1_norm_const" in scalars else 1.0
        aq, sf = quantize(act, kind, norm)
        term = dequant(aq, sf) @ w2[e].T
        if "fc2_alpha" in scalars:
            term *= scalars["fc2_alpha"][e]
        term = term.bfloat16().float()
        if not early:
            term *= scores[row, slot, None]
        result.index_add_(0, row, term)
    return result.bfloat16()


def assert_reference(actual, expected, kind):
    error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-6)
    assert torch.isfinite(actual).all()
    assert float(error) < (0.06 if kind == "nvfp4" else 0.02), float(error)
