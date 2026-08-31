"""cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4: wiring smoke test (shape/dtype/no-NaN)."""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import is_sm120a_supported

pytestmark = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="cute_dsl not available"
)

MXFP8_MAX = 448.0
E2M1_MAX = 6.0
SF_M_ALIGN = 4  # TMA globalStride alignment (16B box / 4B-per-packed-UE8M0-int32)


def skip_if_not_sm120():
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * (SF_M_ALIGN - 1)) // SF_M_ALIGN * SF_M_ALIGN


def ceil_to_ue8m0(sf):
    bits = sf.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exp = exp.clamp(1, 254)
    return (exp << 23).view(torch.float32)


def mxfp8_act_quantize(x, gran_k=128):
    m, k = x.shape
    blocks = k // gran_k
    xf = x.float().reshape(m, blocks, gran_k)
    sf = ceil_to_ue8m0(xf.abs().amax(dim=-1).clamp_min(1e-4) / MXFP8_MAX)
    q = (xf / sf[..., None]).to(torch.float8_e4m3fn).reshape(m, k)
    return q, sf


def pack_ue8m0_to_int32(sf):
    bits = sf.contiguous().view(torch.int32)
    return (bits >> 23).to(torch.uint8).view(torch.int32)


def pack_mxfp8_moe_sfa(a_sf, offsets):
    packed = pack_ue8m0_to_int32(a_sf)
    m = packed.shape[0]
    na = packed.shape[1]
    off = offsets.tolist()
    num_experts = len(off) - 1
    m_padded = compute_padded_offset(m, num_experts)
    out = torch.zeros(na, m_padded, dtype=torch.int32, device=a_sf.device)
    mn_major = packed.t().contiguous()
    for e in range(num_experts):
        s, en = off[e], off[e + 1]
        if s < en:
            po = compute_padded_offset(s, e)
            out[:, po : po + (en - s)] = mn_major[:, s:en]
    return out.view(torch.uint8)


def _e2m1_code(x):
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
    )
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    return code | (sign.to(torch.uint8) << 3)


def mxfp4_weight_quantize(w, gran_k=32):
    n, k = w.shape
    wv = w.view(n, k // gran_k, gran_k)
    sf = ceil_to_ue8m0(wv.abs().float().amax(dim=2).clamp_min(1e-4) / E2M1_MAX)
    scaled = wv * (1.0 / sf.unsqueeze(2))
    codes = _e2m1_code(scaled).view(n, k)
    c2 = codes.view(n, k // 2, 2)
    packed = (c2[:, :, 0] & 0x0F) | ((c2[:, :, 1] & 0x0F) << 4)
    return packed.contiguous(), sf


def pack_mxfp4_moe_sfb(w_sf_list):
    packed = []
    for sf in w_sf_list:
        p = pack_ue8m0_to_int32(sf).t().contiguous()
        mn = (p.shape[1] + SF_M_ALIGN - 1) // SF_M_ALIGN * SF_M_ALIGN
        packed.append(torch.nn.functional.pad(p, (0, mn - p.shape[1])).contiguous())
    return torch.stack(packed).view(torch.uint8)


def make_inputs(m_per_expert_list, n, k):
    torch.random.manual_seed(0)
    num_experts = len(m_per_expert_list)
    offsets = [0]
    for m_pe in m_per_expert_list:
        offsets.append(offsets[-1] + m_pe)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    a = torch.randn((offsets[-1], k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)

    a_q, a_sf = mxfp8_act_quantize(a, gran_k=128)
    a_scale = pack_mxfp8_moe_sfa(a_sf, m_indptr)

    qs, sfs = [], []
    for e in range(num_experts):
        q, sf = mxfp4_weight_quantize(w[e], gran_k=32)
        qs.append(q)
        sfs.append(sf)
    b_q = torch.stack(qs)
    b_scale = pack_mxfp4_moe_sfb(sfs)
    return a_q, a_scale, b_q, b_scale, m_indptr


def test_cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4_smoke():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4

    num_experts, m_per_expert, hidden = 4, 64, 512
    a_q, a_scale, b_q, b_scale, m_indptr = make_inputs(
        [m_per_expert] * num_experts, hidden, hidden
    )
    num_tokens = num_experts * m_per_expert
    tok = torch.randint(0, num_tokens, (num_tokens,), device="cuda", dtype=torch.int32)
    scales = torch.rand(num_tokens, device="cuda", dtype=torch.float32)
    out = cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
        a_q, a_scale, b_q, b_scale, m_indptr, tok, scales, num_tokens
    )
    assert out.shape == (num_tokens, hidden)
    assert not torch.isnan(out.float()).any()
