"""SM12x MXFP8 x MXFP4 internal MoE ops against pure-Torch references."""

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

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


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum().item()
    return 0.0 if denom == 0 else 1.0 - 2.0 * (x * y).sum().item() / denom


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * (SF_M_ALIGN - 1)) // SF_M_ALIGN * SF_M_ALIGN


def ceil_to_ue8m0(sf):
    bits = sf.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exp = exp.clamp(1, 254)
    return (exp << 23).view(torch.float32)


def mxfp8_act_quantize(x, gran_k=128):
    """Per-token E4M3 with a per-gran_k-block UE8M0 scale; sf is [M, K/gran_k]."""
    m, k = x.shape
    blocks = k // gran_k
    xf = x.float().reshape(m, blocks, gran_k)
    sf = ceil_to_ue8m0(xf.abs().amax(dim=-1).clamp_min(1e-4) / MXFP8_MAX)
    q = (xf / sf[..., None]).to(torch.float8_e4m3fn).reshape(m, k)
    return q, sf


def pack_ue8m0_to_int32(sf):
    bits = sf.contiguous().view(torch.int32)
    exponents = (bits >> 23).to(torch.uint8)
    exponents = F.pad(exponents, (0, -exponents.shape[-1] % 4), value=127)
    return exponents.view(torch.int32)


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


def dequant_e2m1_codes(codes):
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=codes.device,
        dtype=torch.float32,
    )
    idx = (codes & 0x07).to(torch.long)
    sign = ((codes & 0x08) != 0) & (idx != 0)
    val = values[idx]
    return torch.where(sign, -val, val)


def ref_moe_gemm(a_q, a_sf, b_q, b_sf, offsets, out_dtype=torch.bfloat16):
    """Pure-torch truth from the natural (unpacked) scales, not the kernel-ABI packed ones."""
    m, k = a_q.shape
    a_deq = (a_q.float().reshape(m, k // 128, 128) * a_sf[:, :, None]).reshape(m, k)
    out = torch.zeros(m, b_q.shape[1], device=a_q.device, dtype=torch.float32)
    off = offsets.tolist()
    for e in range(len(off) - 1):
        s, en = off[e], off[e + 1]
        if s < en:
            n, _kh = b_q[e].shape
            codes = torch.zeros(n, k, dtype=torch.uint8, device=a_q.device)
            codes[:, 0::2] = b_q[e] & 0x0F
            codes[:, 1::2] = (b_q[e] >> 4) & 0x0F
            b_deq = (
                dequant_e2m1_codes(codes).reshape(n, k // 32, 32) * b_sf[e][:, :, None]
            ).reshape(n, k)
            out[s:en] = a_deq[s:en] @ b_deq.t()
    return out.to(out_dtype)


def make_inputs(m_per_expert_list, n, k, ref_dtype=torch.bfloat16):
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

    ref = ref_moe_gemm(a_q, a_sf, b_q, sfs, m_indptr, ref_dtype)
    return a_q, a_scale, b_q, b_scale, m_indptr, ref


def test_cute_dsl_sm12x_moe_gemm_mxfp8_mxfp4_matches_reference():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_moe_gemm_mxfp8_mxfp4

    a_q, a_scale, b_q, b_scale, m_indptr, ref = make_inputs([64, 64, 64, 64], 512, 512)
    out = cute_dsl_sm12x_moe_gemm_mxfp8_mxfp4(a_q, a_scale, b_q, b_scale, m_indptr)
    diff = calc_diff(out.float(), ref.float())
    assert diff < 5e-3, f"calc_diff={diff:.6e}"


def make_gated_inputs(m_per_expert_list, n, k):
    a, a_sf, b, b_sf, offsets, ref2n = make_inputs(
        m_per_expert_list, 2 * n, k, torch.float32
    )
    return (
        a,
        a_sf,
        b,
        b_sf,
        offsets,
        (F.silu(ref2n[:, n:]) * ref2n[:, :n]).to(torch.bfloat16),
    )


def test_cute_dsl_sm12x_fc1_act_mxfp8_mxfp4_matches_reference():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_mxfp8_mxfp4

    a, a_sf, b, b_sf, offsets, ref = make_gated_inputs([64] * 4, 512, 512)
    out = cute_dsl_sm12x_fc1_act_mxfp8_mxfp4(a, a_sf, b, b_sf, offsets)
    assert calc_diff(out.float(), ref.float()) < 8e-3


def test_cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4_smoke():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4

    a, a_sf, b, b_sf, offsets, _ = make_gated_inputs([64] * 4, 512, 512)
    q, sf = cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(a, a_sf, b, b_sf, offsets)
    assert not torch.isnan(q.float()).any() and sf.numel() > 0


def test_cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4_smoke():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4

    a, a_sf, b, b_sf, offsets, _ = make_inputs([64] * 4, 512, 512)
    rows = a.shape[0]
    tok = torch.randint(0, rows, (rows,), device="cuda", dtype=torch.int32)
    out = cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
        a, a_sf, b, b_sf, offsets, tok, torch.rand(rows, device="cuda"), rows
    )
    assert out.shape == (rows, 512) and not torch.isnan(out.float()).any()


@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("tile_m", [64, 128])
def test_fc2_finalize_ignores_bytes_outside_logical_scale_tensor(k: int, tile_m: int):
    """Padded TMA rows must not read poisoned bytes outside the scale operand."""
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4

    a, a_sf, b, b_sf, offsets, gemm_ref = make_inputs([64] * 4, 512, k)
    assert a_sf.dtype == torch.uint8 and a_sf.is_contiguous()
    # Preserve the logical tensor's values, shape, strides, and alignment.
    # Extra backing storage makes an overstated descriptor read NaN scales
    # deterministically, without depending on a neighboring allocation.
    backing = torch.full(
        (4 * a_sf.numel() + 512,), 0xFF, dtype=torch.uint8, device=a_sf.device
    )
    logical_scales = backing[: a_sf.numel()].view(a_sf.shape)
    logical_scales.copy_(a_sf)
    assert logical_scales.stride() == a_sf.stride()
    assert logical_scales.data_ptr() % 16 == 0
    rows = a.shape[0]
    tokens = torch.arange(rows, device=a.device, dtype=torch.int32)
    route_weights = torch.rand(rows, device=a.device)
    actual = cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
        a,
        logical_scales,
        b,
        b_sf,
        offsets,
        tokens,
        route_weights,
        rows,
        tile=(tile_m, 128, 128),
        tune=False,
    )
    torch.cuda.synchronize()
    expected = (gemm_ref.float() * route_weights[:, None]).to(actual.dtype)
    assert torch.isfinite(actual).all()
    diff = calc_diff(actual.float(), expected.float())
    assert diff < 5e-3, f"calc_diff={diff:.6e}"


def poisoned_scale_view(a_sf, dtype):
    """Preserve packed scale bytes and poison only backing beyond the tensor."""
    assert a_sf.dtype == torch.uint8 and a_sf.is_contiguous()
    backing = torch.full(
        (4 * a_sf.numel() + 512,), 0xFF, dtype=torch.uint8, device=a_sf.device
    )
    logical = backing[: a_sf.numel()].view(a_sf.shape)
    logical.copy_(a_sf)
    assert logical.stride() == a_sf.stride()
    assert logical.data_ptr() % 16 == 0
    return logical.view(dtype)


def dequantize_mxfp8_moe_output(q, sf, offsets):
    """Unpack Q1's expert-padded UE8M0 scales for a numerical reference check."""
    assert sf.dtype == torch.int32 and sf.is_contiguous()
    encoded = sf.view(torch.uint8).reshape(sf.shape[0], sf.shape[1], 4)
    encoded = encoded.permute(1, 0, 2).reshape(sf.shape[1], -1)
    scales = torch.ldexp(
        torch.ones_like(encoded, dtype=torch.float32), encoded.int() - 127
    )
    actual = torch.empty_like(q, dtype=torch.float32)
    off = offsets.tolist()
    blocks = q.shape[1] // 128
    for e, (start, end) in enumerate(zip(off[:-1], off[1:], strict=True)):
        if start < end:
            padded = compute_padded_offset(start, e)
            actual[start:end] = (
                q[start:end].float().reshape(end - start, blocks, 128)
                * scales[padded : padded + end - start, :blocks, None]
            ).reshape(end - start, q.shape[1])
    return actual


@pytest.mark.parametrize("op_name", ["gemm", "fc1", "fc1_q1"])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("tile_m", [8, 64])
@pytest.mark.parametrize(
    "scale_dtype", [torch.uint8, torch.int32], ids=["uint8", "int32"]
)
def test_mxfp8_mxfp4_sibling_scale_extent(op_name, k, tile_m, scale_dtype):
    """Check scale extent and K-plane stride for normal and swapped operands."""
    skip_if_not_sm120()
    from flashinfer.fused_moe import (
        cute_dsl_sm12x_fc1_act_mxfp8_mxfp4,
        cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
        cute_dsl_sm12x_moe_gemm_mxfp8_mxfp4,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.moe_epilogue import (
        EpiMethod,
    )

    make = make_inputs if op_name == "gemm" else make_gated_inputs
    a, a_sf, b, b_sf, offsets, expected = make([64] * 4, 512, k)
    logical_scales = poisoned_scale_view(a_sf, scale_dtype)
    ops = {
        "gemm": cute_dsl_sm12x_moe_gemm_mxfp8_mxfp4,
        "fc1": cute_dsl_sm12x_fc1_act_mxfp8_mxfp4,
        "fc1_q1": cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
    }
    if op_name == "fc1_q1":
        epi = EpiMethod.WG_QUANT_SWAP if tile_m == 8 else EpiMethod.WG_S2R_QUANT
    else:
        epi = EpiMethod.DIRECT_STG if tile_m == 8 else EpiMethod.R2G_WG
    actual = ops[op_name](
        a,
        logical_scales,
        b,
        b_sf,
        offsets,
        tile=(tile_m, 128, 128),
        epi=epi,
        tune=False,
    )
    if op_name == "fc1_q1":
        q, sf = actual
        assert torch.isfinite(q.float()).all()
        actual = dequantize_mxfp8_moe_output(q, sf, offsets)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    diff = calc_diff(actual.float(), expected.float())
    limit = 5e-3 if op_name == "gemm" else 8e-3
    assert diff < limit, f"calc_diff={diff:.6e}"


@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("tile_m", [64, 128])
def test_fc2_finalize_int32_scale_extent(k, tile_m):
    """Keep the already-packed Int32 scale ABI valid when fixing byte views."""
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4

    a, a_sf, b, b_sf, offsets, gemm_ref = make_inputs([64] * 4, 512, k)
    logical_scales = poisoned_scale_view(a_sf, torch.int32)
    rows = a.shape[0]
    route_weights = torch.rand(rows, device=a.device)
    actual = cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
        a,
        logical_scales,
        b,
        b_sf,
        offsets,
        torch.arange(rows, device=a.device, dtype=torch.int32),
        route_weights,
        rows,
        tile=(tile_m, 128, 128),
        tune=False,
    )
    torch.cuda.synchronize()
    expected = (gemm_ref.float() * route_weights[:, None]).to(actual.dtype)
    assert torch.isfinite(actual).all()
    diff = calc_diff(actual.float(), expected.float())
    assert diff < 5e-3, f"calc_diff={diff:.6e}"


@pytest.mark.parametrize("op_name", ["gemm", "fc1", "fc1_q1", "fc2"])
@pytest.mark.parametrize(
    "first_dtype", [torch.uint8, torch.int32], ids=["uint8-first", "int32-first"]
)
def test_mxfp8_mxfp4_scale_dtype_cache_transition(op_name, first_dtype, monkeypatch):
    """One operation must accept both packed scale views in either call order."""
    skip_if_not_sm120()
    import importlib

    suffix = {
        "gemm": "gemm",
        "fc1": "fc1_act",
        "fc1_q1": "fc1_act_q1",
        "fc2": "fc2_finalize",
    }[op_name]
    module = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_" + suffix
    )
    api_name = "moe_gemm" if op_name == "gemm" else suffix
    op = getattr(module, "cute_dsl_sm12x_" + api_name + "_mxfp8_mxfp4")
    # Isolate this call-order test, but preserve the cache across both calls.
    monkeypatch.setattr(module, "_COMPILED", {})
    make = make_gated_inputs if op_name in ("fc1", "fc1_q1") else make_inputs
    a, a_sf, b, b_sf, offsets, expected = make([64] * 4, 512, 1024)
    rows = a.shape[0]
    tokens = torch.arange(rows, device=a.device, dtype=torch.int32)
    route_weights = torch.rand(rows, device=a.device)
    if op_name == "fc2":
        expected = (expected.float() * route_weights[:, None]).to(torch.bfloat16)
    second_dtype = torch.int32 if first_dtype == torch.uint8 else torch.uint8
    for scale_dtype in (first_dtype, second_dtype):
        args = (
            a,
            poisoned_scale_view(a_sf, scale_dtype),
            b,
            b_sf.view(scale_dtype),
            offsets,
        )
        if op_name == "fc2":
            args += (tokens, route_weights, rows)
        actual = op(*args, tile=(64, 128, 128), tune=False)
        if op_name == "fc1_q1":
            q, sf = actual
            assert torch.isfinite(q.float()).all()
            actual = dequantize_mxfp8_moe_output(q, sf, offsets)
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        diff = calc_diff(actual.float(), expected.float())
        limit = 8e-3 if op_name in ("fc1", "fc1_q1") else 5e-3
        assert diff < limit, f"{scale_dtype}: calc_diff={diff:.6e}"


def _gather_inputs(bm, k, counts=None, *, n=128, activation=None, seed=19, tokens=19):
    torch.manual_seed(seed)
    if counts is None:
        counts = [0, 1, bm - 1, bm, bm + 1]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    token_x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    token_q, token_sf = mxfp8_act_quantize(token_x)
    token_map = torch.randint(
        0, tokens, (offsets[-1],), dtype=torch.int32, device="cuda"
    )
    indices = token_map.long()
    routed_q = (
        token_q.view(torch.uint8).index_select(0, indices).view(torch.float8_e4m3fn)
    )
    routed_sf = token_sf.index_select(0, indices)
    weights = torch.randn(
        len(counts), 2 * n, k, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    quantized = [mxfp4_weight_quantize(weight) for weight in weights]
    bq = torch.stack([item[0] for item in quantized])
    bsf = pack_mxfp4_moe_sfb([item[1] for item in quantized])
    ref = ref_moe_gemm(
        routed_q,
        routed_sf,
        bq,
        [item[1] for item in quantized],
        indptr,
        out_dtype=torch.float32,
    )
    linear, gate = ref[:, :n], ref[:, n:]
    ref = (
        F.silu(gate) * linear if activation is None else activation(linear, gate)
    ).to(torch.bfloat16)
    return (
        token_q,
        pack_ue8m0_to_int32(token_sf).t().contiguous(),
        routed_q,
        pack_mxfp8_moe_sfa(routed_sf, indptr).view(torch.int32),
        bq,
        bsf,
        indptr,
        token_map,
        ref,
    )


def _dequant_q1(q, sf, indptr):
    out = torch.empty_like(q, dtype=torch.float32)
    offsets = indptr.tolist()
    for expert, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        if begin == end:
            continue
        start = compute_padded_offset(begin, expert)
        packed_rows = torch.empty(
            (end - begin, sf.shape[0]), dtype=sf.dtype, device=sf.device
        )
        packed_rows.copy_(sf[:, start : start + end - begin].t())
        exponents = packed_rows.view(torch.uint8)
        scales = (exponents.to(torch.int32) << 23).view(torch.float32)
        scales = scales[:, : q.shape[1] // 128].repeat_interleave(128, dim=1)
        out[begin:end] = q[begin:end].float() * scales
    return out


@pytest.mark.parametrize(
    "bm,epi_name,k",
    [
        (8, "WG_QUANT_SWAP", 512),
        (32, "WG_S2R_QUANT", 512),
        (32, "WG_QUANT_R2S", 512),
        (64, "WG_S2R_QUANT", 512),
        (64, "WG_QUANT_R2S", 512),
        (8, "WG_QUANT_SWAP", 128),
    ],
)
def test_fc1_q1_gather_matches_routed_and_reference(bm, epi_name, k):
    skip_if_not_sm120()
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    token_q, token_sf, routed_q, routed_sf, bq, bsf, indptr, token_map, ref = (
        _gather_inputs(bm, k)
    )
    epi = getattr(op.EpiMethod, epi_name)
    kwargs = dict(tile=(bm, 128, 128), epi=epi, tune=False)
    gather = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4
    q, sf = gather(token_q, token_sf, bq, bsf, indptr, token_map=token_map, **kwargs)
    q_routed, sf_routed = gather(routed_q, routed_sf, bq, bsf, indptr, **kwargs)
    assert torch.equal(q.view(torch.uint8), q_routed.view(torch.uint8))
    assert torch.equal(sf, sf_routed)
    ref_q, ref_sf = mxfp8_act_quantize(ref)
    ref_deq = ref_q.float() * ref_sf.repeat_interleave(128, dim=1)
    assert calc_diff(_dequant_q1(q, sf, indptr), ref_deq) < 8e-3
    out_q, out_sf = torch.empty_like(q), torch.zeros_like(sf)
    q_reuse, sf_reuse = gather(
        token_q,
        token_sf,
        bq,
        bsf,
        indptr,
        token_map=token_map,
        out_q=out_q,
        out_sf=out_sf,
        **kwargs,
    )
    assert q_reuse.data_ptr() == out_q.data_ptr()
    assert sf_reuse.data_ptr() == out_sf.data_ptr()
    assert torch.equal(q_reuse.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(sf_reuse, sf)
    keys = []
    for inputs in (
        [routed_q, routed_sf, bq, bsf, indptr],
        [token_q, token_sf, bq, bsf, indptr, token_map],
    ):
        runner = op._Fc1ActQ1Runner(torch.float8_e4m3fn, op.ActivationType.Swiglu)
        tactic = (bm, 128, epi)
        assert tactic in runner.get_valid_tactics(inputs, None)
        prepared = runner.forward(inputs, tactic=tactic, do_preparation=True)
        assert prepared[0].shape == q.shape and prepared[1].shape == sf.shape
        actual_q, actual_sf = runner.forward(inputs, tactic=tactic)
        assert torch.equal(actual_q.view(torch.uint8), q.view(torch.uint8))
        assert torch.equal(actual_sf, sf)
        keys.append(runner.get_cache_key_extras(inputs))
    assert keys[0] != keys[1]


@pytest.mark.parametrize(
    "bm,epi_name", [(8, "WG_QUANT_SWAP"), (32, "WG_QUANT_R2S"), (64, "WG_S2R_QUANT")]
)
def test_fc1_gather_crosses_sf_and_scheduler_rings(bm, epi_name):
    skip_if_not_sm120()
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.moe_scheduler import (
        MoeSchedStages,
    )

    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tasks = sms * (MoeSchedStages + 1) + 1
    aq, asf, routed_q, routed_sf, bq, bsf, indptr, token_map, _ = _gather_inputs(
        bm, 3584, counts=[bm * tasks + 1, 0, 3]
    )
    kwargs = dict(tile=(bm, 128, 128), epi=getattr(op.EpiMethod, epi_name), tune=False)
    fn = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4
    q, sf = fn(aq, asf, bq, bsf, indptr, token_map=token_map, **kwargs)
    expected_q, expected_sf = fn(routed_q, routed_sf, bq, bsf, indptr, **kwargs)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(sf, expected_sf)


def test_fc1_gather_activation_offset_exceeds_int32():
    skip_if_not_sm120()
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    _, token_sf, routed_q, routed_sf, bq, bsf, indptr, small_map, _ = _gather_inputs(
        8, 4096, counts=[1]
    )
    tokens = (1 << 31) // 4096 + 1
    # Only the final row is initialized and read; its byte offset exceeds Int32.
    aq = torch.empty(tokens, 4096, dtype=torch.float8_e4m3fn, device="cuda")
    asf = torch.empty(8, tokens, dtype=torch.int32, device="cuda")
    aq[-1:].view(torch.uint8).copy_(routed_q.view(torch.uint8))
    asf[:, -1:].copy_(token_sf.index_select(1, small_map.long()))
    token_map = torch.tensor([tokens - 1], dtype=torch.int32, device="cuda")
    kwargs = dict(tile=(8, 128, 128), epi=op.EpiMethod.WG_QUANT_SWAP, tune=False)
    fn = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4
    q, sf = fn(aq, asf, bq, bsf, indptr, token_map=token_map, **kwargs)
    expected_q, expected_sf = fn(routed_q, routed_sf, bq, bsf, indptr, **kwargs)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(sf, expected_sf)


@pytest.mark.parametrize("bm", (8, 32))
@pytest.mark.parametrize("byte_offset", (1, 16))
def test_fc1_q1_gather_activation_alignment(bm, byte_offset, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_mxfp8_mxfp4_fc1_act_q1 as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    a_q, a_sf, _, _, b_q, b_sf, offsets, mapping, ref = _gather_inputs(
        bm, 512, counts=[0, 2, 0], tokens=2
    )
    storage = torch.empty(
        a_q.numel() + byte_offset, dtype=torch.uint8, device=a_q.device
    )
    shifted = storage[byte_offset:].view(a_q.dtype).view_as(a_q)
    shifted.view(torch.uint8).copy_(a_q.view(torch.uint8))
    assert shifted.is_contiguous() and shifted.storage_offset() == byte_offset
    assert shifted.data_ptr() % 16 == byte_offset % 16
    epi = entry.EpiMethod.WG_QUANT_SWAP if bm == 8 else entry.EpiMethod.WG_S2R_QUANT
    kwargs = dict(
        token_map=mapping, tile=(bm, 128, 128), epi=epi, split_k=1, tune=False
    )
    if byte_offset % 16:

        def unexpected_compile(*args, **kwargs):
            pytest.fail("misaligned gather activation reached kernel compilation")

        monkeypatch.setattr(entry, "compiled_kernel", unexpected_compile)
        with pytest.raises(
            ValueError, match="gather activation must be 16-byte aligned"
        ):
            entry.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
                shifted, a_sf, b_q, b_sf, offsets, **kwargs
            )
        return
    q, sf = entry.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        shifted, a_sf, b_q, b_sf, offsets, **kwargs
    )
    q_base, sf_base = entry.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        a_q, a_sf, b_q, b_sf, offsets, **kwargs
    )
    assert torch.equal(q.view(torch.uint8), q_base.view(torch.uint8))
    assert torch.equal(sf, sf_base)
    ref_q, ref_sf = mxfp8_act_quantize(ref)
    ref_deq = ref_q.float() * ref_sf.repeat_interleave(128, dim=1)
    assert calc_diff(_dequant_q1(q, sf, offsets), ref_deq) < 8e-3


@pytest.mark.parametrize("invalid", ["dtype", "rank", "stride", "device"])
def test_fc1_q1_gather_rejects_invalid_map(invalid):
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4

    aq = torch.zeros(2, 512, dtype=torch.float8_e4m3fn, device="cuda")
    asf = torch.zeros(1, 2, dtype=torch.int32, device="cuda")
    bq = torch.zeros(1, 256, 256, dtype=torch.uint8, device="cuda")
    bsf = torch.zeros(1, 4, 256, dtype=torch.int32, device="cuda")
    offsets = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    token_map = torch.arange(2, dtype=torch.int32, device="cuda")
    if invalid == "dtype":
        token_map = token_map.long()
    elif invalid == "rank":
        token_map = token_map.view(1, 2)
    elif invalid == "stride":
        token_map = torch.arange(4, dtype=torch.int32, device="cuda")[::2]
    else:
        token_map = token_map.cpu()
    with pytest.raises(ValueError, match="token_map must be contiguous rank-1 int32"):
        cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
            aq, asf, bq, bsf, offsets, token_map=token_map, tune=False
        )


def _assert_fc2_finalize(counts, k, bm, unique_tokens=False):
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize"
    )
    aq, asf, bq, bsf, offsets, ref = make_inputs(counts, 128, k)
    rows = aq.shape[0]
    num_tokens = rows if unique_tokens else 17
    tokens = (
        torch.randperm(rows, device="cuda").to(torch.int32)
        if unique_tokens
        else torch.arange(rows, device="cuda", dtype=torch.int32) % num_tokens
    )
    scales = torch.linspace(0.1, 1.0, rows, device="cuda")
    out = op.cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
        aq,
        asf,
        bq,
        bsf,
        offsets,
        tokens,
        scales,
        num_tokens,
        tile=(bm, 128, 128),
        tune=False,
    )
    expected = torch.zeros(num_tokens, 128, dtype=torch.float32, device="cuda")
    expected.index_add_(0, tokens.long(), ref.float() * scales[:, None])
    assert calc_diff(out.float(), expected) < 5e-3
    cfg = op.CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp(128, k, (bm, 128, 128))
    assert cfg.pingpong == (bm in (32, 64) and k <= 1024)


@pytest.mark.parametrize(
    "bm,k",
    [(32, 384), (32, 1024), (32, 2048), (64, 384), (64, 1024), (64, 2048), (128, 512)],
)
def test_fc2_finalize_schedules_match_reference(bm, k):
    skip_if_not_sm120()
    _assert_fc2_finalize([bm + 1, 0, 1, bm - 1], k, bm)


def test_fc2_pingpong_crosses_scheduler_ring():
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.moe_scheduler import (
        MoeSchedStages,
    )

    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tasks = sms * (MoeSchedStages + 1) + 1
    _assert_fc2_finalize([64 * tasks + 1, 0, 3], 384, 64, unique_tokens=True)


@pytest.mark.parametrize(
    "k,rows,bm", [(384, 96, 64), (1024, 96, 64), (2048, 96, 128), (384, 32, 32)]
)
def test_fc2_default_and_runner_select_real_k(k, rows, bm, monkeypatch):
    skip_if_not_sm120()
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize"
    )
    aq, asf, bq, bsf, offsets, _ = make_inputs([rows], 128, k)
    tokens = torch.arange(rows, dtype=torch.int32, device="cuda")
    scales = torch.ones(rows, device="cuda")
    output = torch.zeros(rows, 128, dtype=torch.bfloat16, device="cuda")
    observed = []

    def capture_compile(sample_args, *, op, grid_x, sm_version):
        observed.append((op.tile, op.k, op.pingpong))
        return lambda *args: None

    monkeypatch.setattr(op, "compiled_kernel", capture_compile)
    inputs = [aq, asf, bq, bsf, offsets, tokens, scales, rows]
    op.cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(*inputs, out=output, tune=False)
    op._Fc2FinalizeRunner().launch(inputs, output, None)
    assert observed == [((bm, 128, 128), k, bm in (32, 64) and k <= 1024)] * 2


def _split_k_data(
    k,
    n,
    counts,
    kind="swiglu",
    limit=None,
    beta=4.0,
    linear_beta=25.0,
    seed=19,
    tokens=19,
):
    def activation(linear, gate):
        if kind == "situ":
            return (
                beta
                * torch.tanh(gate / beta)
                * torch.sigmoid(gate)
                * linear_beta
                * torch.tanh(linear / linear_beta)
            )
        if limit is not None:
            linear, gate = linear.clamp(-limit, limit), gate.clamp(max=limit)
        return F.silu(gate) * linear

    aq, asf, _, _, bq, bsf, offsets, mapping, reference = _gather_inputs(
        8, k, counts, n=n, activation=activation, seed=seed, tokens=tokens
    )
    mapping = torch.cat(
        (mapping, torch.full((3,), -1, dtype=torch.int32, device="cuda"))
    )
    return aq, asf, bq, bsf, offsets, mapping, reference


def _check_split_k_partials(op, inputs, q, sf, activation, limit, beta, linear_beta):
    aq, asf, bq, bsf, offsets, mapping = inputs
    experts, channels, packed_k = bq.shape
    k, n, rows = 2 * packed_k, channels // 2, mapping.numel()
    sx = asf.t().contiguous().view(torch.uint8).reshape(aq.shape[0], -1)[:, : k // 128]
    sw = (
        bsf.reshape(experts, -1, channels, 4)
        .permute(0, 2, 1, 3)
        .reshape(experts, channels, -1)[:, :, : k // 32]
    )
    x = aq.float() * torch.exp2(sx.float() - 127).repeat_interleave(128, dim=1)
    codes = torch.empty(experts, channels, k, dtype=torch.uint8, device=bq.device)
    codes[:, :, 0::2], codes[:, :, 1::2] = bq & 15, bq >> 4
    weight = dequant_e2m1_codes(codes) * torch.exp2(sw.float() - 127).repeat_interleave(
        32, dim=2
    )
    partial = torch.full((2, 2, rows, n), 19.75, dtype=torch.float32, device=aq.device)
    expected = partial.clone()
    values, tokens = offsets.tolist(), mapping.tolist()
    for expert, (begin, end) in enumerate(zip(values[:-1], values[1:], strict=True)):
        for route in range(begin, end):
            for split in range(2):
                lo, hi = (
                    (k // 128 * split // 2) * 128,
                    (k // 128 * (split + 1) // 2) * 128,
                )
                value = x[tokens[route], lo:hi] @ weight[expert, :, lo:hi].t()
                expected[0, split, route], expected[1, split, route] = (
                    value[:n],
                    value[n:],
                )
    props = torch.cuda.get_device_properties(aq.device)
    params = (limit,) if activation is op.ActivationType.Swiglu else (beta, linear_beta)
    executable = op._COMPILED_SPLIT_K[(activation, params, props.major, props.minor)]
    tensors = (
        aq,
        bq,
        asf,
        bsf.view(torch.int32),
        q,
        sf.view(torch.uint8),
        offsets,
        mapping,
        partial,
    )
    args = tuple(
        op.from_dlpack(
            value, assumed_align=4 if index in (6, 7) else 16
        ).mark_layout_dynamic(
            leading_dim=value.ndim - 1 if value.numel() == 1 else None
        )
        for index, value in enumerate(tensors)
    ) + (op._stream(),)
    executable(*args)
    torch.testing.assert_close(partial, expected, rtol=1e-5, atol=1e-5)


def _check_split_k_result(q, sf, reference, offsets):
    valid = reference.shape[0]
    q_ref, sf_ref = mxfp8_act_quantize(reference.to(torch.bfloat16))
    assert calc_diff(q[:valid].float(), q_ref.float()) < 1e-2
    exponents = (sf_ref.view(torch.int32) >> 23).to(torch.uint8)
    packed = F.pad(exponents, (0, -exponents.shape[-1] % 4), value=0).view(torch.int32)
    expected = torch.zeros_like(sf)
    valid_rows = torch.zeros(sf.shape[1], dtype=torch.bool, device=sf.device)
    values = offsets.tolist()
    for expert, (begin, end) in enumerate(zip(values[:-1], values[1:], strict=True)):
        start = compute_padded_offset(begin, expert)
        expected[:, start : start + end - begin] = packed[begin:end].t()
        valid_rows[start : start + end - begin] = True
    decoded = (sf.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    truth = (expected.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    assert calc_diff(decoded, truth) < 5e-3
    assert torch.equal(sf[:, ~valid_rows], expected[:, ~valid_rows])


@pytest.mark.parametrize(
    "k,n,counts,kind,limit,beta,linear_beta,tokens",
    [
        (128, 128, (0, 1, 0), "situ", None, 4.0, 25.0, 19),
        (384, 256, (2, 0, 3), "situ", None, 4.0, 25.0, 19),
        (640, 384, (0, 9, 1, 0), "situ", None, 4.0, 25.0, 19),
        (3584, 384, (1, 0, 7, 17), "situ", None, 4.0, 25.0, 19),
        (384, 512, (0, 1, 0, 5), "situ", None, 2.5, 9.0, 19),
        (640, 256, (3, 0, 2), "swiglu", None, 4.0, 25.0, 19),
        (128, 128, (1, 0, 0, 1), "swiglu", 0.25, 4.0, 25.0, 19),
        (128, 128, (0, 0, 0), "situ", None, 4.0, 25.0, 1),
        (3584, 384, (1, 1, 1, 1), "situ", None, 4.0, 25.0, 1),
    ],
)
def test_fc1_split_k_matches_reference(
    k, n, counts, kind, limit, beta, linear_beta, tokens
):
    skip_if_not_sm120()
    from flashinfer.tllm_enums import ActivationType

    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    aq, asf, bq, bsf, offsets, mapping, reference = _split_k_data(
        k, n, counts, kind, limit, beta, linear_beta, tokens=tokens
    )
    q = torch.full((mapping.numel(), n), -1, dtype=aq.dtype, device=aq.device)
    sf = torch.zeros(
        op.out_sf_shape(mapping.numel(), n, len(counts)),
        dtype=torch.int32,
        device=aq.device,
    )
    activation = ActivationType.Situ if kind == "situ" else ActivationType.Swiglu
    actual_q, actual_sf = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        aq,
        asf,
        bq,
        bsf,
        offsets,
        token_map=mapping,
        split_k=2,
        activation=activation,
        swiglu_limit=limit,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
        out_q=q,
        out_sf=sf,
    )
    assert actual_q.data_ptr() == q.data_ptr() and actual_sf.data_ptr() == sf.data_ptr()
    _check_split_k_result(q, sf, reference, offsets)
    _check_split_k_partials(
        op,
        (aq, asf, bq, bsf, offsets, mapping),
        q,
        sf,
        activation,
        limit,
        beta,
        linear_beta,
    )
    _check_split_k_result(q, sf, reference, offsets)
    assert torch.equal(
        q[sum(counts) :].float(), torch.full_like(q[sum(counts) :].float(), -1)
    )


@pytest.mark.parametrize(
    "tile,rows,n,k,sms,expected",
    [
        ((8, 128, 128), 16, 384, 3584, 188, 2),
        ((8, 128, 128), 16, 384, 512, 188, 2),
        ((8, 128, 128), 16, 384, 640, 188, 2),
        ((8, 128, 128), 16, 384, 384, 188, 1),
        ((8, 128, 128), 16, 384, 128, 188, 1),
        ((8, 128, 128), 64, 384, 512, 188, 1),
        ((8, 128, 128), 16, 384, 512, 48, 1),
        ((8, 128, 128), 7, 384, 512, 188, 1),
        ((8, 128, 128), 8, 384, 512, 188, 2),
        ((8, 128, 128), 0, 384, 512, 188, 1),
        ((32, 128, 128), 16, 384, 512, 188, 1),
        ((8, 128, 128), 16, 64, 512, 188, 1),
        ((8, 128, 128), 16, 384, 513, 188, 1),
        ((8, 128, 128), 16, 384, 512, 0, 1),
    ],
)
def test_fc1_split_k_parallelism_guards(tile, rows, n, k, sms, expected):
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    assert (
        op.select_split_k(tile=tile, total_rows=rows, n=n, k=k, num_sms=sms) == expected
    )


@pytest.mark.parametrize("split_k", [None, 1, 2])
@pytest.mark.parametrize("kind,k", [("swiglu", 512), ("situ", 3584)])
def test_fc1_auto_split_k_matches_forced(split_k, kind, k, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.tllm_enums import ActivationType

    monkeypatch.setenv("MOE_AUTOTUNE", "0")
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    aq, asf, bq, bsf, offsets, mapping, reference = _split_k_data(
        k, 384, (0, 8, 0), kind
    )
    entered, original = [], op._run_split_k

    def tracked(*args, **kwargs):
        entered.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(op, "_run_split_k", tracked)
    q, sf = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        aq,
        asf,
        bq,
        bsf,
        offsets,
        token_map=mapping,
        split_k=split_k,
        activation=ActivationType.Situ if kind == "situ" else ActivationType.Swiglu,
    )
    sms = torch.cuda.get_device_properties(aq.device).multi_processor_count
    selected = op.select_split_k(
        tile=(8, 128, 128), total_rows=mapping.numel(), n=384, k=k, num_sms=sms
    )
    assert len(entered) == int((selected if split_k is None else split_k) == 2)
    _check_split_k_result(q, sf, reference, offsets)


@pytest.mark.parametrize(
    "options,tuned,invalid",
    [
        ({"tile": (8, 128, 128), "epi": "WG_QUANT_SWAP"}, False, False),
        ({"tune": True}, True, False),
        ({"epi": "WG_QUANT_R2S"}, False, False),
        ({"tile": (8, 128, 128)}, False, True),
    ],
)
def test_fc1_split_k_preserves_explicit_choices(options, tuned, invalid, monkeypatch):
    skip_if_not_sm120()
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    aq, asf, bq, bsf, offsets, mapping, reference = _split_k_data(512, 384, (0, 8, 0))
    options = {
        key: getattr(op.EpiMethod, value) if key == "epi" else value
        for key, value in options.items()
    }

    def unexpected(*args, **kwargs):
        pytest.fail("an explicit choice entered automatic SplitK2")

    monkeypatch.setattr(op, "_run_split_k", unexpected)
    monkeypatch.setattr(op, "MOE_AUTOTUNE_ENABLED", lambda: tuned)
    monkeypatch.setattr(
        op.AutoTuner.get(),
        "choose_one",
        lambda *args, **kwargs: (0, (8, 128, op.EpiMethod.WG_QUANT_SWAP)),
    )
    if invalid:
        with pytest.raises(TypeError, match="unsupported"):
            op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
                aq, asf, bq, bsf, offsets, token_map=mapping, **options
            )
    else:
        q, sf = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
            aq, asf, bq, bsf, offsets, token_map=mapping, **options
        )
        _check_split_k_result(q, sf, reference, offsets)


@pytest.mark.parametrize("split_k", [None, 1, 2])
def test_fc1_split_k_runner_fallback(split_k, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.tllm_enums import ActivationType

    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    aq, asf, bq, bsf, offsets, mapping, _ = _split_k_data(512, 384, (0, 8, 0))
    inputs, calls = [aq, asf, bq, bsf, offsets, mapping], []
    runner = op._Fc1ActQ1Runner(aq.dtype, ActivationType.Swiglu, split_k=split_k)

    def dispatched(*args, **kwargs):
        calls.append(kwargs)
        return kwargs["out_q"], kwargs["out_sf"]

    monkeypatch.setattr(op, "cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4", dispatched)
    prepared = runner.forward(inputs, tactic=-1, do_preparation=True)
    actual = runner.forward(inputs, tactic=-1)
    assert actual is prepared and len(calls) == 1
    assert calls[0]["split_k"] == split_k and calls[0]["tune"] is False
    assert calls[0]["out_q"] is prepared[0] and calls[0]["out_sf"] is prepared[1]
    assert runner.get_cache_key_extras(inputs)[-1] == split_k


def test_fc1_auto_split_k_preserves_legacy_offsets(monkeypatch):
    skip_if_not_sm120()
    monkeypatch.setenv("MOE_AUTOTUNE", "0")
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    aq, asf, bq, bsf, offsets, mapping, reference = _split_k_data(512, 384, (0, 8, 0))

    def unexpected(*args, **kwargs):
        pytest.fail("legacy offset dtype entered SplitK2")

    monkeypatch.setattr(op, "_run_split_k", unexpected)
    q, sf = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        aq, asf, bq, bsf, offsets.to(torch.int64), token_map=mapping
    )
    _check_split_k_result(q, sf, reference, offsets)


@pytest.mark.parametrize("split_k", [None, 2])
def test_fc1_split_k_two_stream_graphs(split_k, monkeypatch):
    skip_if_not_sm120()
    monkeypatch.setenv("MOE_AUTOTUNE", "0")
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    records = []
    for stream, seed in zip(streams, (19, 29), strict=True):
        aq, asf, bq, bsf, offsets, mapping, reference = _split_k_data(
            512, 384, (0, 8, 0), seed=seed
        )

        def run():
            return op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
                aq, asf, bq, bsf, offsets, token_map=mapping, split_k=split_k
            )

        anchor = tuple(value.clone() for value in run())
        _check_split_k_result(*anchor, reference, offsets)
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        records.append((graph, actual, anchor, (aq, asf, bq, bsf, offsets, mapping)))
    assert not torch.equal(
        records[0][2][0][:8].view(torch.uint8), records[1][2][0][:8].view(torch.uint8)
    )
    for _ in range(8):
        for stream, (graph, _, _, _) in zip(streams, records, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for _, actual, anchor, _ in records:
        assert torch.equal(
            actual[0][:8].view(torch.uint8), anchor[0][:8].view(torch.uint8)
        )
        assert torch.equal(actual[1], anchor[1])


def test_fc1_split_k_reference_rejects_corruption():
    skip_if_not_sm120()
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    aq, asf, bq, bsf, offsets, mapping, reference = _split_k_data(512, 384, (0, 8, 0))
    q, sf = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        aq, asf, bq, bsf, offsets, token_map=mapping, split_k=2
    )
    _check_split_k_result(q, sf, reference, offsets)
    with pytest.raises(AssertionError):
        _check_split_k_result(torch.zeros_like(q), sf, reference, offsets)


def test_fc1_split_k_zero_capacity():
    skip_if_not_sm120()
    op = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
    )
    aq, asf, bq, bsf, offsets, mapping, _ = _split_k_data(128, 128, (0, 0, 0))
    q, sf = op.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        aq, asf, bq, bsf, offsets, token_map=mapping[:0], split_k=2
    )
    assert q.shape == (0, 128) and bool((sf == 0).all())


@pytest.mark.parametrize("split_k", [0, 3, True])
def test_fc1_split_k_rejects_invalid_mode(split_k):
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4

    with pytest.raises(ValueError, match="split_k"):
        cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
            None, None, None, None, None, split_k=split_k
        )
