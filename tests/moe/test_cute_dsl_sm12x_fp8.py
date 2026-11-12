"""SM12x FP8 internal MoE ops against pure-Torch references."""

import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from flashinfer.utils import is_sm120a_supported
from tests.moe.test_sm12x_fp8_q0_route import _quantize_reference

pytestmark = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="cute_dsl not available"
)


def skip_if_not_sm120():
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum().item()
    return 0.0 if denom == 0 else 1.0 - 2.0 * (x * y).sum().item() / denom


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * 3) // 4 * 4


def _repack_fp8_scales(sf, token_offset):
    token_num = sf.shape[0]
    num_experts = token_offset.numel() - 1
    scale_k = sf.size(1)
    m_padded = compute_padded_offset(token_num, num_experts)
    padded = torch.zeros((scale_k, m_padded), dtype=torch.float32, device=sf.device)
    for i in range(num_experts):
        start, end = int(token_offset[i].item()), int(token_offset[i + 1].item())
        if start == end:
            continue
        padded_start = compute_padded_offset(start, i)
        padded[:, padded_start : padded_start + end - start] = sf[start:end].t()
    return padded


def per_token_cast_to_fp8_for_moe_gemm(x, token_offset):
    """Same MN-major padded-scale repacking as tests/grouped_mm/test_cute_sm120_fp8.py."""
    x_fp8, sf = per_token_cast_to_fp8(x)
    return x_fp8, _repack_fp8_scales(sf, token_offset)


def make_inputs(m_per_expert_list, n, k):
    torch.random.manual_seed(0)
    num_experts = len(m_per_expert_list)
    offsets = [0]
    for m_pe in m_per_expert_list:
        offsets.append(offsets[-1] + m_pe)
    total_rows = offsets[-1]
    a = torch.randn((total_rows, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    ref = torch.zeros((total_rows, n), dtype=torch.bfloat16, device="cuda")
    for i in range(num_experts):
        start, end = offsets[i], offsets[i + 1]
        if start < end:
            ref[start:end] = a[start:end] @ b[i].t()

    a_fp8, a_scale = per_token_cast_to_fp8_for_moe_gemm(a, m_indptr)
    b_fp8_list, b_sf_list = [], []
    for i in range(num_experts):
        b_i_fp8, b_i_sf = per_block_cast_to_fp8(b[i])
        b_fp8_list.append(b_i_fp8)
        b_sf_list.append(b_i_sf)
    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_sf_list, dim=0).transpose(-1, -2).contiguous()
    return a_fp8, b_fp8, a_scale, b_scale, m_indptr, ref


def test_cute_dsl_sm12x_moe_gemm_fp8_matches_reference():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_moe_gemm_fp8

    a, b, a_scale, b_scale, m_indptr, ref = make_inputs([64, 64, 64, 64], 512, 512)
    out = cute_dsl_sm12x_moe_gemm_fp8(a, a_scale, b, b_scale, m_indptr)
    diff = calc_diff(out.float(), ref.float())
    assert diff < 1e-3, f"calc_diff={diff:.6e}"


def make_gated_inputs(m_per_expert_list, n, k, swiglu_limit=None):
    torch.random.manual_seed(0)
    offsets = [0]
    for count in m_per_expert_list:
        offsets.append(offsets[-1] + count)
    a = torch.randn(offsets[-1], k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        len(m_per_expert_list), 2 * n, k, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    ref = torch.zeros(offsets[-1], n, dtype=torch.bfloat16, device="cuda")
    for expert in range(len(m_per_expert_list)):
        start, end = offsets[expert], offsets[expert + 1]
        if start < end:
            gemm1 = a[start:end] @ b[expert].t()
            linear, gate = gemm1[:, :n], gemm1[:, n:]
            if swiglu_limit is not None:
                gate = gate.clamp(max=swiglu_limit)
                linear = linear.clamp(-swiglu_limit, swiglu_limit)
            ref[start:end] = F.silu(gate) * linear
    a_fp8, a_scale = per_token_cast_to_fp8_for_moe_gemm(a, m_indptr)
    b_fp8, b_scale = zip(*(per_block_cast_to_fp8(x) for x in b), strict=True)
    return (
        a_fp8,
        torch.stack(b_fp8),
        a_scale,
        torch.stack(b_scale).transpose(-1, -2).contiguous(),
        m_indptr,
        ref,
    )


def test_cute_dsl_sm12x_fc1_act_fp8_matches_reference():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_fp8

    a, b, a_sf, b_sf, offsets, ref = make_gated_inputs([64] * 4, 512, 512)
    out = cute_dsl_sm12x_fc1_act_fp8(a, a_sf, b, b_sf, offsets)
    assert calc_diff(out.float(), ref.float()) < 2e-3


def test_cute_dsl_sm12x_fc1_act_q1_fp8_smoke():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_fp8

    a, b, a_sf, b_sf, offsets, _ = make_gated_inputs([64] * 4, 512, 512)
    q, sf = cute_dsl_sm12x_fc1_act_q1_fp8(a, a_sf, b, b_sf, offsets)
    assert (
        q.dtype == torch.float8_e4m3fn
        and not torch.isnan(q.float()).any()
        and sf.numel() > 0
    )


def test_cute_dsl_sm12x_fc1_act_q1_fp8_swiglu_clamp_and_output_reuse():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_fp8

    limit = 0.25
    a, b, a_sf, b_sf, offsets, ref = make_gated_inputs(
        [64] * 4, 512, 512, swiglu_limit=limit
    )
    out_q = torch.empty_like(a)
    out_sf = torch.zeros(
        (4, compute_padded_offset(a.shape[0], offsets.numel() - 1)),
        dtype=torch.float32,
        device="cuda",
    )
    q, sf = cute_dsl_sm12x_fc1_act_q1_fp8(
        a,
        a_sf,
        b,
        b_sf,
        offsets,
        swiglu_limit=limit,
        out_q=out_q,
        out_sf=out_sf,
    )
    assert q.data_ptr() == out_q.data_ptr()
    assert sf.data_ptr() == out_sf.data_ptr()
    dequant = torch.empty_like(ref, dtype=torch.float32)
    for expert in range(offsets.numel() - 1):
        start, end = map(int, offsets[expert : expert + 2].tolist())
        padded_start = compute_padded_offset(start, expert)
        scales = sf[:, padded_start : padded_start + end - start].t()
        dequant[start:end] = (
            q[start:end].float().reshape(end - start, -1, 128) * scales[:, :, None]
        ).reshape(end - start, -1)
    assert calc_diff(dequant, ref.float()) < 2e-2


def test_cute_dsl_sm12x_fc2_finalize_fp8_smoke():
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc2_finalize_fp8

    a, b, a_sf, b_sf, offsets, _ = make_inputs([64] * 4, 512, 512)
    rows = a.shape[0]
    tok = torch.randint(0, rows, (rows,), device="cuda", dtype=torch.int32)
    out = cute_dsl_sm12x_fc2_finalize_fp8(
        a, a_sf, b, b_sf, offsets, tok, torch.rand(rows, device="cuda"), rows
    )
    assert out.shape == (rows, 512) and not torch.isnan(out.float()).any()


def _make_gather_inputs(counts=(0, 1, 31, 33), k=512, limit=None, *, tokens=19, n=384):
    torch.manual_seed(19)
    x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(
        len(counts), 2 * n, k, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    rows = torch.arange(sum(counts), device="cuda")
    mapping = ((rows * 7 + 5) % tokens).to(torch.int32)
    a_q, sf_rows = per_token_cast_to_fp8(x)
    a_sf = sf_rows.t().contiguous()
    routed_q, routed_sf = per_token_cast_to_fp8_for_moe_gemm(x[mapping.long()], offsets)
    b_q, b_sf = zip(*(per_block_cast_to_fp8(w) for w in weights), strict=True)
    x_dequant = (a_q.float().reshape(tokens, -1, 128) * sf_rows[:, :, None]).reshape(
        tokens, k
    )
    ref = torch.empty(sum(counts), n, dtype=torch.float32, device="cuda")
    for expert in range(len(counts)):
        begin, end = map(int, offsets[expert : expert + 2].tolist())
        if begin < end:
            weight = (
                b_q[expert].float().reshape(2 * n // 128, 128, k // 128, 128)
                * b_sf[expert][:, None, :, None]
            ).reshape(2 * n, k)
            accum = x_dequant[mapping[begin:end].long()] @ weight.t()
            linear, gate = accum[:, :n], accum[:, n:]
            if limit is not None:
                gate = gate.clamp(max=limit)
                linear = linear.clamp(-limit, limit)
            ref[begin:end] = F.silu(gate) * linear
    ref_q, ref_sf_rows = _quantize_reference(ref)
    ref_sf = _repack_fp8_scales(ref_sf_rows, offsets)
    b_q = torch.stack(b_q)
    b_sf = torch.stack(b_sf).transpose(-1, -2).contiguous()
    return dict(
        gather=(a_q, a_sf, b_q, b_sf, offsets),
        routed=(routed_q, routed_sf, b_q, b_sf, offsets),
        mapping=mapping,
        reference=(ref_q, ref_sf),
    )


def _assert_quantized_output(output, reference):
    q, sf = output
    ref_q, ref_sf = reference
    assert q.shape == ref_q.shape and sf.shape == ref_sf.shape
    assert torch.isfinite(q.float()).all() and torch.isfinite(sf).all()
    assert calc_diff(q.float(), ref_q.float()) < 1e-2
    assert calc_diff(sf, ref_sf) < 5e-3


@pytest.mark.parametrize("bm", (8, 32))
@pytest.mark.parametrize("byte_offset", (1, 16))
def test_fp8_fc1_gather_activation_alignment(bm, byte_offset, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc1_act_q1 as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    inputs = _make_gather_inputs(counts=(0, 2, 0), tokens=2, k=512, n=128)
    a_q, a_sf, b_q, b_sf, offsets = inputs["gather"]
    mapping = inputs["mapping"]
    storage = torch.empty(
        a_q.numel() + byte_offset, dtype=torch.uint8, device=a_q.device
    )
    shifted = storage[byte_offset:].view(a_q.dtype).view_as(a_q)
    shifted.view(torch.uint8).copy_(a_q.view(torch.uint8))
    assert shifted.is_contiguous() and shifted.storage_offset() == byte_offset
    assert shifted.data_ptr() % 16 == byte_offset % 16
    epi = entry.EpiMethod.WG_QUANT_SWAP if bm == 8 else entry.EpiMethod.WG_S2R_QUANT
    kwargs = dict(token_map=mapping, tile=(bm, 128, 128), epi=epi)
    if byte_offset % 16:

        def unexpected_compile(*args, **kwargs):
            pytest.fail("misaligned gather activation reached kernel compilation")

        monkeypatch.setattr(entry, "compiled_kernel", unexpected_compile)
        with pytest.raises(
            ValueError, match="gather activation must be 16-byte aligned"
        ):
            entry.cute_dsl_sm12x_fc1_act_q1_fp8(
                shifted, a_sf, b_q, b_sf, offsets, **kwargs
            )
        return
    q, sf = entry.cute_dsl_sm12x_fc1_act_q1_fp8(
        shifted, a_sf, b_q, b_sf, offsets, **kwargs
    )
    q_base, sf_base = entry.cute_dsl_sm12x_fc1_act_q1_fp8(
        a_q, a_sf, b_q, b_sf, offsets, **kwargs
    )
    assert torch.equal(q.view(torch.uint8), q_base.view(torch.uint8))
    assert torch.equal(sf, sf_base)
    _assert_quantized_output((q, sf), inputs["reference"])


@pytest.mark.parametrize("bm", (8, 32, 64))
@pytest.mark.parametrize("limit", (None, 0.25))
def test_fp8_fc1_token_gather_matches_reference(bm, limit, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc1_act_q1 as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    inputs = _make_gather_inputs(limit=limit)
    tactics = [t for t in entry.TACTICS if t[0] == bm]
    assert tactics
    for tactic in tactics:
        tile, epi = entry.split_tactic(tactic)
        ref_q, ref_sf = inputs["reference"]
        out_q, out_sf = torch.empty_like(ref_q), torch.zeros_like(ref_sf)
        result = entry.cute_dsl_sm12x_fc1_act_q1_fp8(
            *inputs["gather"],
            token_map=inputs["mapping"],
            tile=tile,
            epi=epi,
            swiglu_limit=limit,
            out_q=out_q,
            out_sf=out_sf,
        )
        assert result[0].data_ptr() == out_q.data_ptr()
        assert result[1].data_ptr() == out_sf.data_ptr()
        _assert_quantized_output(result, inputs["reference"])
        _assert_quantized_output(
            entry.cute_dsl_sm12x_fc1_act_q1_fp8(
                *inputs["routed"],
                tile=tile,
                epi=epi,
                swiglu_limit=limit,
            ),
            inputs["reference"],
        )


@pytest.mark.parametrize("bm", (8, 32, 64))
def test_fp8_fc1_gather_singleton_scale(bm, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc1_act_q1 as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    inputs = _make_gather_inputs(counts=(3,), k=128, tokens=1, n=128)
    assert inputs["gather"][1].shape == (1, 1)
    tile, epi = entry.split_tactic(next(t for t in entry.TACTICS if t[0] == bm))
    _assert_quantized_output(
        entry.cute_dsl_sm12x_fc1_act_q1_fp8(
            *inputs["gather"], token_map=inputs["mapping"], tile=tile, epi=epi
        ),
        inputs["reference"],
    )


@pytest.mark.parametrize("bm", (8, 32, 64))
def test_fp8_fc1_gather_multiple_sf_cycles_and_work_items(bm, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc1_act_q1 as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    inputs = _make_gather_inputs(counts=(bm * sms + 3, 0, 3), k=3584, limit=0.25)
    tactics = [t for t in entry.TACTICS if t[0] == bm]
    assert tactics
    for tactic in tactics:
        tile, epi = entry.split_tactic(tactic)
        _assert_quantized_output(
            entry.cute_dsl_sm12x_fc1_act_q1_fp8(
                *inputs["gather"],
                token_map=inputs["mapping"],
                tile=tile,
                epi=epi,
                swiglu_limit=0.25,
            ),
            inputs["reference"],
        )


def test_fp8_fc1_gather_cache_mode_and_graph_replay(monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc1_act_q1 as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    inputs = _make_gather_inputs(counts=(0, 3, 7, 9))
    tile, epi = entry.split_tactic(next(t for t in entry.TACTICS if t[0] == 64))
    for args, mapping in (
        (inputs["routed"], None),
        (inputs["gather"], inputs["mapping"]),
    ):
        _assert_quantized_output(
            entry.cute_dsl_sm12x_fc1_act_q1_fp8(
                *args,
                token_map=mapping,
                tile=tile,
                epi=epi,
            ),
            inputs["reference"],
        )
    assert {
        key[-1] for key in entry._COMPILED if key[0] == (tile[0], tile[1], epi)
    } == {False, True}
    ref_q, ref_sf = inputs["reference"]
    out_q, out_sf = torch.empty_like(ref_q), torch.zeros_like(ref_sf)

    def run():
        entry.cute_dsl_sm12x_fc1_act_q1_fp8(
            *inputs["gather"],
            token_map=inputs["mapping"],
            tile=tile,
            epi=epi,
            out_q=out_q,
            out_sf=out_sf,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        out_q.view(torch.uint8).fill_(255)
        graph.replay()
        torch.cuda.synchronize()
        _assert_quantized_output((out_q, out_sf), inputs["reference"])


@pytest.mark.parametrize(
    "bad",
    (
        "map_dtype",
        "map_rank",
        "map_stride",
        "map_device",
        "sf_dtype",
        "sf_rank",
        "sf_width",
        "sf_stride",
        "sf_device",
        "sf_k",
    ),
)
def test_fp8_fc1_gather_rejects_invalid_inputs(bad):
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_fp8

    inputs = _make_gather_inputs()
    a_q, a_sf, b_q, b_sf, offsets = inputs["gather"]
    mapping = inputs["mapping"]
    if bad == "map_dtype":
        mapping = mapping.long()
    elif bad == "map_rank":
        mapping = mapping.unsqueeze(0)
    elif bad == "map_stride":
        mapping = mapping.repeat_interleave(2)[::2]
    elif bad == "map_device":
        mapping = mapping.cpu()
    elif bad == "sf_dtype":
        a_sf = a_sf.bfloat16()
    elif bad == "sf_rank":
        a_sf = a_sf.unsqueeze(0)
    elif bad == "sf_width":
        a_sf = a_sf[:, :-1].contiguous()
    elif bad == "sf_stride":
        a_sf = a_sf.repeat_interleave(2, dim=1)[:, ::2]
    elif bad == "sf_device":
        a_sf = a_sf.cpu()
    else:
        a_sf = a_sf[:-1].contiguous()
    with pytest.raises(ValueError, match="token_map|gather inputs|a_scale K extent"):
        cute_dsl_sm12x_fc1_act_q1_fp8(a_q, a_sf, b_q, b_sf, offsets, token_map=mapping)


@pytest.mark.parametrize("bm", (32, 64, 128))
@pytest.mark.parametrize("k", (128, 384, 512, 640))
def test_fp8_fc2_copy_modes_ragged_rows(bm, k, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc2_finalize as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    a, b, a_sf, b_sf, offsets, ref = make_inputs([0, bm - 3, bm + 1], 256, k)
    rows = a.shape[0]
    tokens = torch.arange(rows, dtype=torch.int32, device="cuda")
    scales = torch.ones(rows, device="cuda")
    assert int(offsets[2].item()) % 4 != 0
    tactics = [t for t in entry.TACTICS if t[0] == bm]
    assert tactics
    for tactic in tactics:
        tile, epi = entry.split_tactic(tactic)
        out = entry.cute_dsl_sm12x_fc2_finalize_fp8(
            a, a_sf, b, b_sf, offsets, tokens, scales, rows + 3, tile=tile, epi=epi
        )
        assert torch.isfinite(out).all()
        assert calc_diff(out[:rows].float(), ref.float()) < 5e-3
        assert torch.count_nonzero(out[rows:]) == 0


@pytest.mark.parametrize("k", (384, 512, 640))
def test_fp8_fc2_strided_a(k, monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc2_finalize as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    a, b, a_sf, b_sf, offsets, ref = make_inputs([0, 61, 65], 256, k)
    storage = torch.full((a.shape[0], k + 128), 3.0, device="cuda").to(a.dtype)
    pitched = storage[:, :k]
    pitched.copy_(a)
    assert pitched.stride() == (k + 128, 1)
    tokens = torch.arange(a.shape[0], dtype=torch.int32, device="cuda")
    out = entry.cute_dsl_sm12x_fc2_finalize_fp8(
        pitched,
        a_sf,
        b,
        b_sf,
        offsets,
        tokens,
        torch.ones_like(tokens, dtype=torch.float32),
        a.shape[0] + 3,
        tile=(64, 128, 128),
    )
    assert torch.isfinite(out).all()
    assert calc_diff(out[: a.shape[0]].float(), ref.float()) < 5e-3
    assert torch.count_nonzero(out[a.shape[0] :]) == 0


def test_fp8_fc2_singleton_weight_scale(monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe import cute_dsl_sm12x_fc2_finalize_fp8

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    a, b, a_sf, b_sf, offsets, ref = make_inputs([3], 128, 128)
    assert b_sf.shape == (1, 1, 1)
    tokens = torch.arange(3, dtype=torch.int32, device="cuda")
    out = cute_dsl_sm12x_fc2_finalize_fp8(
        a,
        a_sf,
        b,
        b_sf,
        offsets,
        tokens,
        torch.ones(3, device="cuda"),
        3,
        tile=(64, 128, 128),
    )
    assert torch.isfinite(out).all()
    assert calc_diff(out.float(), ref.float()) < 5e-3


def test_fp8_fc2_cache_mode_switches_match_reference(monkeypatch):
    skip_if_not_sm120()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_fp8_fc2_finalize as entry,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    modes = []
    original_compile = entry.compiled_kernel

    def track_compile(sample_args, *, op, grid_x, sm_version):
        modes.append(op.use_cp_async)
        return original_compile(
            sample_args, op=op, grid_x=grid_x, sm_version=sm_version
        )

    monkeypatch.setattr(entry, "compiled_kernel", track_compile)
    for n, k in ((256, 384), (128, 128), (384, 512), (256, 640), (256, 384)):
        a, b, a_sf, b_sf, offsets, reference = make_inputs([0, 61, 65], n, k)
        rows = a.shape[0]
        tokens = torch.arange(rows, dtype=torch.int32, device="cuda")
        scales = torch.linspace(0.1, 1.0, rows, device="cuda")
        out = entry.cute_dsl_sm12x_fc2_finalize_fp8(
            a,
            a_sf,
            b,
            b_sf,
            offsets,
            tokens,
            scales,
            rows + 3,
            tile=(64, 128, 128),
        )
        assert torch.isfinite(out).all()
        assert calc_diff(out[:rows].float(), reference.float() * scales[:, None]) < 5e-3
        assert torch.count_nonzero(out[rows:]) == 0
    assert modes == [True, False, True, False, True]
