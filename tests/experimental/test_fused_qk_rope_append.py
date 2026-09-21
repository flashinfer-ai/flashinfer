"""Correctness tests for experimental attention preprocessing ported from Tencent hpc-ops."""

import pytest
import torch

import flashinfer


def _requires_hpc_rope():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("hpc fused RoPE requires SM90+")


def _case(q_heads=8, kv_heads=1):
    torch.manual_seed(3)
    head_dim = page_size = 128
    page_size = 4
    q_indptr = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([5, 4], device="cuda", dtype=torch.int32)
    page_indices = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    rows = int(q_indptr[-1])
    width = (q_heads + 2 * kv_heads) * head_dim
    qkv = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([2, 3, 4, 2, 3], device="cuda", dtype=torch.long)
    angles = torch.randn(16, head_dim // 2, device="cuda")
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1).float()
    q_weight = torch.rand(head_dim, device="cuda") + 0.5
    k_weight = torch.rand(head_dim, device="cuda") + 0.5
    key_cache = torch.randn(
        4,
        page_size,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    return {
        "qkv": qkv,
        "cos_sin": cos_sin,
        "seq_lens": seq_lens,
        "q_indptr": q_indptr,
        "page_indices": page_indices,
        "positions": positions,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
    }


def _rmsnorm(x, weight):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight


def _rope(x, cos_sin, positions):
    x = x.float()
    half = x.shape[-1] // 2
    cos = cos_sin[positions, :half]
    sin = cos_sin[positions, half:]
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    left, right = x[..., :half], x[..., half:]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)


def _reference(case, policy):
    qh, kvh, d = case["q_heads"], case["kv_heads"], case["head_dim"]
    qkv = case["qkv"]
    q = qkv[:, : qh * d].view(-1, qh, d)
    k = qkv[:, qh * d : (qh + kvh) * d].view(-1, kvh, d)
    v = qkv[:, (qh + kvh) * d :].view(-1, kvh, d)
    if policy == 2:
        q, k = _rmsnorm(q, case["q_weight"]), _rmsnorm(k, case["k_weight"])
    q = _rope(q, case["cos_sin"], case["positions"])
    k = _rope(k, case["cos_sin"], case["positions"])
    if policy == 1:
        q, k = _rmsnorm(q, case["q_weight"]), _rmsnorm(k, case["k_weight"])
    return q, k, v


def _assert_cache(case, key_cache, value_cache, key_ref, value_ref, *, fp8=False):
    mapping = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3)]
    for row, (batch, position) in enumerate(mapping):
        page = int(case["page_indices"][batch, position // case["page_size"]])
        offset = position % case["page_size"]
        if fp8:
            torch.testing.assert_close(
                key_cache[page, offset].float(), key_ref[row], rtol=0.15, atol=0.03
            )
            torch.testing.assert_close(
                value_cache[page, offset].float(),
                value_ref[row],
                rtol=0.15,
                atol=0.03,
            )
        else:
            torch.testing.assert_close(
                key_cache[page, offset].float(), key_ref[row], rtol=0.02, atol=0.02
            )
            torch.testing.assert_close(
                value_cache[page, offset].float(),
                value_ref[row].float(),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("policy", [0, 1, 2])
def test_fused_qk_norm_rope_append_bf16(policy):
    _requires_hpc_rope()
    case = _case()
    out = flashinfer.rope.fused_qk_rmsnorm_rope_append_paged_kv_cache(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        (case["key_cache"], case["value_cache"]),
        True,
        case["q_weight"] if policy else None,
        case["k_weight"] if policy else None,
        policy,
    )
    q_ref, k_ref, v_ref = _reference(case, policy)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), q_ref, rtol=0.02, atol=0.02)
    _assert_cache(case, case["key_cache"], case["value_cache"], k_ref, v_ref)
    # The kernel clears unused rows in the final cache page for deterministic
    # downstream loads.
    assert (case["key_cache"][1, 1:] == 0).all()
    assert (case["value_cache"][1, 1:] == 0).all()


@pytest.mark.parametrize("quant_policy", [1, 2])
def test_fused_qk_norm_rope_append_fp8(quant_policy):
    _requires_hpc_rope()
    case = _case()
    key_cache = torch.empty_like(case["key_cache"], dtype=torch.float8_e4m3fn)
    value_cache = torch.empty_like(key_cache)
    k_scale = torch.tensor([0.02], device="cuda")
    v_scale = torch.tensor([0.02], device="cuda")
    q_static_scale = torch.tensor([0.02], device="cuda")
    out, q_scale, flags = (
        flashinfer.rope.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            (key_cache, value_cache),
            True,
            k_scale,
            v_scale,
            quant_policy,
            max_seqlen=3,
            q_scale_inv=(1 / q_static_scale if quant_policy == 2 else None),
            q_norm_weight=case["q_weight"],
            k_norm_weight=case["k_weight"],
            qk_norm_policy=2,
        )
    )
    q_ref, k_ref, v_ref = _reference(case, 2)
    if quant_policy == 1:
        token_in_request = [0, 1, 2, 0, 1]
        for row, (batch, token) in enumerate(
            zip([0, 0, 0, 1, 1], token_in_request, strict=True)
        ):
            scale = q_scale[batch, :, token]
            torch.testing.assert_close(
                scale, q_ref[row].abs().amax(dim=-1) / 448, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                out[row].float() * scale[:, None], q_ref[row], rtol=0.15, atol=0.03
            )
    else:
        assert q_scale.numel() == 0
        torch.testing.assert_close(
            out.float() * q_static_scale, q_ref, rtol=0.15, atol=0.03
        )
    assert (flags == 0).all()
    _assert_cache(
        case,
        key_cache,
        value_cache,
        k_ref / k_scale,
        v_ref.float() / v_scale,
        fp8=True,
    )


def test_fused_qk_norm_rope_64q_8kv_dispatch():
    _requires_hpc_rope()
    case = _case(q_heads=64, kv_heads=8)
    out_k = torch.empty(5, 8, 128, device="cuda", dtype=torch.bfloat16)
    out_v = torch.empty_like(out_k)
    out = flashinfer.rope.fused_qk_rmsnorm_rope_append_paged_kv_cache(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        (case["key_cache"], case["value_cache"]),
        True,
        case["q_weight"],
        case["k_weight"],
        2,
        out_k=out_k,
        out_v=out_v,
    )
    q_ref, k_ref, v_ref = _reference(case, 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), q_ref, rtol=0.02, atol=0.02)
    torch.testing.assert_close(out_k.float(), k_ref, rtol=0.02, atol=0.02)
    torch.testing.assert_close(out_v.float(), v_ref.float(), rtol=0, atol=0)


@pytest.mark.parametrize("upper_max", [0.0, -1.0, 448.01, float("inf"), float("nan")])
def test_fused_qk_norm_rope_fp8_rejects_invalid_upper_max(upper_max):
    _requires_hpc_rope()
    case = _case()
    key_cache = torch.empty_like(case["key_cache"], dtype=torch.float8_e4m3fn)
    value_cache = torch.empty_like(key_cache)
    scale = torch.tensor([0.02], device="cuda")
    with pytest.raises(ValueError, match="upper_max"):
        flashinfer.rope.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            (key_cache, value_cache),
            True,
            scale,
            scale,
            1,
            max_seqlen=3,
            upper_max=upper_max,
        )


def test_fused_qk_norm_rope_rejects_misshapen_output_buffers():
    _requires_hpc_rope()
    case = _case()
    bad_out_q = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="out_q"):
        flashinfer.rope.fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            (case["key_cache"], case["value_cache"]),
            True,
            out_q=bad_out_q,
        )

    key_cache = torch.empty_like(case["key_cache"], dtype=torch.float8_e4m3fn)
    value_cache = torch.empty_like(key_cache)
    scale = torch.tensor([0.02], device="cuda")
    bad_q_scale = torch.empty(1, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="q_scale"):
        flashinfer.rope.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            (key_cache, value_cache),
            True,
            scale,
            scale,
            1,
            max_seqlen=3,
            q_scale=bad_q_scale,
        )


def test_fused_qk_norm_rope_fp8_zeroes_flags_for_empty_q_request():
    _requires_hpc_rope()
    case = _case()
    qkv = case["qkv"][:1]
    q_indptr = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    key_cache = torch.empty_like(case["key_cache"], dtype=torch.float8_e4m3fn)
    value_cache = torch.empty_like(key_cache)
    scale = torch.tensor([0.02], device="cuda")
    flags = torch.full((2, 1), 7, device="cuda", dtype=torch.int32)

    _, _, actual_flags = (
        flashinfer.rope.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            qkv,
            case["cos_sin"],
            seq_lens,
            q_indptr,
            case["page_indices"],
            (key_cache, value_cache),
            True,
            scale,
            scale,
            1,
            max_seqlen=1,
            split_k_flag=flags,
        )
    )
    torch.testing.assert_close(actual_flags, torch.zeros_like(actual_flags))


def test_fused_qk_norm_rope_fp8_flags_per_request_max_seqlen_overflow():
    """A long individual request must not silently omit its Q scale writes."""
    _requires_hpc_rope()
    case = _case()
    # Total rows (5) still fit num_requests * max_seqlen (6), but request 0
    # individually exceeds max_seqlen. This was the previously missed case.
    q_indptr = torch.tensor([0, 4, 5], device="cuda", dtype=torch.int32)
    key_cache = torch.zeros_like(case["key_cache"], dtype=torch.float8_e4m3fn)
    value_cache = torch.zeros_like(key_cache)
    scale = torch.tensor([0.02], device="cuda")
    flags = torch.zeros((2, 1), device="cuda", dtype=torch.int32)
    out_q = torch.zeros(
        case["qkv"].shape[0], 8, 128, device="cuda", dtype=torch.float8_e4m3fn
    )

    actual_out, _, actual_flags = (
        flashinfer.rope.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            q_indptr,
            case["page_indices"],
            (key_cache, value_cache),
            True,
            scale,
            scale,
            1,
            max_seqlen=3,
            split_k_flag=flags,
            out_q=out_q,
        )
    )
    torch.cuda.synchronize()
    assert (actual_flags == -1).all()
    assert (actual_out.float() == 0).all()
    assert (key_cache.float() == 0).all()
    assert (value_cache.float() == 0).all()
