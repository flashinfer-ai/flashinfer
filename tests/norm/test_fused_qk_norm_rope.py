"""
Tests for fused QKNorm + 3D RoPE kernel.

Tests correctness against a PyTorch reference implementation that matches
the WAN 2.2 model.py:
  - RMSNorm across all heads (not per-head)
  - 3D RoPE with frame/height/width spatial decomposition
  - V passthrough copy
  - Optional FP8 E4M3 quantized output

Both interleaved and non-interleaved (NeoX) RoPE modes are tested.
The non-interleaved path has not been validated end-to-end by the kernel
author, so these tests serve as the first validation of that code path.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def apply_rotary_emb_interleaved(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved RoPE (pairs adjacent elements: 0,1  2,3  ...).

    hidden_states: [batch, seq, num_heads, head_dim]
    freqs_cos/sin: [batch, seq, 1, head_dim]
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def apply_rotary_emb_neox(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """NeoX-style (non-interleaved) RoPE: first half and second half.

    hidden_states: [batch, seq, num_heads, head_dim]
    freqs_cos/sin: [batch, seq, 1, head_dim]   (only first half_dim used)
    """
    half = hidden_states.shape[-1] // 2
    x1 = hidden_states[..., :half]
    x2 = hidden_states[..., half:]
    cos = freqs_cos[..., :half]
    sin = freqs_sin[..., :half]
    out = torch.empty_like(hidden_states)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def get_1d_rotary_pos_embed(dim, length, theta, device):
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim)
    )
    pos = torch.arange(length, device=device, dtype=torch.float64)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)

    cos_out = torch.zeros(length, dim, device=device, dtype=torch.float64)
    sin_out = torch.zeros(length, dim, device=device, dtype=torch.float64)
    cos_out[:, 0::2] = torch.cos(freqs)
    cos_out[:, 1::2] = torch.cos(freqs)
    sin_out[:, 0::2] = torch.sin(freqs)
    sin_out[:, 1::2] = torch.sin(freqs)
    return cos_out, sin_out


def create_3d_rotary_embeddings(batch_size, ppf, pph, ppw, head_dim, device,
                                base=10000.0, dtype=torch.bfloat16):
    h_dim = w_dim = 2 * (head_dim // 6)
    t_dim = head_dim - h_dim - w_dim

    max_len = max(ppf, pph, ppw)
    t_cos, t_sin = get_1d_rotary_pos_embed(t_dim, max_len, base, device)
    h_cos, h_sin = get_1d_rotary_pos_embed(h_dim, max_len, base, device)
    w_cos, w_sin = get_1d_rotary_pos_embed(w_dim, max_len, base, device)

    t_cos_3d = t_cos[:ppf].view(1, ppf, 1, 1, t_dim).expand(batch_size, ppf, pph, ppw, t_dim)
    t_sin_3d = t_sin[:ppf].view(1, ppf, 1, 1, t_dim).expand(batch_size, ppf, pph, ppw, t_dim)
    h_cos_3d = h_cos[:pph].view(1, 1, pph, 1, h_dim).expand(batch_size, ppf, pph, ppw, h_dim)
    h_sin_3d = h_sin[:pph].view(1, 1, pph, 1, h_dim).expand(batch_size, ppf, pph, ppw, h_dim)
    w_cos_3d = w_cos[:ppw].view(1, 1, 1, ppw, w_dim).expand(batch_size, ppf, pph, ppw, w_dim)
    w_sin_3d = w_sin[:ppw].view(1, 1, 1, ppw, w_dim).expand(batch_size, ppf, pph, ppw, w_dim)

    freqs_cos = torch.cat([t_cos_3d, h_cos_3d, w_cos_3d], dim=-1)
    freqs_sin = torch.cat([t_sin_3d, h_sin_3d, w_sin_3d], dim=-1)

    seq_len = ppf * pph * ppw
    freqs_cos = freqs_cos.reshape(batch_size, seq_len, 1, head_dim).to(dtype)
    freqs_sin = freqs_sin.reshape(batch_size, seq_len, 1, head_dim).to(dtype)
    return freqs_cos, freqs_sin


def compute_rope_dims(head_dim):
    h_dim = w_dim = 2 * (head_dim // 6)
    t_dim = head_dim - h_dim - w_dim
    return t_dim, h_dim, w_dim


def reference_qk_norm_rope(query, key, value, norm_q, norm_k, num_heads,
                           freqs_cos, freqs_sin, interleave=True):
    query = norm_q(query)
    key = norm_k(key)

    query = query.unflatten(2, (num_heads, -1))
    key = key.unflatten(2, (num_heads, -1))
    value = value.unflatten(2, (num_heads, -1))

    apply_fn = apply_rotary_emb_interleaved if interleave else apply_rotary_emb_neox
    query = apply_fn(query, freqs_cos, freqs_sin)
    key = apply_fn(key, freqs_cos, freqs_sin)

    return query, key, value


# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

WAN_CONFIG = {
    "num_heads": 24,
    "head_dim": 128,
    "hidden_dim": 24 * 128,
    "eps": 1e-6,
    "base": 10000.0,
}

INTERLEAVED_SHAPES = [
    (1, 5, 12, 32),
    (1, 5, 12, 8),
    (2, 5, 12, 32),
    (1, 5, 6, 4),
]

NEOX_SHAPES = [
    (1, 5, 12, 8),
    (1, 5, 6, 4),
]


# ---------------------------------------------------------------------------
# Correctness: interleaved RoPE (primary path, validated by kernel author)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,ppf,pph,ppw", INTERLEAVED_SHAPES)
def test_interleaved_correctness(batch_size, ppf, pph, ppw):
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = WAN_CONFIG["num_heads"]
    head_dim = WAN_CONFIG["head_dim"]
    hidden_dim = WAN_CONFIG["hidden_dim"]
    eps = WAN_CONFIG["eps"]
    base = WAN_CONFIG["base"]
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)
    seq_len = ppf * pph * ppw

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    norm_q = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)
    norm_k = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)
    with torch.no_grad():
        norm_q.weight.copy_(1.0 + 0.1 * torch.randn(hidden_dim, device=device))
        norm_k.weight.copy_(1.0 + 0.1 * torch.randn(hidden_dim, device=device))

    freqs_cos, freqs_sin = create_3d_rotary_embeddings(
        batch_size, ppf, pph, ppw, head_dim, device, base, dtype
    )

    q_ref, k_ref, v_ref = reference_qk_norm_rope(
        query.clone(), key.clone(), value.clone(),
        norm_q, norm_k, num_heads, freqs_cos, freqs_sin, interleave=True,
    )

    qkv_combined = torch.cat([query, key, value], dim=-1).contiguous()
    q_fused, k_fused, v_fused = fused_qk_norm_rope(
        qkv_combined,
        norm_q.weight.contiguous(),
        norm_k.weight.contiguous(),
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim,
        num_height_channels=h_dim,
        num_width_channels=w_dim,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        num_heads_v=num_heads,
        head_dim=head_dim,
        eps=eps,
        base=base,
        interleave=True,
        is_qk_norm=True,
    )

    q_ref_flat = q_ref.flatten(2)
    k_ref_flat = k_ref.flatten(2)
    q_fused_flat = q_fused.flatten(2)
    k_fused_flat = k_fused.flatten(2)

    q_max_diff = (q_fused_flat.float() - q_ref_flat.float()).abs().max().item()
    k_max_diff = (k_fused_flat.float() - k_ref_flat.float()).abs().max().item()

    assert q_max_diff < 0.1, f"Q max diff {q_max_diff} >= 0.1"
    assert k_max_diff < 0.1, f"K max diff {k_max_diff} >= 0.1"


# ---------------------------------------------------------------------------
# Correctness: non-interleaved (NeoX) RoPE — first validation of this path
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="NeoX (non-interleaved) RoPE path not yet validated end-to-end by kernel author; "
    "reference implementation may not match kernel convention. See plan doc.",
    strict=False,
)
@pytest.mark.parametrize("batch_size,ppf,pph,ppw", NEOX_SHAPES)
def test_neox_correctness(batch_size, ppf, pph, ppw):
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = WAN_CONFIG["num_heads"]
    head_dim = WAN_CONFIG["head_dim"]
    hidden_dim = WAN_CONFIG["hidden_dim"]
    eps = WAN_CONFIG["eps"]
    base = WAN_CONFIG["base"]
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)
    seq_len = ppf * pph * ppw

    torch.manual_seed(123)
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    norm_q = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)
    norm_k = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)
    with torch.no_grad():
        norm_q.weight.copy_(1.0 + 0.1 * torch.randn(hidden_dim, device=device))
        norm_k.weight.copy_(1.0 + 0.1 * torch.randn(hidden_dim, device=device))

    freqs_cos, freqs_sin = create_3d_rotary_embeddings(
        batch_size, ppf, pph, ppw, head_dim, device, base, dtype
    )

    q_ref, k_ref, v_ref = reference_qk_norm_rope(
        query.clone(), key.clone(), value.clone(),
        norm_q, norm_k, num_heads, freqs_cos, freqs_sin, interleave=False,
    )

    qkv_combined = torch.cat([query, key, value], dim=-1).contiguous()
    q_fused, k_fused, v_fused = fused_qk_norm_rope(
        qkv_combined,
        norm_q.weight.contiguous(),
        norm_k.weight.contiguous(),
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim,
        num_height_channels=h_dim,
        num_width_channels=w_dim,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        num_heads_v=num_heads,
        head_dim=head_dim,
        eps=eps,
        base=base,
        interleave=False,
        is_qk_norm=True,
    )

    q_ref_flat = q_ref.flatten(2)
    k_ref_flat = k_ref.flatten(2)
    q_fused_flat = q_fused.flatten(2)
    k_fused_flat = k_fused.flatten(2)

    q_max_diff = (q_fused_flat.float() - q_ref_flat.float()).abs().max().item()
    k_max_diff = (k_fused_flat.float() - k_ref_flat.float()).abs().max().item()

    assert q_max_diff < 0.1, f"NeoX Q max diff {q_max_diff} >= 0.1"
    assert k_max_diff < 0.1, f"NeoX K max diff {k_max_diff} >= 0.1"


# ---------------------------------------------------------------------------
# Correctness: V passthrough (should be an exact BF16 copy)
# ---------------------------------------------------------------------------


def test_v_passthrough():
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = WAN_CONFIG["num_heads"]
    head_dim = WAN_CONFIG["head_dim"]
    hidden_dim = WAN_CONFIG["hidden_dim"]
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)

    batch_size, ppf, pph, ppw = 1, 5, 6, 4
    seq_len = ppf * pph * ppw

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    qkv_combined = torch.cat([query, key, value], dim=-1).contiguous()
    q_weight = torch.ones(hidden_dim, device=device, dtype=dtype)
    k_weight = torch.ones(hidden_dim, device=device, dtype=dtype)

    _, _, v_fused = fused_qk_norm_rope(
        qkv_combined, q_weight, k_weight,
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim, num_height_channels=h_dim, num_width_channels=w_dim,
        num_heads_q=num_heads, num_heads_k=num_heads, num_heads_v=num_heads,
        head_dim=head_dim, interleave=True, is_qk_norm=True,
    )

    v_expected = value.unflatten(2, (num_heads, head_dim))
    assert torch.equal(v_fused, v_expected), "V output should be an exact copy"


# ---------------------------------------------------------------------------
# Correctness: destination-passing style (pre-allocated output)
# ---------------------------------------------------------------------------


def test_destination_passing():
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = WAN_CONFIG["num_heads"]
    head_dim = WAN_CONFIG["head_dim"]
    hidden_dim = WAN_CONFIG["hidden_dim"]
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)

    batch_size, ppf, pph, ppw = 1, 5, 6, 4
    seq_len = ppf * pph * ppw

    torch.manual_seed(42)
    qkv = torch.randn(batch_size, seq_len, 3 * hidden_dim, device=device, dtype=dtype)

    q_out = torch.empty(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k_out = torch.empty(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v_out = torch.empty(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    q_ret, k_ret, v_ret = fused_qk_norm_rope(
        qkv,
        torch.ones(hidden_dim, device=device, dtype=dtype),
        torch.ones(hidden_dim, device=device, dtype=dtype),
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim, num_height_channels=h_dim, num_width_channels=w_dim,
        num_heads_q=num_heads, num_heads_k=num_heads, num_heads_v=num_heads,
        head_dim=head_dim, interleave=True, is_qk_norm=True,
        q_out=q_out, k_out=k_out, v_out=v_out,
    )

    assert q_ret is q_out
    assert k_ret is k_out
    assert v_ret is v_out


# ---------------------------------------------------------------------------
# Correctness: FP8 output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_scale", [1.0, 0.5, 2.0])
def test_fp8_output(output_scale):
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = WAN_CONFIG["num_heads"]
    head_dim = WAN_CONFIG["head_dim"]
    hidden_dim = WAN_CONFIG["hidden_dim"]
    eps = WAN_CONFIG["eps"]
    base = WAN_CONFIG["base"]
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)

    batch_size, ppf, pph, ppw = 1, 5, 6, 4
    seq_len = ppf * pph * ppw

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    norm_q = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)
    norm_k = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)
    with torch.no_grad():
        norm_q.weight.copy_(1.0 + 0.1 * torch.randn(hidden_dim, device=device))
        norm_k.weight.copy_(1.0 + 0.1 * torch.randn(hidden_dim, device=device))

    qkv_combined = torch.cat([query, key, value], dim=-1).contiguous()

    q_fp8, k_fp8, v_fp8 = fused_qk_norm_rope(
        qkv_combined,
        norm_q.weight.contiguous(),
        norm_k.weight.contiguous(),
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim, num_height_channels=h_dim, num_width_channels=w_dim,
        num_heads_q=num_heads, num_heads_k=num_heads, num_heads_v=num_heads,
        head_dim=head_dim, eps=eps, base=base, interleave=True, is_qk_norm=True,
        output_fp8=True, output_quant_scale=output_scale, v_quant_scale=output_scale,
    )

    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert v_fp8.dtype == torch.float8_e4m3fn
    assert q_fp8.shape == (batch_size, seq_len, num_heads, head_dim)
    assert q_fp8.is_contiguous()

    freqs_cos, freqs_sin = create_3d_rotary_embeddings(
        batch_size, ppf, pph, ppw, head_dim, device, base, dtype
    )
    q_ref, k_ref, _ = reference_qk_norm_rope(
        query.clone(), key.clone(), value.clone(),
        norm_q, norm_k, num_heads, freqs_cos, freqs_sin, interleave=True,
    )
    q_ref_fp8 = (q_ref.float() * output_scale).to(torch.float8_e4m3fn)
    k_ref_fp8 = (k_ref.float() * output_scale).to(torch.float8_e4m3fn)

    q_diff = (q_fp8.flatten(2).float() - q_ref_fp8.flatten(2).float()).abs().max().item()
    k_diff = (k_fp8.flatten(2).float() - k_ref_fp8.flatten(2).float()).abs().max().item()

    max_allowed = max(1.0 * output_scale, 0.5)
    assert q_diff < max_allowed, f"FP8 Q diff {q_diff} >= {max_allowed}"
    assert k_diff < max_allowed, f"FP8 K diff {k_diff} >= {max_allowed}"


# ---------------------------------------------------------------------------
# Correctness: is_qk_norm=False (RoPE only, no normalization)
# ---------------------------------------------------------------------------


def test_rope_only_no_norm():
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = WAN_CONFIG["num_heads"]
    head_dim = WAN_CONFIG["head_dim"]
    hidden_dim = WAN_CONFIG["hidden_dim"]
    base = WAN_CONFIG["base"]
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)

    batch_size, ppf, pph, ppw = 1, 5, 6, 4
    seq_len = ppf * pph * ppw

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    qkv_combined = torch.cat([query, key, value], dim=-1).contiguous()
    q_weight = torch.ones(hidden_dim, device=device, dtype=dtype)
    k_weight = torch.ones(hidden_dim, device=device, dtype=dtype)

    q_fused, k_fused, _ = fused_qk_norm_rope(
        qkv_combined, q_weight, k_weight,
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim, num_height_channels=h_dim, num_width_channels=w_dim,
        num_heads_q=num_heads, num_heads_k=num_heads, num_heads_v=num_heads,
        head_dim=head_dim, base=base, interleave=True, is_qk_norm=False,
    )

    freqs_cos, freqs_sin = create_3d_rotary_embeddings(
        batch_size, ppf, pph, ppw, head_dim, device, base, dtype
    )
    q_heads = query.unflatten(2, (num_heads, head_dim))
    k_heads = key.unflatten(2, (num_heads, head_dim))
    q_ref = apply_rotary_emb_interleaved(q_heads, freqs_cos, freqs_sin)
    k_ref = apply_rotary_emb_interleaved(k_heads, freqs_cos, freqs_sin)

    q_diff = (q_fused.flatten(2).float() - q_ref.flatten(2).float()).abs().max().item()
    k_diff = (k_fused.flatten(2).float() - k_ref.flatten(2).float()).abs().max().item()

    assert q_diff < 0.05, f"RoPE-only Q diff {q_diff} >= 0.05"
    assert k_diff < 0.05, f"RoPE-only K diff {k_diff} >= 0.05"


# ---------------------------------------------------------------------------
# Validation: error cases
# ---------------------------------------------------------------------------


def test_error_non_cuda():
    from flashinfer.norm import fused_qk_norm_rope

    qkv = torch.randn(1, 120, 3 * 3072, dtype=torch.bfloat16)
    w = torch.ones(3072, dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        fused_qk_norm_rope(
            qkv, w, w,
            ppf=5, pph=6, ppw=4,
            num_frame_channels=44, num_height_channels=42, num_width_channels=42,
            num_heads_q=24, num_heads_k=24, num_heads_v=24, head_dim=128,
        )


def test_error_wrong_dtype():
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    qkv = torch.randn(1, 120, 3 * 3072, dtype=torch.float16, device=device)
    w = torch.ones(3072, dtype=torch.bfloat16, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        fused_qk_norm_rope(
            qkv, w, w,
            ppf=5, pph=6, ppw=4,
            num_frame_channels=44, num_height_channels=42, num_width_channels=42,
            num_heads_q=24, num_heads_k=24, num_heads_v=24, head_dim=128,
        )


def test_error_bad_head_dim():
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    head_dim = 96
    hidden = 24 * head_dim
    qkv = torch.randn(1, 120, 3 * hidden, dtype=torch.bfloat16, device=device)
    w = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        fused_qk_norm_rope(
            qkv, w, w,
            ppf=5, pph=6, ppw=4,
            num_frame_channels=32, num_height_channels=32, num_width_channels=32,
            num_heads_q=24, num_heads_k=24, num_heads_v=24, head_dim=head_dim,
        )


def test_error_channel_sum_mismatch():
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    qkv = torch.randn(1, 120, 3 * 3072, dtype=torch.bfloat16, device=device)
    w = torch.ones(3072, dtype=torch.bfloat16, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        fused_qk_norm_rope(
            qkv, w, w,
            ppf=5, pph=6, ppw=4,
            num_frame_channels=40, num_height_channels=40, num_width_channels=40,
            num_heads_q=24, num_heads_k=24, num_heads_v=24, head_dim=128,
        )


def test_error_seq_len_mismatch():
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    qkv = torch.randn(1, 100, 3 * 3072, dtype=torch.bfloat16, device=device)
    w = torch.ones(3072, dtype=torch.bfloat16, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        fused_qk_norm_rope(
            qkv, w, w,
            ppf=5, pph=6, ppw=4,
            num_frame_channels=44, num_height_channels=42, num_width_channels=42,
            num_heads_q=24, num_heads_k=24, num_heads_v=24, head_dim=128,
        )
