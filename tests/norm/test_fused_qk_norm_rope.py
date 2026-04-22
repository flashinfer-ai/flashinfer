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
    """NeoX-style RoPE matching the kernel's per-element frequency mapping.

    The kernel's NeoX path uses (dim_idx * 2) & ((1 << log_head_dim) - 1) to
    map each element to a frequency index, then swaps first/second halves via
    warp shuffle. Each element gets its own cos/sin value.

    hidden_states: [batch, seq, num_heads, head_dim]
    freqs_cos/sin: [batch, seq, 1, head_dim] — per-element cos/sin values
        computed using the kernel's mapped frequency index convention.
    """
    half = hidden_states.shape[-1] // 2
    x1 = hidden_states[..., :half]
    x2 = hidden_states[..., half:]
    cos1 = freqs_cos[..., :half]
    sin1 = freqs_sin[..., :half]
    cos2 = freqs_cos[..., half:]
    sin2 = freqs_sin[..., half:]
    out = torch.empty_like(hidden_states)
    out[..., :half] = x1 * cos1 - x2 * sin1
    out[..., half:] = x2 * cos2 + x1 * sin2
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


def create_3d_rotary_embeddings_neox(batch_size, ppf, pph, ppw, head_dim, device,
                                     base=10000.0, dtype=torch.bfloat16):
    """Create NeoX-style 3D rotary embeddings matching the kernel's per-element mapping.

    The kernel's NeoX path applies (dim_idx * 2) & ((1 << log_head_dim) - 1) to
    compute a mapped dimension index, then uses that to look up both the frequency
    AND the spatial dimension (frame/height/width) for position ID selection.

    This means each element gets its own cos/sin, and adjacent elements within
    a float2 pair can map to different spatial dimensions.

    Returns freqs_cos, freqs_sin with shape [batch, seq_len, 1, head_dim].
    """
    h_dim = w_dim = 2 * (head_dim // 6)
    t_dim = head_dim - h_dim - w_dim
    log_head_dim = head_dim.bit_length() - 1
    numElemsPerThread = head_dim // 32

    freq_table = []
    for i in range(t_dim // 2):
        freq_table.append(base ** (-2.0 * i / t_dim))
    for i in range(h_dim // 2):
        freq_table.append(base ** (-2.0 * i / h_dim))
    for i in range(w_dim // 2):
        freq_table.append(base ** (-2.0 * i / w_dim))
    freq_table = torch.tensor(freq_table, dtype=torch.float64, device=device)

    height_slice_start = t_dim
    width_slice_start = t_dim + h_dim

    # Build per-element freq and spatial-dim assignment following the kernel's mapping
    freq_per_elem = torch.zeros(head_dim, dtype=torch.float64, device=device)
    spatial_dim_per_elem = torch.zeros(head_dim, dtype=torch.long, device=device)

    for elem_idx in range(head_dim):
        laneId = elem_idx // numElemsPerThread
        within_lane = elem_idx % numElemsPerThread
        ii = within_lane // 2
        comp = within_lane % 2
        raw = laneId * numElemsPerThread + ii * 2 + comp
        mapped = (raw * 2) & ((1 << log_head_dim) - 1)
        freq_idx = mapped >> 1
        freq_per_elem[elem_idx] = freq_table[freq_idx]
        if mapped >= width_slice_start:
            spatial_dim_per_elem[elem_idx] = 2  # width
        elif mapped >= height_slice_start:
            spatial_dim_per_elem[elem_idx] = 1  # height
        else:
            spatial_dim_per_elem[elem_idx] = 0  # frame

    seq_len = ppf * pph * ppw
    cos_out = torch.zeros(batch_size, seq_len, 1, head_dim, dtype=torch.float64, device=device)
    sin_out = torch.zeros(batch_size, seq_len, 1, head_dim, dtype=torch.float64, device=device)

    for b in range(batch_size):
        for s in range(seq_len):
            tok = s
            pos_t = tok // (pph * ppw)
            pos_x = tok % (pph * ppw)
            pos_h = pos_x // ppw
            pos_w = pos_x % ppw
            pos_ids = torch.tensor([pos_t, pos_h, pos_w], dtype=torch.float64, device=device)

            for d in range(head_dim):
                pos_id = pos_ids[spatial_dim_per_elem[d]]
                theta = pos_id * freq_per_elem[d]
                cos_out[b, s, 0, d] = torch.cos(theta)
                sin_out[b, s, 0, d] = torch.sin(theta)

    return cos_out.to(dtype), sin_out.to(dtype)


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

# Configs from official WAN model releases (github.com/Wan-Video/Wan2.1, Wan2.2)
WAN_CONFIGS = {
    "wan2.1-1.3B": {  # wan/configs/wan_t2v_1_3B.py: dim=1536, num_heads=12
        "num_heads": 12,
        "head_dim": 128,
        "hidden_dim": 12 * 128,  # 1536
        "eps": 1e-6,
        "base": 10000.0,
    },
    "wan2.2-5B": {  # wan/configs/wan_ti2v_5B.py: dim=3072, num_heads=24
        "num_heads": 24,
        "head_dim": 128,
        "hidden_dim": 24 * 128,  # 3072
        "eps": 1e-6,
        "base": 10000.0,
    },
    "wan2.1-14B": {  # wan/configs/wan_t2v_14B.py: dim=5120, num_heads=40
        "num_heads": 40,
        "head_dim": 128,
        "hidden_dim": 40 * 128,  # 5120
        "eps": 1e-6,
        "base": 10000.0,
    },
}

# Default config used for most tests (WAN 2.2 5B, the production target)
WAN_CONFIG = WAN_CONFIGS["wan2.2-5B"]

INTERLEAVED_SHAPES = [
    (1, 5, 12, 32),   # Production: 5x12x32=1920
    (1, 5, 12, 8),    # Smaller: 5x12x8=480
    (1, 5, 48, 32),   # Larger: 5x48x32=7680
    (2, 5, 12, 32),   # batch=2
    (1, 5, 6, 4),     # Tiny: 5x6x4=120
    (4, 5, 12, 32),   # batch=4
    (1, 5, 12, 16),   # Half seq: 5x12x16=960
    (1, 10, 12, 32),  # Double frames: 10x12x32=3840
]

NEOX_SHAPES = [
    (1, 5, 12, 8),
    (1, 5, 6, 4),
    (2, 5, 12, 32),
]


# ---------------------------------------------------------------------------
# Correctness: interleaved RoPE (primary path, used in production)
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


@pytest.mark.parametrize("batch_size,ppf,pph,ppw", NEOX_SHAPES)
def test_neox_correctness(batch_size, ppf, pph, ppw):
    """NeoX (non-interleaved) RoPE path validation."""
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

    freqs_cos, freqs_sin = create_3d_rotary_embeddings_neox(
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
# Correctness: 2D (pre-flattened) input
# ---------------------------------------------------------------------------


def test_2d_input():
    """2D [num_tokens, hidden] input should produce same results as 3D."""
    from flashinfer.norm import fused_qk_norm_rope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = WAN_CONFIG["num_heads"]
    head_dim = WAN_CONFIG["head_dim"]
    hidden_dim = WAN_CONFIG["hidden_dim"]
    eps = WAN_CONFIG["eps"]
    base = WAN_CONFIG["base"]
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)

    batch_size, ppf, pph, ppw = 2, 5, 6, 4
    seq_len = ppf * pph * ppw
    num_tokens = batch_size * seq_len

    torch.manual_seed(42)
    qkv_3d = torch.randn(batch_size, seq_len, 3 * hidden_dim, device=device, dtype=dtype)
    qkv_2d = qkv_3d.view(num_tokens, 3 * hidden_dim).contiguous()

    kwargs = dict(
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim, num_height_channels=h_dim, num_width_channels=w_dim,
        num_heads_q=num_heads, num_heads_k=num_heads, num_heads_v=num_heads,
        head_dim=head_dim, eps=eps, base=base, interleave=True, is_qk_norm=True,
    )
    q_weight = torch.ones(hidden_dim, device=device, dtype=dtype)
    k_weight = torch.ones(hidden_dim, device=device, dtype=dtype)

    q_3d, k_3d, v_3d = fused_qk_norm_rope(qkv_3d, q_weight, k_weight, **kwargs)
    q_2d, k_2d, v_2d = fused_qk_norm_rope(qkv_2d, q_weight, k_weight, **kwargs)

    assert q_3d.ndim == 4, f"3D input should give 4D output, got {q_3d.ndim}D"
    assert q_2d.ndim == 3, f"2D input should give 3D output, got {q_2d.ndim}D"
    assert q_3d.shape == (batch_size, seq_len, num_heads, head_dim)
    assert q_2d.shape == (num_tokens, num_heads, head_dim)

    assert torch.equal(q_3d.view(num_tokens, num_heads, head_dim), q_2d)
    assert torch.equal(k_3d.view(num_tokens, num_heads, head_dim), k_2d)
    assert torch.equal(v_3d.view(num_tokens, num_heads, head_dim), v_2d)


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
# Correctness: multi-config (WAN 1.3B, 5B, 14B model sizes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_name",
    [
        "wan2.1-1.3B",
        "wan2.2-5B",
        pytest.param(
            "wan2.1-14B",
            marks=pytest.mark.xfail(
                reason="14B has num_heads=40 which exceeds kernel MAX_HEADS=32",
                raises=ValueError,
                strict=True,
            ),
        ),
    ],
)
def test_multi_config(config_name):
    """Test across WAN model sizes: 1.3B (12 heads), 5B (24 heads), 14B (40 heads)."""
    from flashinfer.norm import fused_qk_norm_rope

    cfg = WAN_CONFIGS[config_name]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    hidden_dim = cfg["hidden_dim"]
    eps = cfg["eps"]
    base = cfg["base"]
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

    freqs_cos, freqs_sin = create_3d_rotary_embeddings(
        batch_size, ppf, pph, ppw, head_dim, device, base, dtype
    )

    q_ref, k_ref, _ = reference_qk_norm_rope(
        query.clone(), key.clone(), value.clone(),
        norm_q, norm_k, num_heads, freqs_cos, freqs_sin, interleave=True,
    )

    qkv_combined = torch.cat([query, key, value], dim=-1).contiguous()
    q_fused, k_fused, _ = fused_qk_norm_rope(
        qkv_combined,
        norm_q.weight.contiguous(),
        norm_k.weight.contiguous(),
        ppf=ppf, pph=pph, ppw=ppw,
        num_frame_channels=t_dim, num_height_channels=h_dim, num_width_channels=w_dim,
        num_heads_q=num_heads, num_heads_k=num_heads, num_heads_v=num_heads,
        head_dim=head_dim, eps=eps, base=base, interleave=True, is_qk_norm=True,
    )

    q_diff = (q_fused.flatten(2).float() - q_ref.flatten(2).float()).abs().max().item()
    k_diff = (k_fused.flatten(2).float() - k_ref.flatten(2).float()).abs().max().item()

    assert q_diff < 0.1, f"{config_name} Q max diff {q_diff} >= 0.1"
    assert k_diff < 0.1, f"{config_name} K max diff {k_diff} >= 0.1"


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
