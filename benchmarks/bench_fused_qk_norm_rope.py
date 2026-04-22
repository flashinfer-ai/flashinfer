"""
Benchmark for fused QKNorm + 3D RoPE kernel vs eager PyTorch baseline.

Measures performance across WAN model shapes and compares:
- Eager: separate nn.RMSNorm + manual interleaved RoPE in PyTorch
- Fused: flashinfer.video_gen_ops.fused_qk_norm_rope (single kernel)

Usage:
    python benchmarks/bench_fused_qk_norm_rope.py
    python benchmarks/bench_fused_qk_norm_rope.py --gpu 2   # run on specific GPU
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn

from flashinfer.testing.utils import bench_gpu_time
from flashinfer.video_gen_ops import fused_qk_norm_rope


def compute_rope_dims(head_dim):
    h_dim = w_dim = 2 * (head_dim // 6)
    t_dim = head_dim - h_dim - w_dim
    return t_dim, h_dim, w_dim


def apply_rotary_emb_interleaved(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def get_1d_rotary_pos_embed(dim, length, theta, device):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim)
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
    return (
        freqs_cos.reshape(batch_size, seq_len, 1, head_dim).to(dtype),
        freqs_sin.reshape(batch_size, seq_len, 1, head_dim).to(dtype),
    )


BENCH_SHAPES = [
    # (batch, ppf, pph, ppw, description)
    (1, 5, 12, 32, "480p production (1920 tokens)"),
    (1, 5, 12, 8, "480p small (480 tokens)"),
    (1, 5, 48, 32, "720p large (7680 tokens)"),
    (2, 5, 12, 32, "batch=2 (3840 tokens)"),
    (1, 5, 6, 4, "tiny (120 tokens)"),
    (4, 5, 12, 32, "batch=4 (7680 tokens)"),
    (1, 5, 12, 16, "half seq (960 tokens)"),
    (1, 10, 12, 32, "double frames (3840 tokens)"),
]


def bench_one_shape(batch_size, ppf, pph, ppw, num_heads, head_dim, eps, base, device):
    seq_len = ppf * pph * ppw
    hidden_dim = num_heads * head_dim
    t_dim, h_dim, w_dim = compute_rope_dims(head_dim)
    dtype = torch.bfloat16

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    norm_q = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)
    norm_k = nn.RMSNorm(hidden_dim, eps=eps).to(device).to(dtype)

    freqs_cos, freqs_sin = create_3d_rotary_embeddings(
        batch_size, ppf, pph, ppw, head_dim, device, base, dtype
    )

    qkv_combined = torch.cat([query, key, value], dim=-1).contiguous()
    q_weight = norm_q.weight.contiguous()
    k_weight = norm_k.weight.contiguous()

    def eager_fn():
        q_normed = norm_q(query)
        k_normed = norm_k(key)
        q_heads = q_normed.unflatten(2, (num_heads, -1))
        k_heads = k_normed.unflatten(2, (num_heads, -1))
        v_heads = value.unflatten(2, (num_heads, -1))
        q_out = apply_rotary_emb_interleaved(q_heads, freqs_cos, freqs_sin)
        k_out = apply_rotary_emb_interleaved(k_heads, freqs_cos, freqs_sin)
        return q_out, k_out, v_heads

    def fused_fn():
        return fused_qk_norm_rope(
            qkv_combined, q_weight, k_weight,
            ppf=ppf, pph=pph, ppw=ppw,
            num_frame_channels=t_dim, num_height_channels=h_dim, num_width_channels=w_dim,
            num_heads_q=num_heads, num_heads_k=num_heads, num_heads_v=num_heads,
            head_dim=head_dim, eps=eps, base=base, interleave=True, is_qk_norm=True,
        )

    eager_times = bench_gpu_time(eager_fn, enable_cupti=True, dry_run_iters=10, repeat_iters=100)
    fused_times = bench_gpu_time(fused_fn, enable_cupti=True, dry_run_iters=10, repeat_iters=100)

    eager_ms = float(np.median(eager_times))
    fused_ms = float(np.median(fused_times))
    return eager_ms, fused_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused QKNorm + 3D RoPE")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)

    num_heads = 24
    head_dim = 128
    eps = 1e-6
    base = 10000.0

    print(f"GPU: {gpu_name}")
    print(f"Config: WAN 2.2 5B (num_heads={num_heads}, head_dim={head_dim})")
    print()
    print(f"{'Shape':<50} {'Eager (ms)':>12} {'Fused (ms)':>12} {'Speedup':>10}")
    print("-" * 90)

    for batch_size, ppf, pph, ppw, desc in BENCH_SHAPES:
        seq_len = ppf * pph * ppw
        shape_str = f"B={batch_size} {ppf}x{pph}x{ppw}={seq_len:>5} ({desc})"

        eager_ms, fused_ms = bench_one_shape(
            batch_size, ppf, pph, ppw, num_heads, head_dim, eps, base, device,
        )

        speedup = eager_ms / fused_ms if fused_ms > 0 else 0
        print(f"{shape_str:<50} {eager_ms:>12.4f} {fused_ms:>12.4f} {speedup:>9.2f}x")

    print("-" * 90)


if __name__ == "__main__":
    main()
