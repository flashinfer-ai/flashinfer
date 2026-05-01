"""
Benchmark for fused DIT LayerNorm kernels vs eager PyTorch baseline.

Measures performance across WAN model shapes for all three modes:
- gate_residual_gamma_beta
- gate_residual_scale_shift
- residual_scale_shift

Usage:
    python benchmarks/bench_fused_dit_layernorm.py
    python benchmarks/bench_fused_dit_layernorm.py --gpu 2
"""

import argparse

import numpy as np
import torch

from flashinfer.diffusion_ops import (
    fused_dit_gate_residual_layernorm_gamma_beta,
    fused_dit_gate_residual_layernorm_scale_shift,
    fused_dit_residual_layernorm_scale_shift,
)
from flashinfer.testing.utils import bench_gpu_time

EPSILON = 1e-6
HIDDEN_DIM = 3072

BENCH_SHAPES = [
    (1, 1920, "bs=1 seq=1920 (production)"),
    (1, 768, "bs=1 seq=768"),
    (2, 1920, "bs=2 seq=1920"),
    (2, 768, "bs=2 seq=768"),
    (4, 1920, "bs=4 seq=1920"),
]


def make_inputs(batch_size, seq_len, device):
    torch.manual_seed(42)
    input_t = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_t)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)

    scale_shift_table = torch.randn(
        1, 6, HIDDEN_DIM, dtype=torch.float32, device=device
    )
    temb = torch.randn(
        batch_size, seq_len, 6, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    temb_chunks = temb.chunk(6, dim=2)
    table_chunks = scale_shift_table.chunk(6, dim=1)

    return {
        "input": input_t,
        "residual": residual,
        "gamma": gamma,
        "beta": beta,
        "gate": temb_chunks[2].squeeze(2),
        "gate_bias": table_chunks[2].squeeze(1),
        "scale": temb_chunks[1].squeeze(2),
        "scale_bias": table_chunks[1].squeeze(1),
        "shift": temb_chunks[0].squeeze(2),
        "shift_bias": table_chunks[0].squeeze(1),
        "c_gate": temb_chunks[5].squeeze(2),
        "c_gate_bias": table_chunks[5].squeeze(1),
        "c_scale": temb_chunks[4].squeeze(2),
        "c_scale_bias": table_chunks[4].squeeze(1),
        "c_shift": temb_chunks[3].squeeze(2),
        "c_shift_bias": table_chunks[3].squeeze(1),
    }


def eager_gate_residual_gamma_beta(d):
    r = d["residual"].float() + d["input"].float() * (
        d["gate"].float() + d["gate_bias"].float()
    )
    n = torch.layer_norm(
        r, [HIDDEN_DIM], weight=d["gamma"], bias=d["beta"], eps=EPSILON
    )
    return r.to(torch.bfloat16), n.to(torch.bfloat16)


def eager_gate_residual_scale_shift(d):
    r = d["residual"].float() + d["input"].float() * (
        d["c_gate"].float() + d["c_gate_bias"].float()
    )
    n = torch.layer_norm(r, [HIDDEN_DIM], eps=EPSILON)
    n = n * (1 + d["scale"].float() + d["scale_bias"].float()) + (
        d["shift"].float() + d["shift_bias"].float()
    )
    return r.to(torch.bfloat16), n.to(torch.bfloat16)


def eager_residual_scale_shift(d):
    r = d["residual"].float() + d["input"].float()
    n = torch.layer_norm(r, [HIDDEN_DIM], eps=EPSILON)
    n = n * (1 + d["c_scale"].float() + d["c_scale_bias"].float()) + (
        d["c_shift"].float() + d["c_shift_bias"].float()
    )
    return r.to(torch.bfloat16), n.to(torch.bfloat16)


def bench_mode(mode_name, eager_fn, fused_fn, shapes, device):
    print(f"\n{'=' * 80}")
    print(f"Mode: {mode_name}")
    print(f"{'=' * 80}")
    print(f"{'Shape':<40} {'Eager (ms)':>12} {'Fused (ms)':>12} {'Speedup':>10}")
    print("-" * 80)

    for batch_size, seq_len, desc in shapes:
        d = make_inputs(batch_size, seq_len, device)

        eager_times = bench_gpu_time(
            lambda: eager_fn(d), enable_cupti=True, dry_run_iters=10, repeat_iters=100
        )
        fused_times = bench_gpu_time(
            lambda: fused_fn(d), enable_cupti=True, dry_run_iters=10, repeat_iters=100
        )

        eager_ms = float(np.median(eager_times))
        fused_ms = float(np.median(fused_times))
        speedup = eager_ms / fused_ms if fused_ms > 0 else 0

        print(f"{desc:<40} {eager_ms:>12.4f} {fused_ms:>12.4f} {speedup:>9.2f}x")

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused DIT LayerNorm")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}")
    print(f"Config: WAN 2.2 5B (hidden_dim={HIDDEN_DIM})")

    bench_mode(
        "gate_residual_gamma_beta",
        eager_gate_residual_gamma_beta,
        lambda d: fused_dit_gate_residual_layernorm_gamma_beta(
            d["input"],
            d["residual"],
            d["gate"],
            d["gamma"],
            d["beta"],
            gate_bias=d["gate_bias"],
            epsilon=EPSILON,
        ),
        BENCH_SHAPES,
        device,
    )

    bench_mode(
        "gate_residual_scale_shift",
        eager_gate_residual_scale_shift,
        lambda d: fused_dit_gate_residual_layernorm_scale_shift(
            d["input"],
            d["residual"],
            d["c_gate"],
            d["scale"],
            d["shift"],
            gate_bias=d["c_gate_bias"],
            scale_bias=d["scale_bias"],
            shift_bias=d["shift_bias"],
            epsilon=EPSILON,
        ),
        BENCH_SHAPES,
        device,
    )

    bench_mode(
        "residual_scale_shift",
        eager_residual_scale_shift,
        lambda d: fused_dit_residual_layernorm_scale_shift(
            d["input"],
            d["c_scale"],
            d["c_shift"],
            residual=d["residual"],
            scale_bias=d["c_scale_bias"],
            shift_bias=d["c_shift_bias"],
            epsilon=EPSILON,
        ),
        BENCH_SHAPES,
        device,
    )


if __name__ == "__main__":
    main()
