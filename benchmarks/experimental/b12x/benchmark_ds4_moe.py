"""DS4-Flash TP2 MoE timing on synthetic MXFP4 weights.

E=256, K=4096, I_tp=1024, top-k 6. Each b12x mode consumes BF16 inputs.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from b12x._lib.intrinsics import _fp4_encode_nibbles, fp4_quantize_values_torch

DS4_E = 256
DS4_K = 4096
DS4_I_TP = 1024
DS4_TOPK = 6


def quantize_mxfp4_batched(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, rows, cols] float -> (packed fp4 [E, rows, cols//2] u8,
    e8m0 scales [E, rows, cols//32] u8). Batched port of the
    test_w4a8_dynamic_kernel reference quantizer."""
    E, rows, cols = w.shape
    blocked = w.view(E, rows, cols // 32, 32)
    bmax = blocked.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(bmax > 0, bmax / 6.0, torch.ones_like(bmax))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    byte = (
        torch.where(bmax > 0, exponent + 127, torch.zeros_like(exponent))
        .to(torch.uint8)
        .squeeze(-1)
        .contiguous()
    )
    scale = torch.where(bmax > 0, torch.exp2(exponent), torch.zeros_like(exponent))
    q = fp4_quantize_values_torch(
        torch.where(
            scale > 0, blocked / scale.clamp(min=1e-30), torch.zeros_like(blocked)
        ).view(E, rows, cols)
    )
    nib = _fp4_encode_nibbles(q)
    pair = nib.view(E, rows, cols // 2, 2)
    packed = (pair[..., 0] | (pair[..., 1] << 4)).contiguous()
    return packed, byte


def _make_quantized_stack(
    E: int, rows: int, cols: int, *, gen: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random mxfp4 stack built a few experts at a time (the float source and
    quantizer temporaries are ~4x the packed size)."""
    packed = torch.empty(E, rows, cols // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(E, rows, cols // 32, dtype=torch.uint8, device=device)
    chunk = max(1, min(E, (1 << 28) // max(1, rows * cols)))
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        w = torch.randn(e1 - e0, rows, cols, generator=gen, device=device) * 0.05
        p, s = quantize_mxfp4_batched(w)
        packed[e0:e1] = p
        scales[e0:e1] = s
        del w, p, s
    return packed, scales


def make_synthetic_mxfp4_moe(
    E: int, k: int, n: int, *, seed: int, device: torch.device
) -> dict:
    """Kernel-order ([up; gate]) random mxfp4 expert weights + e8m0 grids."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    w13_fp4, w13_mx = _make_quantized_stack(E, 2 * n, k, gen=gen, device=device)
    w2_fp4, w2_mx = _make_quantized_stack(E, k, n, gen=gen, device=device)
    ones = torch.ones(E, dtype=torch.float32, device=device)
    return {
        "w13_fp4": w13_fp4,
        "w13_mx": w13_mx,
        "w2_fp4": w2_fp4,
        "w2_mx": w2_mx,
        "alphas": ones,
        "input_scale": ones,
    }


def moe_flops(m: int, k: int, n: int, topk: int) -> float:
    rows = m * topk
    return float(rows) * (2.0 * k * 2 * n + 2.0 * n * k)


def _bench_b12x(
    mode: str,
    experts,
    m: int,
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> float:
    from b12x.moe.fused_moe._impl import (
        allocate_tp_moe_workspace_pool,
        b12x_moe_fp4,
        build_tp_moe_fp4_binding,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    gen = torch.Generator(device=device)
    gen.manual_seed(1000 + m)
    x = (torch.randn(m, DS4_K, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(m, DS4_E, generator=gen, device=device)
    topk_logits, topk_ids = torch.topk(logits, DS4_TOPK, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    topk_ids = topk_ids.to(torch.int32)
    out = torch.empty(m, DS4_K, dtype=torch.bfloat16, device=device)

    workspace = allocate_tp_moe_workspace_pool()
    binding = build_tp_moe_fp4_binding(
        scratch=workspace,
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=out,
        input_scales_static=True,
        quant_mode=mode,
    )

    def launch():
        b12x_moe_fp4(binding=binding)

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _prepare_b12x_experts(mode: str, source_weights: dict):
    """Transfer one private source copy into the mode's canonical owner."""
    from b12x.moe.fused_moe._impl import (
        plan_b12x_fp4_moe_weights,
        prepare_b12x_fp4_moe_weights,
    )

    owned = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in source_weights.items()
    }
    plan = plan_b12x_fp4_moe_weights(
        quant_modes=mode,
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=DS4_E,
        hidden_size=DS4_K,
        intermediate_size=DS4_I_TP,
        w13_layout="w13",
    )
    return prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_global_scale=owned["alphas"],
        w2_global_scale=owned["alphas"],
        w1_fp4=owned["w13_fp4"],
        w1_blockscale=owned["w13_mx"],
        w2_fp4=owned["w2_fp4"],
        w2_blockscale=owned["w2_mx"],
        a1_gscale=owned["input_scale"],
        a2_gscale=owned["input_scale"],
        params_dtype=torch.bfloat16,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="w4a8_mx,w4a16")
    parser.add_argument("--m", default="1024,4096,8192,16384")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parents[3],
    ).stdout.strip()
    print(
        f"# DS4-Flash TP2 shapes: E={DS4_E} K={DS4_K} I_tp={DS4_I_TP} "
        f"topk={DS4_TOPK} | commit={commit} | "
        f"gpu={torch.cuda.get_device_name(device)} | "
        f"iters={args.iters} warmup={args.warmup}"
    )
    weights = make_synthetic_mxfp4_moe(
        DS4_E, DS4_K, DS4_I_TP, seed=args.seed, device=device
    )
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    if not modes or set(modes) - {"w4a8_mx", "w4a16"}:
        parser.error("--modes must select w4a8_mx and/or w4a16")
    experts_by_mode = {mode: _prepare_b12x_experts(mode, weights) for mode in modes}
    ms_list = [int(s) for s in args.m.split(",") if s.strip()]
    print(f"{'m':>7} | " + " | ".join(f"{mode:>22}" for mode in modes))
    for m in ms_list:
        cells = []
        for mode in modes:
            ms = _bench_b12x(
                mode,
                experts_by_mode[mode],
                m,
                iters=args.iters,
                warmup=args.warmup,
                device=device,
            )
            tflops = moe_flops(m, DS4_K, DS4_I_TP, DS4_TOPK) / (ms * 1e-3) / 1e12
            cells.append(f"{ms:8.3f} ms {tflops:6.1f} TF")
        print(f"{m:>7} | " + " | ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
