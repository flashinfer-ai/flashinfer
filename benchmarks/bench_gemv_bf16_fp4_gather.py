# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""
Gathered-row BF16 x NVFP4 GEMV with scatter epilogue (SM12x).

    out[m, off + row_ids[m, j]] = alpha * a[m] @ dequant(b[row_ids[m, j]])

This is the dynamic-row part of a sparse draft LM head: each draft token reads
``J`` vocabulary rows picked by a top-k over a cheap scorer and writes their
exact logits into a full-vocabulary tensor.  Default shapes: ``R = 215552``
candidate rows (vocabulary 248320 minus 32768 static rows), ``J = 16384``
picks, hidden ``K`` in {2048, 2560}.

Row ids come from ``torch.topk(sorted=False)`` over random scores, which
returns them ascending; ``--ids-from flashinfer`` uses ``flashinfer.top_k``
instead.  Ids are drawn independently per token, so cross-token row overlap
is the random baseline.

Columns:

    us          median GPU time of gemv_bf16_fp4_gather (L2 flushed per iter)
    torch us    gather + dequantize to bf16 + bmm + scatter_ in PyTorch
    GB/s        M * J * (K/2 + K/16) bytes of rows requested / t
    max rel     max |kernel - fp32 reference| / max |reference|

Usage:

    python benchmarks/bench_gemv_bf16_fp4_gather.py
    python benchmarks/bench_gemv_bf16_fp4_gather.py --k 2560 --m 1 8 32 --cupti
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

_E2M1 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip


def torch_gather_gemv(a, b, sf, alpha, row_ids, out, off, lut):
    """Dequantize the picked rows to bf16, batched GEMV, scatter."""
    m, j = row_ids.shape
    flat = row_ids.reshape(-1)
    codes = b.index_select(0, flat).long()
    w = torch.stack((lut[codes & 15], lut[codes >> 4]), dim=-1).reshape(m * j, -1)
    scale = sf.index_select(0, flat).to(torch.bfloat16).repeat_interleave(16, dim=1)
    w = (w * scale).reshape(m, j, -1)
    logits = torch.bmm(w, a.unsqueeze(-1)).squeeze(-1) * alpha.to(torch.bfloat16)
    out.scatter_(1, row_ids + off, logits.to(out.dtype))
    return out


def run(args, k, m):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    r, j, off = args.rows, args.slots, args.col_offset
    b = torch.randint(0, 256, (r, k // 2), dtype=torch.uint8, device=dev, generator=g)
    sf = (torch.rand(r, k // 16, device=dev, generator=g) * 1.5 + 0.25).to(
        torch.float8_e4m3fn
    )
    alpha = torch.tensor([0.0125], device=dev)
    a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    scores = torch.randn(m, r, device=dev)
    if args.ids_from == "flashinfer":
        row_ids = flashinfer.top_k(scores.bfloat16(), j, sorted=False)[1].contiguous()
    else:
        row_ids = scores.topk(j, sorted=False).indices.contiguous()
    out = torch.full((m, off + r), float("-inf"), dtype=torch.bfloat16, device=dev)

    flashinfer.gemv_bf16_fp4_gather(a, b, sf, alpha, row_ids, out, out_col_offset=off)
    torch.cuda.synchronize()
    lut = torch.tensor(_E2M1, device=dev)
    ref = torch.empty(m, j, device=dev)
    for i in range(m):
        codes = b[row_ids[i]].long()
        w = torch.stack((lut[codes & 15], lut[codes >> 4]), dim=-1).reshape(j, -1)
        w = w * sf[row_ids[i]].float().repeat_interleave(16, dim=1) * alpha
        ref[i] = w @ a[i].float()
    got = out.float().gather(1, row_ids + off)
    max_rel = ((got - ref).abs().max() / ref.abs().max()).item()

    def kernel():
        flashinfer.gemv_bf16_fp4_gather(
            a, b, sf, alpha, row_ids, out, out_col_offset=off
        )

    t = np.median(bench_gpu_time(kernel, enable_cupti=args.cupti))
    t_torch = float("nan")
    if not args.no_torch:
        lut_bf16 = lut.to(torch.bfloat16)
        t_torch = np.median(
            bench_gpu_time(
                lambda: torch_gather_gemv(a, b, sf, alpha, row_ids, out, off, lut_bf16),
                enable_cupti=args.cupti,
            )
        )
    gbs = m * j * (k // 2 + k // 16) / (t * 1e-3) / 1e9
    print(
        f"{k:6d} {m:4d} {j:6d} {t * 1e3:10.1f} {t_torch * 1e3:10.1f} "
        f"{t_torch / t:8.2f}x {gbs:8.0f} {max_rel:10.2e}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--k", type=int, nargs="+", default=[2048, 2560])
    p.add_argument("--m", type=int, nargs="+", default=[1, 4, 8, 32])
    p.add_argument("--rows", type=int, default=215552)
    p.add_argument("--slots", type=int, default=16384)
    p.add_argument("--col-offset", type=int, default=32768)
    p.add_argument("--ids-from", choices=["torch", "flashinfer"], default="torch")
    p.add_argument("--cupti", action="store_true")
    p.add_argument("--no-torch", action="store_true", help="skip the torch path")
    args = p.parse_args()
    print(f"device: {torch.cuda.get_device_name()}  R={args.rows}  ids={args.ids_from}")
    print(
        f"{'K':>6} {'M':>4} {'J':>6} {'us':>10} {'torch us':>10} {'speedup':>9} "
        f"{'GB/s':>8} {'max rel':>10}"
    )
    for k in args.k:
        for m in args.m:
            run(args, k, m)


if __name__ == "__main__":
    main()
