# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark: Cake fused QK RMSNorm + NeoX RoPE + FP8 quantize + paged KV append.

Arms (identical inputs, caller-owned buffers re-created per arm):

- ``cake``: :func:`flashinfer.cake_fused_qk_rope_fp8_append.cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache`
  (one launch).
- ``stable``: the step-wise pipeline from stable FlashInfer / torch ops
  (``flashinfer.norm.rmsnorm`` -> ``apply_rope_pos_ids_inplace`` -> torch FP8 quantize
  -> ``flashinfer.page.append_paged_kv_cache``), report only: its BF16 intermediates
  cannot meet the FP8 contract tolerance and it does not clear the last-page tail.

Timing: ``flashinfer.testing.bench_gpu_time`` (CUPTI, cold L2) over the whole call (the
union of every kernel one public call issues), median over ``--iters`` after ``--warmup``,
arms interleaved per group so clock drift affects all arms alike.

    python benchmarks/bench_cake_fused_qk_rope_fp8_append.py --rows P1,P2,H1 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

import torch

import flashinfer
from flashinfer.cake_fused_qk_rope_fp8_append import (
    cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache,
)
from flashinfer.testing.utils import bench_gpu_time

HEAD_DIM = 128
FP8_MAX = 448.0


@dataclass(frozen=True)
class Row:
    name: str
    hq: int
    hkv: int
    batch: int
    q_len: int
    ctx: int
    quant: int = 1
    norm: int = 2
    page_size: int = 64


ROWS = {
    r.name: r
    for r in (
        # decode (B32 / Q1) and prefill (B8 / Q16), both head configurations, both quant policies
        Row("D1", 8, 1, 32, 1, 2048, quant=1),
        Row("D1n0", 8, 1, 32, 1, 2048, quant=1, norm=0),
        Row("D1n1", 8, 1, 32, 1, 2048, quant=1, norm=1),
        Row("D2", 8, 1, 32, 1, 2048, quant=2),
        Row("P1", 8, 1, 8, 16, 2048, quant=1),
        Row("P2", 8, 1, 8, 16, 2048, quant=2),
        Row("H1", 64, 8, 32, 1, 2048, quant=1),
        Row("H2", 64, 8, 32, 1, 2048, quant=2),
        Row("HP1", 64, 8, 8, 16, 2048, quant=1),
        Row("HP2", 64, 8, 8, 16, 2048, quant=2),
        # dynamic-scale padding boundary (max_seqlen = Q)
        Row("E127", 8, 1, 4, 127, 2048, quant=1),
        Row("E128", 8, 1, 4, 128, 2048, quant=1),
        Row("E129", 8, 1, 4, 129, 2048, quant=1),
        # report-only extremes
        Row("X1", 8, 1, 1, 1, 2048),
        Row("X128", 8, 1, 128, 1, 2048),
        Row("X256", 8, 1, 256, 1, 2048),
        Row("XP1024", 8, 1, 1, 1024, 2048),
        Row("Xpage16", 8, 1, 32, 1, 2048, page_size=16),
        Row("Xpage128", 8, 1, 32, 1, 2048, page_size=128),
        Row("Xctx32k", 8, 1, 32, 1, 32768),
    )
}


def rotary_cos_sin_table(
    max_positions: int, device, base: float = 10000.0
) -> torch.Tensor:
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    pos = torch.arange(max_positions, dtype=torch.float64, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)


def build_inputs(row: Row, device, seed: int = 893):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = row.batch
    q_lens = [row.q_len] * B
    seq_lens = [row.ctx + row.q_len] * B
    pages_per_req = [(s + row.page_size - 1) // row.page_size for s in seq_lens]
    max_pages = max(pages_per_req)
    total_pages = sum(pages_per_req) + 4
    perm = torch.randperm(total_pages, generator=g)
    page_indices = torch.zeros((B, max_pages), dtype=torch.int32)
    cursor = 0
    for b in range(B):
        n = pages_per_req[b]
        page_indices[b, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
    q_indptr = torch.zeros(B + 1, dtype=torch.int32)
    q_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    T = int(q_indptr[-1])
    width = (row.hq + 2 * row.hkv) * HEAD_DIM
    qkv = (
        torch.randn((T, width), generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    q_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16)
    k_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16)
    positions = torch.cat(
        [
            torch.arange(row.ctx, row.ctx + row.q_len, dtype=torch.int32)
            for _ in range(B)
        ]
    )
    batch_indices = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int32), row.q_len
    )
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(torch.tensor(pages_per_req, dtype=torch.int32), 0)
    kv_indices = torch.cat([page_indices[b, : pages_per_req[b]] for b in range(B)])
    kv_last_page_len = torch.tensor(
        [(s - 1) % row.page_size + 1 for s in seq_lens], dtype=torch.int32
    )
    return {
        "row": row,
        "T": T,
        "is_prefill": row.q_len > 1,
        "max_seqlen": row.q_len if row.q_len > 1 else 0,
        "qkv": qkv,
        "cos_sin": rotary_cos_sin_table(max(seq_lens) + 1, device),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "q_indptr": q_indptr.to(device),
        "page_indices": page_indices.to(device),
        "q_norm_weight": q_w.float().to(device),
        "k_norm_weight": k_w.float().to(device),
        "q_norm_weight_bf16": q_w.to(device),
        "k_norm_weight_bf16": k_w.to(device),
        "k_scale": torch.tensor([0.02], dtype=torch.float32, device=device),
        "v_scale": torch.tensor([0.03], dtype=torch.float32, device=device),
        "q_scale_inv": torch.tensor([1.0 / 0.02], dtype=torch.float32, device=device),
        "positions": positions.to(device),
        "batch_indices": batch_indices.to(device),
        "kv_indptr": kv_indptr.to(device),
        "kv_indices": kv_indices.to(device),
        "kv_last_page_len": kv_last_page_len.to(device),
        "total_pages": total_pages,
    }


def fresh_buffers(inp, device):
    row = inp["row"]
    shape = (inp["total_pages"], row.page_size, row.hkv, HEAD_DIM)
    kc = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    vc = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    oq = torch.empty(
        (inp["T"], row.hq, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    if row.quant == 2:
        qs = torch.empty(0, dtype=torch.float32, device=device)
    elif inp["is_prefill"]:
        aligned = (inp["max_seqlen"] + 127) // 128 * 128
        qs = torch.empty(
            (row.batch, row.hq, aligned), dtype=torch.float32, device=device
        )
    else:
        qs = torch.empty((inp["T"], row.hq), dtype=torch.float32, device=device)
    flags = torch.zeros((row.batch, row.hkv), dtype=torch.int32, device=device)
    return kc, vc, oq, qs, flags


def _fused_kwargs(inp, bufs):
    row = inp["row"]
    kc, vc, oq, qs, flags = bufs
    return dict(
        args=(
            inp["qkv"],
            inp["cos_sin"],
            inp["seq_lens"],
            inp["q_indptr"],
            inp["page_indices"],
            (kc, vc),
            inp["is_prefill"],
            inp["k_scale"],
            inp["v_scale"],
            row.quant,
        ),
        kwargs=dict(
            max_seqlen=inp["max_seqlen"],
            upper_max=FP8_MAX,
            q_scale_inv=inp["q_scale_inv"] if row.quant == 2 else None,
            q_norm_weight=inp["q_norm_weight"] if row.norm else None,
            k_norm_weight=inp["k_norm_weight"] if row.norm else None,
            qk_norm_policy=row.norm,
            out_q=oq,
            q_scale=qs,
            split_k_flag=flags,
        ),
    )


def make_cake_call(inp, device):
    bufs = fresh_buffers(inp, device)
    bound = _fused_kwargs(inp, bufs)

    def call():
        cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            *bound["args"], **bound["kwargs"]
        )

    return call, bufs


def make_stable_call(inp, device):
    """rmsnorm + apply_rope_pos_ids_inplace + torch FP8 quantize + append_paged_kv_cache."""
    row = inp["row"]
    kc, vc, oq, qs, flags = bufs = fresh_buffers(inp, device)
    T = inp["T"]
    q = inp["qkv"][:, : row.hq * HEAD_DIM].view(T, row.hq, HEAD_DIM)
    k = inp["qkv"][:, row.hq * HEAD_DIM : (row.hq + row.hkv) * HEAD_DIM].view(
        T, row.hkv, HEAD_DIM
    )
    v = inp["qkv"][:, (row.hq + row.hkv) * HEAD_DIM :].view(T, row.hkv, HEAD_DIM)
    tmp_q, tmp_k = torch.empty_like(q), torch.empty_like(k)
    norm_q, norm_k = torch.empty_like(q), torch.empty_like(k)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    qw, kw = inp["q_norm_weight_bf16"], inp["k_norm_weight_bf16"]
    paged = (kc, vc)
    k_inv, v_inv = 1.0 / inp["k_scale"], 1.0 / inp["v_scale"]

    def call():
        if row.norm == 2:
            flashinfer.norm.rmsnorm(
                q.reshape(-1, HEAD_DIM), qw, out=norm_q.view(-1, HEAD_DIM)
            )
            flashinfer.norm.rmsnorm(
                k.reshape(-1, HEAD_DIM), kw, out=norm_k.view(-1, HEAD_DIM)
            )
            flashinfer.rope.apply_rope_pos_ids_inplace(
                norm_q, norm_k, inp["positions"], interleave=False
            )
            q_src, k_src = norm_q, norm_k
        elif row.norm == 1:
            tmp_q.copy_(q)
            tmp_k.copy_(k)
            flashinfer.rope.apply_rope_pos_ids_inplace(
                tmp_q, tmp_k, inp["positions"], interleave=False
            )
            flashinfer.norm.rmsnorm(
                tmp_q.view(-1, HEAD_DIM), qw, out=norm_q.view(-1, HEAD_DIM)
            )
            flashinfer.norm.rmsnorm(
                tmp_k.view(-1, HEAD_DIM), kw, out=norm_k.view(-1, HEAD_DIM)
            )
            q_src, k_src = norm_q, norm_k
        else:
            tmp_q.copy_(q)
            tmp_k.copy_(k)
            flashinfer.rope.apply_rope_pos_ids_inplace(
                tmp_q, tmp_k, inp["positions"], interleave=False
            )
            q_src, k_src = tmp_q, tmp_k
        qf = q_src.float()
        if row.quant == 1:
            scale = qf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / FP8_MAX
            oq.copy_((qf / scale).clamp(-FP8_MAX, FP8_MAX))
            if inp["is_prefill"]:
                qs.view(row.batch, row.hq, -1)[:, :, : row.q_len].copy_(
                    scale.view(row.batch, row.q_len, row.hq).permute(0, 2, 1)
                )
            else:
                qs.copy_(scale.view(T, row.hq))
        else:
            oq.copy_((qf * inp["q_scale_inv"]).clamp(-FP8_MAX, FP8_MAX))
        k_fp8.copy_((k_src.float() * k_inv).clamp(-FP8_MAX, FP8_MAX))
        v_fp8.copy_((v.float() * v_inv).clamp(-FP8_MAX, FP8_MAX))
        flashinfer.page.append_paged_kv_cache(
            k_fp8,
            v_fp8,
            inp["batch_indices"],
            inp["positions"],
            paged,
            inp["kv_indices"],
            inp["kv_indptr"],
            inp["kv_last_page_len"],
            kv_layout="NHD",
        )

    return call, bufs


def time_arm(call, *, warmup: int, iters: int):
    times = bench_gpu_time(
        call,
        dry_run_iters=warmup,
        repeat_iters=iters,
        cold_l2_cache=True,
        enable_cupti=True,
    )
    return float(statistics.median(times)) * 1e3  # ms -> us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(ROWS), help="comma list of row names")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--no-stable", action="store_true", help="skip the stable pipeline arm"
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    arms = ["cake"] + ([] if args.no_stable else ["stable"])
    results = []
    print(f"device {torch.cuda.get_device_name(device)}  arms {arms}")
    print(
        f"{'row':10s} {'cfg':28s} "
        + " ".join(f"{a:>10s}" for a in arms)
        + "   speedup vs others"
    )
    for name in args.rows.split(","):
        row = ROWS[name]
        inp = build_inputs(row, device)
        samples = {a: [] for a in arms}
        for _ in range(args.groups):
            for arm in arms:
                if arm == "cake":
                    call, _bufs = make_cake_call(inp, device)
                else:
                    call, _bufs = make_stable_call(inp, device)
                samples[arm].append(
                    time_arm(call, warmup=args.warmup, iters=args.iters)
                )
        med = {a: statistics.median(samples[a]) for a in arms}
        cfg = f"({row.hq},{row.hkv}) B{row.batch} Q{row.q_len} ctx{row.ctx} q{row.quant} n{row.norm} p{row.page_size}"
        ratios = " ".join(
            f"{a}/cake={med[a] / med['cake']:.3f}" for a in arms if a != "cake"
        )
        print(
            f"{name:10s} {cfg:28s} "
            + " ".join(f"{med[a]:10.2f}" for a in arms)
            + f"   {ratios}"
        )
        results.append(
            {"row": name, "config": cfg, "median_us": med, "samples_us": samples}
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "arms": arms,
                    "rows": results,
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
