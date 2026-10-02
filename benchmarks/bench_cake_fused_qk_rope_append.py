"""Benchmark the Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16) kernel.

Times the Cake route (``cake_fused_qk_rmsnorm_rope_append_paged_kv_cache``) against the stable
FlashInfer pipeline (``rmsnorm`` x2 + ``apply_rope_pos_ids_inplace`` + ``append_paged_kv_cache``) and,
optionally, a baseline fused implementation imported from ``--baseline-module`` (for example the
FlashInfer PR #5405 package ``flashinfer.experimental.fused_qk_rope_append`` when running from a checkout
that contains it).  All arms see the same inputs, the same BF16-rounded norm weights and their own
poison-filled caches / outputs.

Timing: ``flashinfer.testing.bench_gpu_time`` (CUPTI, cold L2) over the whole call (union of every
launch of one arm), eager and CUDA-graph replay, paired A/B/B/A groups.

Usage::

    python benchmarks/bench_cake_fused_qk_rope_append.py                       # default rows
    python benchmarks/bench_cake_fused_qk_rope_append.py --rows M1,M2,H1 --json out.json
    python benchmarks/bench_cake_fused_qk_rope_append.py --baseline-module flashinfer.experimental.fused_qk_rope_append
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from dataclasses import dataclass

import torch

import flashinfer
from flashinfer.cake_fused_qk_rope_append import (
    cake_fused_qk_rmsnorm_rope_append_paged_kv_cache,
)
from flashinfer.testing.utils import bench_gpu_time

HEAD_DIM = 128
POISON = 1234.0


@dataclass(frozen=True)
class Row:
    name: str
    hq: int
    hkv: int
    batch: int
    q_len: int
    ctx: int
    page_size: int = 64
    policy: int = 2


ROWS = {
    r.name: r
    for r in (
        Row("M1", 8, 1, 32, 1, 2048),
        Row("M1a", 8, 1, 32, 1, 2048, policy=0),
        Row("M1b", 8, 1, 32, 1, 2048, policy=1),
        Row("M2", 8, 1, 8, 16, 2048),
        Row("H1", 64, 8, 32, 1, 2048),
        Row("H2", 64, 8, 8, 16, 2048),
        Row("H3", 64, 8, 128, 1, 2048),
        Row("H4", 64, 8, 1, 128, 2048),
        Row("D1", 8, 1, 1, 1, 2048),
        Row("D3", 8, 1, 128, 1, 2048),
        Row("D4", 8, 1, 256, 1, 2048),
        Row("P1", 8, 1, 1, 128, 2048),
        Row("P2", 8, 1, 4, 64, 2048),
        Row("E1", 8, 1, 32, 1, 2048, page_size=16),
        Row("E2", 8, 1, 32, 1, 2048, page_size=128),
        Row("E3", 8, 1, 32, 1, 0),
        Row("E4", 8, 1, 8, 16, 2047),
    )
}


def rotary_cos_sin_table(max_positions: int, device, base: float = 10000.0) -> torch.Tensor:
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half))
    pos = torch.arange(max_positions, dtype=torch.float64, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)


def build_inputs(row: Row, device, seed: int = 892):
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
    qkv = torch.randn((T, width), generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    q_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16)
    k_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16)
    positions = torch.cat(
        [torch.arange(row.ctx, row.ctx + row.q_len, dtype=torch.int32) for _ in range(B)]
    )
    batch_indices = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), row.q_len)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(torch.tensor(pages_per_req, dtype=torch.int32), 0)
    kv_indices = torch.cat([page_indices[b, : pages_per_req[b]] for b in range(B)])
    kv_last_page_len = torch.tensor([(s - 1) % row.page_size + 1 for s in seq_lens], dtype=torch.int32)
    return {
        "row": row,
        "T": T,
        "qkv": qkv,
        "cos_sin": rotary_cos_sin_table(max(seq_lens) + 1, device),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "q_indptr": q_indptr.to(device),
        "page_indices": page_indices.to(device),
        "q_norm_weight": q_w.float().to(device),
        "k_norm_weight": k_w.float().to(device),
        "q_norm_weight_bf16": q_w.to(device),
        "k_norm_weight_bf16": k_w.to(device),
        "positions": positions.to(device),
        "batch_indices": batch_indices.to(device),
        "kv_indptr": kv_indptr.to(device),
        "kv_indices": kv_indices.to(device),
        "kv_last_page_len": kv_last_page_len.to(device),
        "total_pages": total_pages,
    }


def fresh_buffers(inp, device):
    row = inp["row"]
    kc = torch.full((inp["total_pages"], row.page_size, row.hkv, HEAD_DIM), POISON, dtype=torch.bfloat16, device=device)
    vc = torch.full_like(kc, POISON)
    oq = torch.full((inp["T"], row.hq, HEAD_DIM), POISON, dtype=torch.bfloat16, device=device)
    return kc, vc, oq


def make_cake_call(inp, device):
    row = inp["row"]
    kc, vc, oq = fresh_buffers(inp, device)
    policy = row.policy

    def call():
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            inp["qkv"], inp["cos_sin"], inp["seq_lens"], inp["q_indptr"], inp["page_indices"], kc, vc,
            num_q_heads=row.hq, num_kv_heads=row.hkv, qk_norm_policy=policy,
            q_norm_weight=inp["q_norm_weight"] if policy else None,
            k_norm_weight=inp["k_norm_weight"] if policy else None, out_q=oq,
        )

    return call, (kc, vc, oq)


def make_stable_call(inp, device):
    """rmsnorm(q) + rmsnorm(k) + apply_rope_pos_ids_inplace + append_paged_kv_cache (no last-page clear)."""
    row = inp["row"]
    kc, vc, oq = fresh_buffers(inp, device)
    T = inp["T"]
    q = inp["qkv"][:, : row.hq * HEAD_DIM].view(T, row.hq, HEAD_DIM)
    k = inp["qkv"][:, row.hq * HEAD_DIM : (row.hq + row.hkv) * HEAD_DIM].view(T, row.hkv, HEAD_DIM)
    v = inp["qkv"][:, (row.hq + row.hkv) * HEAD_DIM :].view(T, row.hkv, HEAD_DIM)
    norm_k = torch.empty_like(k)
    tmp_q = torch.empty_like(q)
    tmp_k = torch.empty_like(k)
    qw, kw = inp["q_norm_weight_bf16"], inp["k_norm_weight_bf16"]
    paged = (kc, vc)

    def call():
        if row.policy == 2:
            flashinfer.norm.rmsnorm(q.reshape(-1, HEAD_DIM), qw, out=oq.view(-1, HEAD_DIM))
            flashinfer.norm.rmsnorm(k.reshape(-1, HEAD_DIM), kw, out=norm_k.view(-1, HEAD_DIM))
            flashinfer.rope.apply_rope_pos_ids_inplace(oq, norm_k, inp["positions"], interleave=False)
            key_src = norm_k
        elif row.policy == 1:
            tmp_q.copy_(q)
            tmp_k.copy_(k)
            flashinfer.rope.apply_rope_pos_ids_inplace(tmp_q, tmp_k, inp["positions"], interleave=False)
            flashinfer.norm.rmsnorm(tmp_q.view(-1, HEAD_DIM), qw, out=oq.view(-1, HEAD_DIM))
            flashinfer.norm.rmsnorm(tmp_k.view(-1, HEAD_DIM), kw, out=norm_k.view(-1, HEAD_DIM))
            key_src = norm_k
        else:
            oq.copy_(q)
            tmp_k.copy_(k)
            flashinfer.rope.apply_rope_pos_ids_inplace(oq, tmp_k, inp["positions"], interleave=False)
            key_src = tmp_k
        flashinfer.page.append_paged_kv_cache(
            key_src, v, inp["batch_indices"], inp["positions"], paged, inp["kv_indices"], inp["kv_indptr"],
            inp["kv_last_page_len"], kv_layout="NHD",
        )

    return call, (kc, vc, oq)


def make_baseline_call(inp, device, module_name: str):
    """Fused baseline from an importable package exposing ``fused_qk_norm_rope_append_paged_kv_cache``."""
    mod = importlib.import_module(module_name)
    fn = getattr(mod, "fused_qk_norm_rope_append_paged_kv_cache", None)
    if fn is None:
        fn = importlib.import_module(module_name + ".backend").fused_qk_norm_rope_append_paged_kv_cache
    row = inp["row"]
    kc, vc, oq = fresh_buffers(inp, device)

    def call():
        fn(
            inp["qkv"], inp["cos_sin"], inp["seq_lens"], inp["q_indptr"], inp["page_indices"], kc, vc,
            num_q_heads=row.hq, num_kv_heads=row.hkv, qk_norm_policy=row.policy,
            q_norm_weight=inp["q_norm_weight"], k_norm_weight=inp["k_norm_weight"], out_q=oq,
        )

    return call, (kc, vc, oq)


def time_arm(call, *, graph: bool, warmup: int, iters: int):
    if graph:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                call()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                call()
        torch.cuda.synchronize()
        fn = g.replay
    else:
        fn = call
    times = bench_gpu_time(fn, dry_run_iters=warmup, repeat_iters=iters, cold_l2_cache=True, enable_cupti=True)
    return float(statistics.median(times)) * 1e3  # ms -> us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(ROWS), help="comma list of row names")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-stable", action="store_true", help="skip the stable pipeline arm")
    ap.add_argument("--baseline-module", default=None, help="importable fused baseline package (e.g. PR #5405)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    arms = ["cake"] + ([] if args.no_stable else ["stable"]) + (["baseline"] if args.baseline_module else [])
    results = []
    print(f"device {torch.cuda.get_device_name(device)} | arms {arms} | groups {args.groups} x iters {args.iters}")
    header = f"{'row':5s} {'geometry':30s} " + " ".join(f"{a + '-' + m:>14s}" for a in arms for m in ("graph", "eager"))
    print(header)
    for name in args.rows.split(","):
        row = ROWS[name]
        inp = build_inputs(row, device)
        calls = {}
        for arm in arms:
            if arm == "cake":
                calls[arm] = make_cake_call(inp, device)[0]
            elif arm == "stable":
                calls[arm] = make_stable_call(inp, device)[0]
            else:
                calls[arm] = make_baseline_call(inp, device, args.baseline_module)[0]
        samples = {(a, m): [] for a in arms for m in ("graph", "eager")}
        for g in range(args.groups):
            order = arms if g % 2 == 0 else list(reversed(arms))
            for arm in order:
                for mode in ("graph", "eager"):
                    samples[(arm, mode)].append(
                        time_arm(calls[arm], graph=(mode == "graph"), warmup=args.warmup, iters=args.iters)
                    )
        med = {k: statistics.median(v) for k, v in samples.items()}
        geometry = f"{row.hq}/{row.hkv} B{row.batch} Q{row.q_len} ctx{row.ctx} p{row.page_size} pol{row.policy}"
        print(f"{name:5s} {geometry:30s} " + " ".join(f"{med[(a, m)]:>11.2f} us" for a in arms for m in ("graph", "eager")))
        results.append({"row": name, "geometry": geometry, "medians_us": {f"{a}-{m}": med[(a, m)] for a, m in med}})
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": torch.cuda.get_device_name(device), "rows": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
