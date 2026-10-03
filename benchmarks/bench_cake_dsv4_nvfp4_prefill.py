# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark the CAKE DeepSeek-V4 NVFP4 (384-byte cache) sparse-MLA prefill on SM100/SM103.

Rows: H in {16, 32, 64, 128} x K in {128, 512} x T in {128, 512, 2048, 8192} x
{single main cache (page 64), dual main + extra cache (page 64, extra top-k = K)}
as a ragged two-request batch (request lengths linspace(T/2, T)), plus four
single-request H128 rows (K in {128, 512}, T in {2048, 8192}). Every arm is the
whole public ``trtllm_batch_decode_sparse_mla_dsv4`` call timed with CUPTI and a
cold L2 between iterations; the optional baseline arms map the same logical
BF16 Q / KV / selection / lengths / sink onto the dense-cache CAKE routes
(``--baselines cake-bf16 cake-fp8``) and TRTLLM-GEN (``trtllm-gen-bf16``,
``trtllm-gen-fp8``). The NVFP4 cache pack time is reported separately.

Usage::

    python benchmarks/bench_cake_dsv4_nvfp4_prefill.py [--rows REGEX] \\
        [--baselines cake-bf16 cake-fp8 trtllm-gen-bf16] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics

import torch

from flashinfer.mla import (
    get_cake_dsv4_workspace_bytes,
    nvfp4_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)
from flashinfer.testing import bench_gpu_time_with_cupti

HEAD_DIM = 512
SCALE = HEAD_DIM**-0.55
HEADS = (16, 32, 64, 128)
TOPKS = (128, 512)
TOKENS = (128, 512, 2048, 8192)
POOL_TOKENS = 131072  # KV pool tokens per request: the random top-k gathers miss L2
PAGE = 64
WORKSPACE_BYTES = 128 * 1024 * 1024
BASELINES = {
    "cake-bf16": (torch.bfloat16, "cake"),
    "cake-fp8": (torch.float8_e4m3fn, "cake"),
    "trtllm-gen-bf16": (torch.bfloat16, "trtllm-gen"),
    "trtllm-gen-fp8": (torch.float8_e4m3fn, "trtllm-gen"),
}


def rows():
    out = []
    for heads in HEADS:
        for topk in TOPKS:
            for tokens in TOKENS:
                for dual in (False, True):
                    out.append(
                        dict(
                            name=f"h{heads}-k{topk}-t{tokens}-{'dual' if dual else 'single'}",
                            heads=heads,
                            topk=topk,
                            tokens=tokens,
                            dual=dual,
                            batch=2,
                        )
                    )
    for topk in TOPKS:
        for tokens in (2048, 8192):
            out.append(
                dict(
                    name=f"h128-k{topk}-t{tokens}-single-seq1",
                    heads=128,
                    topk=topk,
                    tokens=tokens,
                    dual=False,
                    batch=1,
                )
            )
    return out


def q_lengths(batch: int, tokens: int):
    if batch == 1:
        return [tokens]
    lens = (
        torch.linspace(max(1, (tokens + 1) // 2), tokens, batch)
        .round()
        .to(torch.int32)
        .tolist()
    )
    lens[-1] += tokens - sum(lens)  # keep the row's token count exact
    return lens


def selection(gen, rows_: int, width: int, pool_tokens: int, device):
    """Random unique token ids inside a random active prefix, -1 past it."""
    lens = torch.randint(
        width // 2, width + 1, (rows_,), generator=gen, device=device
    ).to(torch.int32)
    ranks = torch.rand((rows_, width), generator=gen, device=device).argsort(dim=1)
    base = torch.randint(
        0, pool_tokens - width, (rows_, 1), generator=gen, device=device
    )
    table = (base + ranks).to(torch.int32)
    pos = torch.arange(width, device=device).unsqueeze(0)
    table[pos >= lens.unsqueeze(1)] = -1
    return table, lens


def make_inputs(row, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    q_lens = q_lengths(row["batch"], row["tokens"])
    tokens = sum(q_lens)
    heads, topk = row["heads"], row["topk"]
    pages = row["batch"] * POOL_TOKENS // PAGE
    main_latent = (
        torch.randn((pages, PAGE, HEAD_DIM), generator=gen, device=device) * 0.5
    ).to(torch.bfloat16)
    main_idx, main_lens = selection(gen, tokens, topk, pages * PAGE, device)
    inputs = dict(
        row=row,
        tokens=tokens,
        q_lens=q_lens,
        query=(
            torch.randn((tokens, heads, HEAD_DIM), generator=gen, device=device) * 0.6
        ).to(torch.bfloat16),
        main_latent=main_latent,
        main_idx=main_idx,
        main_lens=main_lens,
        extra_latent=None,
        extra_idx=None,
        extra_lens=None,
        sinks=(torch.randn((heads,), generator=gen, device=device) * 0.3).float(),
        seq_lens=torch.full(
            (row["batch"],), POOL_TOKENS, dtype=torch.int32, device=device
        ),
        cum_seq_lens_q=torch.tensor(
            [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
            dtype=torch.int32,
            device=device,
        ),
        max_q_len=max(q_lens),
        workspace=torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device),
    )
    if row["dual"]:
        inputs["extra_latent"] = (
            torch.randn((pages, PAGE, HEAD_DIM), generator=gen, device=device) * 0.5
            + 0.05
        ).to(torch.bfloat16)
        inputs["extra_idx"], inputs["extra_lens"] = selection(
            gen, tokens, topk, pages * PAGE, device
        )
    return inputs


def median_ms(fn, repeat_time_ms: int):
    return float(
        statistics.median(
            bench_gpu_time_with_cupti(
                fn, cold_l2_cache=True, repeat_time_ms=repeat_time_ms
            )
        )
    )


def nvfp4_call(inputs):
    main_cache = nvfp4_quantize_pack_sparse_mla_cache(inputs["main_latent"])
    extra_cache = (
        nvfp4_quantize_pack_sparse_mla_cache(inputs["extra_latent"])
        if inputs["extra_latent"] is not None
        else None
    )
    out = torch.empty_like(inputs["query"])

    def call():
        return trtllm_batch_decode_sparse_mla_dsv4(
            inputs["query"],
            main_cache,
            inputs["workspace"],
            inputs["main_idx"],
            compressed_kv_cache=extra_cache,
            swa_topk_lens=inputs["main_lens"],
            extra_sparse_indices=inputs["extra_idx"],
            extra_sparse_topk_lens=inputs["extra_lens"],
            seq_lens=inputs["seq_lens"],
            out=out,
            bmm1_scale=SCALE,
            bmm2_scale=1.0,
            sinks=inputs["sinks"],
            cum_seq_lens_q=inputs["cum_seq_lens_q"],
            max_q_len=inputs["max_q_len"],
            enable_pdl=False,
            backend="cake",
            kv_cache_format="nvfp4",
        )

    def pack():
        nvfp4_quantize_pack_sparse_mla_cache(inputs["main_latent"])

    return (
        call,
        pack,
        main_cache.numel() + (extra_cache.numel() if extra_cache is not None else 0),
    )


def baseline_call(inputs, dtype, backend):
    """Dense-cache arms on the same logical problem: one merged pool, one combined table,
    per-row lengths folded into -1 masks (the 128 window slots must stay valid)."""
    device = inputs["query"].device
    main = inputs["main_latent"]
    merged = (
        main
        if inputs["extra_latent"] is None
        else torch.cat((main, inputs["extra_latent"]), dim=0)
    )
    cache = merged.to(dtype).unsqueeze(1).contiguous()
    main_idx = inputs["main_idx"].clone()
    if bool((inputs["main_lens"] < 128).any()):
        raise ValueError("baseline ABI: the 128 window slots must be valid")
    segments = [main_idx]
    if inputs["extra_idx"] is not None:
        extra = inputs["extra_idx"].clone()
        extra[extra >= 0] += main.shape[0] * PAGE
        segments.append(extra)
    combined = torch.cat(segments, dim=1).contiguous()
    valid = combined >= 0
    width = combined.shape[1]
    last = torch.where(
        valid.any(dim=1),
        width - 1 - valid.flip(1).to(torch.int32).argmax(dim=1),
        torch.zeros((), dtype=torch.long, device=device),
    )
    lens = torch.maximum(last + 1, torch.full_like(last, 128)).to(torch.int32)
    query = inputs["query"].to(dtype)
    out = torch.empty(inputs["query"].shape, dtype=torch.bfloat16, device=device)
    if dtype == torch.float8_e4m3fn:
        bmm1 = torch.tensor([SCALE], dtype=torch.float32, device=device)
        bmm2 = torch.tensor([1.0], dtype=torch.float32, device=device)
    else:
        bmm1, bmm2 = SCALE, 1.0
    workspace = inputs["workspace"]
    if backend == "cake":
        needed = get_cake_dsv4_workspace_bytes(
            inputs["tokens"], inputs["query"].shape[1], width, dtype
        )
        if needed > workspace.numel():
            workspace = torch.zeros(needed, dtype=torch.uint8, device=device)

    def call():
        return trtllm_batch_decode_sparse_mla_dsv4(
            query,
            cache,
            workspace,
            combined,
            compressed_kv_cache=cache,
            sparse_topk_lens=lens,
            seq_lens=inputs["seq_lens"],
            out=out,
            bmm1_scale=bmm1,
            bmm2_scale=bmm2,
            sinks=inputs["sinks"],
            cum_seq_lens_q=inputs["cum_seq_lens_q"],
            max_q_len=inputs["max_q_len"],
            enable_pdl=False,
            backend=backend,
        )

    return call


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", default=".*", help="regex over row names")
    parser.add_argument("--baselines", nargs="*", default=[], choices=sorted(BASELINES))
    parser.add_argument(
        "--repeat-ms",
        type=int,
        default=300,
        help="CUPTI measurement window per arm (ms)",
    )
    parser.add_argument(
        "--json", default=None, help="write per-row results to this file"
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability(device)
    if major != 10:
        raise SystemExit("the CAKE DSv4 NVFP4 route requires SM100/SM103")
    selected = [row for row in rows() if re.search(args.rows, row["name"])]
    arms = ["cake-nvfp4", *args.baselines]
    print(
        f"{'row':34s} {'tokens':>6s} "
        + " ".join(f"{arm:>16s}" for arm in arms)
        + f" {'pack_ms':>8s} {'cache_MiB':>9s}"
    )
    results = []
    for row in selected:
        inputs = make_inputs(row, device)
        call, pack, cache_bytes = nvfp4_call(inputs)
        call()
        torch.cuda.synchronize()
        record = dict(
            row=row["name"], tokens=inputs["tokens"], cache_bytes=cache_bytes, arms={}
        )
        record["arms"]["cake-nvfp4"] = median_ms(call, args.repeat_ms)
        record["pack_ms"] = median_ms(pack, args.repeat_ms)
        for arm in args.baselines:
            dtype, backend = BASELINES[arm]
            try:
                fn = baseline_call(inputs, dtype, backend)
                fn()
                torch.cuda.synchronize()
                record["arms"][arm] = median_ms(fn, args.repeat_ms)
            except Exception as exc:  # the baseline route does not admit this row
                record["arms"][arm] = None
                record.setdefault("refused", {})[arm] = f"{type(exc).__name__}: {exc}"[
                    :200
                ]
        cells = []
        for arm in arms:
            ms = record["arms"][arm]
            if ms is None:
                cells.append(f"{'refused':>16s}")
            elif arm == "cake-nvfp4":
                cells.append(f"{ms * 1000:13.1f} us")
            else:
                cells.append(
                    f"{ms * 1000:9.1f} us {ms / record['arms']['cake-nvfp4']:4.2f}x"
                )
        print(
            f"{row['name']:34s} {inputs['tokens']:6d} "
            + " ".join(cells)
            + f" {record['pack_ms'] * 1000:8.1f} {cache_bytes / 2**20:9.1f}"
        )
        results.append(record)
        del inputs
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                dict(device=torch.cuda.get_device_name(device), rows=results),
                handle,
                indent=1,
            )


if __name__ == "__main__":
    main()
