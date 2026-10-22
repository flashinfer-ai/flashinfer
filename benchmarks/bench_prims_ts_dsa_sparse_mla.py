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

"""Benchmark the PrimTS 1CTA DSA (token-sparse) MLA decode kernels.

GLM-5.3 / DeepSeek-V3.2 decode shapes: ``T = batch * q_len`` packed query
tokens, 64 (or 128) heads, QK dim 576 / V dim 512, each token attending to
``top_k`` rows of a flat KV pool.  Requests are laid out like a serving
engine's paged cache: request ``b`` owns ``rows_per_request`` consecutive rows
and every one of its tokens routes to ``top_k`` random rows of that span, so
decode (``q_len=1``) is a cold HBM gather and prefill-shaped runs (large
``q_len``) mostly hit L2.

Times are cold-L2 CUPTI medians of CUDA-graph replays; ``--baseline`` adds
``trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=...)`` on the E4M3
cache built from the same rows.  Run from the repository root:

    python benchmarks/bench_prims_ts_dsa_sparse_mla.py --kernel nvfp4 --batch 16 128 256
    python benchmarks/bench_prims_ts_dsa_sparse_mla.py --kernel fp8 --baseline --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_helpers import dsa_nvfp4_cache_reference as ref  # noqa: E402

import flashinfer  # noqa: E402
from flashinfer.attention.prims_ts import (  # noqa: E402
    prims_ts_dsa_sparse_mla_fp8,
    prims_ts_dsa_sparse_mla_nvfp4,
)
from flashinfer.testing import bench_gpu_time  # noqa: E402

HEAD_DIM, V_DIM = 576, 512
FP8_ROW_BYTES, NVFP4_ROW_BYTES = 576, 320 + 40
HBM_BYTES_PER_S = 8.0e12  # B200 nominal


def request_shaped_indices(
    batch: int,
    q_len: int,
    top_k: int,
    rows_per_request: int,
    min_context: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """``[batch * q_len, top_k]`` INT32 routes; request ``b`` owns rows ``[b * R, (b + 1) * R)``.

    Each request draws a context length in ``[min_context, R]``; tokens of a
    request with fewer than ``top_k`` rows keep ``-1`` in the unused slots.
    """
    total_q = batch * q_len
    out = torch.full((total_q, top_k), -1, dtype=torch.int32, device=device)
    ctx = torch.randint(
        min_context, rows_per_request + 1, (batch,), generator=gen, device=device
    )
    for b in range(batch):
        c = int(ctx[b])
        k = min(top_k, c)
        base = b * rows_per_request
        for j in range(q_len):
            perm = torch.randperm(c, generator=gen, device=device)[:k]
            out[b * q_len + j, :k] = (perm + base).to(torch.int32)
    return out


def roofline_us_per_token(heads: int, top_k: int, kv_row_bytes: int) -> float:
    """HBM time per query token at 8 TB/s: routed rows + Q + BF16 output + indices."""
    bytes_per_token = (
        top_k * kv_row_bytes + heads * HEAD_DIM + heads * V_DIM * 2 + top_k * 4
    )
    return bytes_per_token / HBM_BYTES_PER_S * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kernel", choices=["fp8", "nvfp4"], default="nvfp4")
    parser.add_argument("--heads", type=int, nargs="+", default=[64])
    parser.add_argument("--top-k", type=int, nargs="+", default=[2048])
    parser.add_argument("--q-len", type=int, nargs="+", default=[1])
    parser.add_argument("--batch", type=int, nargs="+", default=[16, 128, 256])
    parser.add_argument(
        "--pool-gb",
        type=float,
        default=1.5,
        help="E4M3 pool size the rows are drawn from",
    )
    parser.add_argument("--rows-per-request", type=int, default=8192)
    parser.add_argument("--min-context", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--skip-corr",
        type=float,
        default=0.0,
        help="skip-correction threshold in log2 units (0 = exact rescale, 8 = max for E4M3 P)",
    )
    parser.add_argument(
        "--amax", type=float, default=168.0, help="TRT-LLM amax of the NVFP4 cache"
    )
    parser.add_argument(
        "--baseline", action="store_true", help="also time trtllm-gen FP8 sparse MLA"
    )
    parser.add_argument(
        "--refcheck",
        action="store_true",
        help="compare the first shape against the PyTorch reference before timing",
    )
    parser.add_argument(
        "--csv", default="", help="append one row per shape to this file"
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(
        f"GPU {props.name} torch {torch.__version__} flashinfer {flashinfer.__version__}"
    )
    gen = torch.Generator(device=device).manual_seed(0)
    rows = int(args.pool_gb * (1 << 30)) // HEAD_DIM // 64 * 64
    if rows < max(args.batch) * args.rows_per_request:
        raise SystemExit(
            f"--pool-gb {args.pool_gb} holds {rows} rows; need "
            f"{max(args.batch) * args.rows_per_request} for the largest batch"
        )
    kv = (torch.randn(rows, HEAD_DIM, device=device, generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    sm_scale = 1.0 / 16.0
    workspace = (
        torch.zeros(512 << 20, dtype=torch.int8, device=device)
        if args.baseline
        else None
    )

    if args.kernel == "fp8":
        kv_row_bytes = FP8_ROW_BYTES

        def reference_rows():
            return kv.float()

        def run_ours(q, idx, lens, out, top_k):
            prims_ts_dsa_sparse_mla_fp8(
                q,
                kv,
                idx,
                None,
                None,
                sparse_mla_top_k=top_k,
                sparse_mla_top_k_lens=lens,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                skip_corr_threshold=args.skip_corr,
                out=out,
            )
    else:
        kv_row_bytes = NVFP4_ROW_BYTES
        data_pool = torch.empty(rows, 320, dtype=torch.uint8, device=device)
        scale_pool = torch.empty(rows, 40, dtype=torch.float8_e4m3fn, device=device)
        t0 = time.time()
        for r0 in range(0, rows, 1 << 18):
            d, sc = ref.quantize_rows(
                kv[r0 : r0 + (1 << 18)].float(), 64, amax=args.amax
            )
            data_pool[r0 : r0 + d.shape[0]] = d
            scale_pool[r0 : r0 + d.shape[0]] = sc
        torch.cuda.synchronize()
        print(
            f"quantized {rows} rows to the TRT-LLM NVFP4 layout in {time.time() - t0:.1f}s"
        )
        dequant_scale = ref.dequant_scale(args.amax)

        def reference_rows():
            return ref.dequantize_rows(data_pool, scale_pool, 64, amax=args.amax)

        def run_ours(q, idx, lens, out, top_k):
            prims_ts_dsa_sparse_mla_nvfp4(
                q,
                data_pool,
                scale_pool,
                idx,
                None,
                None,
                sparse_mla_top_k=top_k,
                sparse_mla_top_k_lens=lens,
                kv_global_dequant_scale=dequant_scale,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                skip_corr_threshold=args.skip_corr,
                out=out,
            )

    fields = [
        "tag", "kernel", "heads", "top_k", "q_len", "batch", "total_q",
        "ours_us", "ours_warm_us", "trtllm_fp8_us", "speedup_vs_trtllm",
        "ours_us_per_token", "roofline_us_per_token",
    ]  # fmt: skip
    results = []
    checked = False
    for heads in args.heads:
        for top_k in args.top_k:
            for q_len in args.q_len:
                for batch in args.batch:
                    total_q = batch * q_len
                    idx = request_shaped_indices(
                        batch,
                        q_len,
                        top_k,
                        args.rows_per_request,
                        args.min_context,
                        gen,
                        device,
                    )
                    lens = (idx >= 0).sum(dim=1).to(torch.int32)
                    q = (
                        torch.randn(
                            total_q, heads, HEAD_DIM, device=device, generator=gen
                        )
                        * 0.5
                    ).to(torch.float8_e4m3fn)
                    out = torch.empty(
                        total_q, heads, V_DIM, dtype=torch.bfloat16, device=device
                    )

                    def ours():
                        run_ours(q, idx, lens, out, top_k)

                    def trtllm():
                        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
                            query=q.unsqueeze(1),
                            kv_cache=kv.view(-1, 1, 64, HEAD_DIM),
                            workspace_buffer=workspace,
                            qk_nope_head_dim=192,
                            kv_lora_rank=512,
                            qk_rope_head_dim=64,
                            block_tables=idx.unsqueeze(1),
                            seq_lens=lens,
                            max_seq_len=top_k,
                            sparse_mla_top_k=top_k,
                            bmm1_scale=sm_scale,
                            bmm2_scale=1.0,
                        )

                    ours()
                    torch.cuda.synchronize()
                    if args.refcheck and not checked:
                        # NVFP4: the kernel quantizes Q to NVFP4 on chip, so the
                        # reference sees the same Q image.
                        q_ref = ref.quantize_q_nvfp4(q) if args.kernel == "nvfp4" else q
                        expected, _ = ref.sparse_mla_reference(
                            q_ref,
                            reference_rows(),
                            idx,
                            bmm1_scale=sm_scale,
                            bmm2_scale=1.0,
                            top_k_lens=lens,
                        )
                        rel = (
                            (out.float() - expected.float()).norm()
                            / expected.float().norm()
                        ).item()
                        print(
                            f"refcheck: rel_l2 {rel:.3e} vs the FP32 reference over dequantized rows"
                        )
                        checked = True
                    cold = (
                        statistics.median(
                            bench_gpu_time(
                                ours,
                                use_cuda_graph=True,
                                cold_l2_cache=True,
                                repeat_iters=args.repeat,
                            )
                        )
                        * 1e3
                    )
                    warm = (
                        statistics.median(
                            bench_gpu_time(
                                ours,
                                use_cuda_graph=True,
                                cold_l2_cache=False,
                                repeat_iters=args.repeat,
                            )
                        )
                        * 1e3
                    )
                    base = float("nan")
                    if args.baseline:
                        trtllm()
                        torch.cuda.synchronize()
                        base = (
                            statistics.median(
                                bench_gpu_time(
                                    trtllm,
                                    use_cuda_graph=True,
                                    cold_l2_cache=True,
                                    repeat_iters=args.repeat,
                                )
                            )
                            * 1e3
                        )
                    row = dict(
                        tag=args.tag, kernel=args.kernel, heads=heads, top_k=top_k, q_len=q_len,
                        batch=batch, total_q=total_q, ours_us=round(cold, 2), ours_warm_us=round(warm, 2),
                        trtllm_fp8_us=round(base, 2), speedup_vs_trtllm=round(base / cold, 3),
                        ours_us_per_token=round(cold / total_q, 4),
                        roofline_us_per_token=round(roofline_us_per_token(heads, top_k, kv_row_bytes), 4),
                    )  # fmt: skip
                    results.append(row)
                    print(
                        " ".join(f"{k}={v}" for k, v in row.items() if k != "tag"),
                        flush=True,
                    )
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerows(results)
        print(f"appended {len(results)} rows to {args.csv}")


if __name__ == "__main__":
    main()
