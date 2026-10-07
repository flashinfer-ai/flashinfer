# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired SM120 DeepSeek-V4.1 mixed-cache sparse-MLA decode benchmark: ``backend="sparse"`` vs ``backend="cake"``.

Both arms run through ``SparseMLASm120Wrapper(kv_cache_format="fp8",
kv_scale_format="ue8m0_g32", extra_kv_fp4=True, compute_precision=...)`` with
caller-owned split scratch over the same 528-byte FP8 main pool and 288-byte
V41_FP4 extra pool, indices, lengths and sink, so the comparison covers each
route's planner plus its stage and merge launches. Pools are HBM resident by
default (``--pool-mb`` per cache, far above the L2), indices are drawn without
replacement per token, the L2 is cold before every timed call and the call is
replayed from a CUDA graph; CUPTI timing is used when cupti-python is present
(``--no-cupti`` falls back to CUDA events).

Row sets follow the kernel's acceptance table (``--rows``):

* ``P`` shape sweep: H=64, main topk 128 + extra topk 512; (main page, extra
  page) = (64,64) x T in {1,8,32,64,128} and (32,32), (128,128), (32,64),
  (64,32), (128,64) x T in {8,32,128};
* ``H`` head counts: H in {8,16,32,128} x T in {8,32,128}, 128 + 512 @ (64,64);
* ``S`` SWA-only: H=64, main topk 128, no extra cache, page 64, T in {8,32,128};
* ``O`` operands: H=64, T=32, 128 + 512 @ (64,64) and (32,64) with ~70 % main
  lengths (one token empty), ragged extra lengths, ``-1`` tails, sinks and
  ``lse_scale`` 0.5.

Pass ``--num-tokens`` / ``--num-heads`` / ``--topk`` / ``--page-size`` /
``--extra-topk`` / ``--extra-page-size`` for a custom grid instead, and
``--bench-writers`` to time the cache quantize/pack writers separately.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

import flashinfer
from flashinfer.mla import (
    cake_sparse_mla_sm120_dsv41_mixed_num_chunks,
    dsv41_fp4_quantize_pack_sparse_mla_cache,
    dsv41_fp8_quantize_pack_sparse_mla_cache,
)
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm12x_supported

_D = 512
_MAIN_BYTES = 528
_EXTRA_BYTES = 288
_MIXED = dict(kv_cache_format="fp8", kv_scale_format="ue8m0_g32", extra_kv_fp4=True)
_ROW_SETS = ("P", "H", "S", "O")


@dataclasses.dataclass(frozen=True)
class Row:
    row_set: str
    num_tokens: int
    num_heads: int
    topk: int
    page_size: int
    extra_topk: int
    extra_page_size: int
    operands: bool = False  # ragged lengths, -1 tails, sink, lse_scale 0.5

    def label(self) -> str:
        extra = f"{self.extra_topk}@{self.extra_page_size}" if self.extra_topk else "-"
        return (
            f"{self.row_set},{self.num_tokens},{self.num_heads},{self.topk},"
            f"{self.page_size},{extra},{int(self.operands)}"
        )


def _acceptance_rows(sets: Iterable[str]) -> List[Row]:
    rows: List[Row] = []
    if "P" in sets:
        rows += [Row("P", t, 64, 128, 64, 512, 64) for t in (1, 8, 32, 64, 128)]
        for main_page, extra_page in (
            (32, 32),
            (128, 128),
            (32, 64),
            (64, 32),
            (128, 64),
        ):
            rows += [
                Row("P", t, 64, 128, main_page, 512, extra_page) for t in (8, 32, 128)
            ]
    if "H" in sets:
        rows += [
            Row("H", t, h, 128, 64, 512, 64)
            for h in (8, 16, 32, 128)
            for t in (8, 32, 128)
        ]
    if "S" in sets:
        rows += [Row("S", t, 64, 128, 64, 0, 64) for t in (8, 32, 128)]
    if "O" in sets:
        rows += [
            Row("O", 32, 64, 128, main_page, 512, 64, operands=True)
            for main_page in (64, 32)
        ]
    return rows


def _custom_rows(args: argparse.Namespace) -> List[Row]:
    rows: List[Row] = []
    for page_size in args.page_size:
        for num_tokens in args.num_tokens:
            for num_heads in args.num_heads:
                for topk in args.topk:
                    rows.append(
                        Row(
                            "C",
                            num_tokens,
                            num_heads,
                            topk,
                            page_size,
                            args.extra_topk,
                            args.extra_page_size,
                            operands=args.with_lengths_sink,
                        )
                    )
    return rows


def _median_us(fn, args: argparse.Namespace) -> float:
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn,
        dry_run_time_ms=args.warmup_ms,
        repeat_time_ms=args.measure_ms,
        enable_cupti=not args.no_cupti,
        use_cuda_graph=not args.no_cuda_graph,
        cold_l2_cache=True,
    )
    return float(np.median(measurements)) * 1e3


class _Pools:
    """One packed pool per (cache kind, page size); HBM resident by default."""

    def __init__(self, pool_mb: int, seed: int) -> None:
        self._pool_mb = pool_mb
        self._seed = seed
        self._pools: Dict[Tuple[str, int], torch.Tensor] = {}

    def _num_pages(self, page_size: int, bytes_per_token: int, min_slots: int) -> int:
        by_bytes = -(-(self._pool_mb << 20) // (page_size * bytes_per_token))
        by_slots = -(-min_slots // page_size)
        return max(by_bytes, by_slots, 1)

    def get(self, kind: str, page_size: int, min_slots: int) -> torch.Tensor:
        key = (kind, page_size)
        pool = self._pools.get(key)
        bytes_per_token = _MAIN_BYTES if kind == "main" else _EXTRA_BYTES
        need = self._num_pages(page_size, bytes_per_token, min_slots)
        if pool is not None and pool.shape[0] >= need:
            return pool
        writer = (
            dsv41_fp8_quantize_pack_sparse_mla_cache
            if kind == "main"
            else dsv41_fp4_quantize_pack_sparse_mla_cache
        )
        pool = torch.empty(
            need, 1, page_size, bytes_per_token, dtype=torch.uint8, device="cuda"
        )
        generator = torch.Generator(device="cuda").manual_seed(
            self._seed + len(self._pools)
        )
        chunk = max(
            1, (64 << 20) // (page_size * _D * 2)
        )  # ~64 MiB of BF16 latents per step
        for start in range(0, need, chunk):
            stop = min(need, start + chunk)
            latent = (
                torch.randn(
                    stop - start,
                    page_size,
                    _D,
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=generator,
                )
                / 10.0
            ).clamp(-1, 1)
            pool[start:stop] = writer(latent)
        self._pools[key] = pool
        return pool


def _indices_without_replacement(
    num_tokens: int, topk: int, num_slots: int
) -> torch.Tensor:
    rows = [torch.randperm(num_slots, device="cuda")[:topk] for _ in range(num_tokens)]
    return torch.stack(rows).to(torch.int32)


def _operands(
    row: Row, num_heads: int
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], float
]:
    if not row.operands:
        return None, None, None, 1.0
    lengths = torch.full(
        (row.num_tokens,), int(round(row.topk * 0.7)), dtype=torch.int32, device="cuda"
    )
    lengths[0] = 0
    extra_lengths = (
        torch.randint(
            row.extra_topk // 2,
            row.extra_topk + 1,
            (row.num_tokens,),
            dtype=torch.int32,
            device="cuda",
        )
        if row.extra_topk
        else None
    )
    sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")
    return lengths, extra_lengths, sink, 0.5


def _bench_writers(args: argparse.Namespace) -> None:
    print("writer,page,pages,median_us,GB_per_s_out")
    for page_size in sorted(set(args.page_size) | {64}):
        latent = (
            torch.randn(256, page_size, _D, dtype=torch.bfloat16, device="cuda") / 10.0
        ).clamp(-1, 1)
        for name, writer, bytes_per_token in (
            (
                "dsv41_fp8_quantize_pack",
                dsv41_fp8_quantize_pack_sparse_mla_cache,
                _MAIN_BYTES,
            ),
            (
                "dsv41_fp4_quantize_pack",
                dsv41_fp4_quantize_pack_sparse_mla_cache,
                _EXTRA_BYTES,
            ),
        ):
            median = _median_us(lambda: writer(latent), args)
            out_bytes = 256 * page_size * bytes_per_token
            print(f"{name},{page_size},256,{median:.2f},{out_bytes / median / 1e3:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="+", choices=_ROW_SETS, default=list(_ROW_SETS))
    parser.add_argument("--num-tokens", type=int, nargs="+", default=None)
    parser.add_argument("--num-heads", type=int, nargs="+", default=(64,))
    parser.add_argument("--topk", type=int, nargs="+", default=(128,))
    parser.add_argument("--page-size", type=int, nargs="+", default=(64,))
    parser.add_argument("--extra-topk", type=int, default=512)
    parser.add_argument("--extra-page-size", type=int, default=64)
    parser.add_argument("--with-lengths-sink", action="store_true")
    parser.add_argument(
        "--pool-mb",
        type=int,
        default=512,
        help="bytes per cache pool (HBM resident by default)",
    )
    parser.add_argument(
        "--precision", nargs="+", default=("bf16",), choices=("bf16", "fp8")
    )
    parser.add_argument("--backends", nargs="+", default=("sparse", "cake"))
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measure-ms", type=int, default=200)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-cupti", action="store_true")
    parser.add_argument("--bench-writers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not is_sm12x_supported(torch.device("cuda")):
        raise SystemExit("SM120/SM121 required")
    torch.manual_seed(args.seed)
    if args.bench_writers:
        _bench_writers(args)
    rows = (
        _custom_rows(args)
        if args.num_tokens is not None
        else _acceptance_rows(args.rows)
    )
    pools = _Pools(args.pool_mb, args.seed)
    sm_scale = _D**-0.5
    print(
        "set,tokens,heads,topk,page,extra,operands,precision,backend,median_us,speedup_vs_first"
    )
    for row in rows:
        cache = pools.get("main", row.page_size, row.topk)
        extra_cache = (
            pools.get("extra", row.extra_page_size, row.extra_topk)
            if row.extra_topk
            else None
        )
        q = (
            torch.randn(
                row.num_tokens, row.num_heads, _D, dtype=torch.bfloat16, device="cuda"
            )
            / 10.0
        ).clamp(-1, 1)
        indices = _indices_without_replacement(
            row.num_tokens, row.topk, cache.shape[0] * row.page_size
        )
        extra_indices = (
            _indices_without_replacement(
                row.num_tokens,
                row.extra_topk,
                extra_cache.shape[0] * row.extra_page_size,
            )
            if extra_cache is not None
            else None
        )
        lengths, extra_lengths, sink, lse_scale = _operands(row, row.num_heads)
        if row.operands:
            indices[:, row.topk - 8 :] = -1
        # Shared caller-owned scratch: the hand-written route needs ceil(topk/64) + ceil(extra/64)
        # splits, the Cake route at most its chunk count; both accept a larger split axis.
        splits = max(
            (row.topk + 63) // 64 + (row.extra_topk + 63) // 64,
            cake_sparse_mla_sm120_dsv41_mixed_num_chunks(row.topk, row.extra_topk),
        )
        mid_out = torch.empty(
            row.num_tokens,
            row.num_heads,
            splits,
            _D,
            dtype=torch.bfloat16,
            device="cuda",
        )
        mid_lse = torch.empty(
            row.num_tokens, row.num_heads, splits, dtype=torch.float32, device="cuda"
        )
        output = torch.empty_like(q)
        out_lse = torch.empty(
            row.num_tokens, row.num_heads, dtype=torch.float32, device="cuda"
        )
        for precision in args.precision:
            first = None
            for backend in args.backends:
                try:
                    runner = flashinfer.mla.SparseMLASm120Wrapper(
                        backend=backend,
                        compute_precision=precision,
                        device=q.device,
                        **_MIXED,
                    )

                    def run() -> None:
                        runner.run(
                            q,
                            cache,
                            indices,
                            output,
                            sm_scale,
                            topk_length=lengths,
                            attn_sink=sink,
                            extra_kv_cache=extra_cache,
                            extra_indices=extra_indices,
                            extra_topk_length=extra_lengths,
                            out_lse=out_lse,
                            mid_out=mid_out,
                            mid_lse=mid_lse,
                            lse_scale=lse_scale,
                        )

                    median = _median_us(run, args)
                except (
                    ValueError,
                    FileNotFoundError,
                ) as error:  # a shape outside one route's envelope, or kernels absent
                    print(
                        f"{row.label()},{precision},{backend},skip,{str(error).splitlines()[0]}"
                    )
                    continue
                first = median if first is None else first
                print(
                    f"{row.label()},{precision},{backend},{median:.2f},{first / median:.3f}"
                )


if __name__ == "__main__":
    main()
