"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Benchmark the experimental Cake NVFP4 paged-KV MSA decode on SM100/SM103/SM107.

Times ``flashinfer.msa_ops.prepare_msa_nvfp4_sparse_decode`` (the generated
persistent, short-item or request-packed union program) against
``msa_sparse_decode_attention`` on the same planar NVFP4 pages, top-k 16
selections and queries, with CUPTI and a cold L2 between iterations.  The rows
are the MiniMax-M3 decode matrix used to qualify the programs: tensor-parallel
ranks 1/2/4/8 (64/32/16/8 query heads over 4/2/1/1 KV heads), KV lengths from
a 257-token tail to one million tokens, one to eight query tokens per request;
plus the ``uch_*`` rows of the speculative-decode matrix (two to four query
tokens per request with a controlled fraction of shared top-k pages) that the
``use_cache_hit=True`` union program is qualified on.

``--use-cache-hit`` selects the arms of the Cake backend to time (repeatable;
default ``false``).  ``true`` applies to rows with ``seqlen_q`` in ``[2, 4]``
(the union program) and is a no-op at ``seqlen_q == 1``; other rows report it
as not applicable.  ``auto`` currently selects exactly the ``false`` programs.

Usage::

    python benchmarks/bench_cake_msa_nvfp4_decode.py [--rows tp1_b128_kv8k_q1 ...]
        [--use-cache-hit false true auto] [--json out.json]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.msa_ops import (  # noqa: E402
    msa_sparse_decode_attention,
    prepare_msa_nvfp4_sparse_decode,
)
from flashinfer.testing import bench_gpu_time_with_cupti  # noqa: E402
from tests.test_helpers.cake_msa_nvfp4_inputs import (  # noqa: E402
    build_decode_inputs,
    upstream_route_kwargs,
)

# name, batch, KV tokens per request, KV heads, query heads per KV head, seqlen_q, ragged,
# overlap fraction of the per-request selection draw (None: independent per token)
ROWS = [
    ("tp1_b128_kv8k_q1", 128, 8192, 4, 16, 1, False, None),
    ("tp1_b32_kv64k_q1", 32, 65536, 4, 16, 1, False, None),
    ("tp1_b32_kv64k_q1_ragged", 32, 65536, 4, 16, 1, True, None),
    ("tp1_b8_kv100k_q1", 8, 100_000, 4, 16, 1, False, None),
    ("tp1_b1_kv200k_q1", 1, 200_000, 4, 16, 1, False, None),
    ("tp2_b64_kv16k_q1", 64, 16384, 2, 16, 1, False, None),
    ("tp4_b128_kv8k_q1", 128, 8192, 1, 16, 1, False, None),
    ("tp4_b1_kv1m_q1", 1, 1_048_576, 1, 16, 1, False, None),
    ("tp8_b128_kv64k_q1", 128, 65536, 1, 8, 1, False, None),
    ("tp1_b2_kv257_q1_tail", 2, 257, 4, 16, 1, False, None),
    ("tp1_b32_kv8k_q2", 32, 8192, 4, 16, 2, False, None),
    ("tp1_b32_kv8k_q4", 32, 8192, 4, 16, 4, False, None),
    ("tp1_b32_kv8k_q8", 32, 8192, 4, 16, 8, False, None),
    ("tp4_b32_kv64k_q8", 32, 65536, 1, 16, 8, False, None),
    ("tp1_b8_kv100k_q8", 8, 100_000, 4, 16, 8, False, None),
    # Speculative-decode rows (use_cache_hit): tp2 is the primary serving topology.
    ("uch_tp2_q4_b1_kv150k_ov50", 1, 153600, 2, 16, 4, False, 0.5),
    ("uch_tp2_q4_b8_kv150k_ov0", 8, 153600, 2, 16, 4, False, 0.0),
    ("uch_tp2_q4_b8_kv150k_ov50", 8, 153600, 2, 16, 4, False, 0.5),
    ("uch_tp2_q4_b8_kv150k_ov100", 8, 153600, 2, 16, 4, False, 1.0),
    ("uch_tp2_q4_b32_kv150k_ov50", 32, 153600, 2, 16, 4, False, 0.5),
    ("uch_tp2_q4_b128_kv150k_ov50", 128, 153600, 2, 16, 4, False, 0.5),
    ("uch_tp2_q4_b8_kv8k_ov50", 8, 8192, 2, 16, 4, False, 0.5),
    ("uch_tp2_q4_b128_kv8k_ov50", 128, 8192, 2, 16, 4, False, 0.5),
    ("uch_tp1_q4_b8_kv150k_ov50", 8, 153600, 4, 16, 4, False, 0.5),
    ("uch_tp4_q4_b8_kv150k_ov50", 8, 153600, 1, 16, 4, False, 0.5),
    ("uch_tp2_q2_b32_kv8k_ov50", 32, 8192, 2, 16, 2, False, 0.5),
]
USE_CACHE_HIT_ARMS = {"false": False, "true": True, "auto": "auto"}


def ragged_lengths(batch, nominal, seqlen_q, seed):
    """Symmetric +-(5..25)% spread around ``nominal`` (pairs), shuffled."""
    g = torch.Generator().manual_seed(seed + batch * 1009 + nominal)
    pairs = batch // 2
    lo, hi = max(1, nominal // 20), max(1, nominal // 4)
    deltas = torch.randint(lo, hi + 1, (pairs,), generator=g, dtype=torch.int64)
    parts = [nominal - deltas, nominal + deltas]
    if batch % 2:
        parts.append(torch.tensor([nominal], dtype=torch.int64))
    values = torch.cat(parts)[torch.randperm(batch, generator=g)]
    return values.clamp(min=seqlen_q).tolist()


def median_us(fn):
    times = bench_gpu_time_with_cupti(
        fn, dry_run_iters=10, repeat_iters=50, cold_l2_cache=True
    )
    return statistics.median(times) * 1e3


def _prepare(inputs, out, use_cache_hit):
    return prepare_msa_nvfp4_sparse_decode(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        seqlen_q=inputs["seqlen_q"],
        softmax_scale=inputs["softmax_scale"],
        out=out,
        use_cache_hit=use_cache_hit,
    )


def bench_row(
    name,
    batch,
    kv,
    num_kv_heads,
    group,
    seqlen_q,
    ragged,
    overlap,
    *,
    arms,
    device,
    seed,
):
    seq_lens = (
        ragged_lengths(batch, kv, seqlen_q, seed)
        if ragged and batch > 1
        else [kv] * batch
    )
    extra = (
        {}
        if overlap is None
        else dict(q_overlap_fraction=overlap, page_table_order="identity")
    )
    inputs = build_decode_inputs(
        seq_lens,
        num_kv_heads=num_kv_heads,
        group_size=group,
        seqlen_q=seqlen_q,
        device=device,
        seed=seed,
        **extra,
    )
    out_route = torch.empty_like(inputs["q"])
    route_kwargs = upstream_route_kwargs(inputs)

    def route():
        msa_sparse_decode_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            out=out_route,
            **route_kwargs,
        )

    try:
        route()
        torch.cuda.synchronize()
        route_declined = None
    except NotImplementedError as exc:  # geometry outside the route's allowlist
        route_declined = str(exc).splitlines()[0]
    route_us = median_us(route) if route_declined is None else None

    items = batch * seqlen_q * num_kv_heads
    results = []
    for arm in arms:
        out_cake = torch.empty_like(inputs["q"])
        try:
            runner = _prepare(inputs, out_cake, USE_CACHE_HIT_ARMS[arm])
        except (ValueError, NotImplementedError) as exc:
            # ``true`` outside the union domain (seqlen_q > 4), or a part without the program.
            results.append(
                dict(
                    row=name,
                    use_cache_hit=arm,
                    batch=batch,
                    kv=kv,
                    num_kv_heads=num_kv_heads,
                    group=group,
                    seqlen_q=seqlen_q,
                    ragged=ragged,
                    overlap=overlap,
                    items=items,
                    route=None,
                    splits=None,
                    tail=None,
                    union_q=None,
                    cake_us=None,
                    route_us=route_us,
                    max_abs_diff_vs_route=None,
                    route_declined=route_declined,
                    not_applicable=str(exc).splitlines()[0],
                )
            )
            continue
        runner()
        torch.cuda.synchronize()
        cake_us = median_us(runner)
        max_abs = (
            float((out_cake.float() - out_route.float()).abs().max())
            if route_declined is None
            else None
        )
        results.append(
            dict(
                row=name,
                use_cache_hit=arm,
                batch=batch,
                kv=kv,
                num_kv_heads=num_kv_heads,
                group=group,
                seqlen_q=seqlen_q,
                ragged=ragged,
                overlap=overlap,
                items=items,
                route=runner.route,
                splits=runner.splits,
                tail=runner.tail,
                union_q=runner.union_q,
                cake_us=cake_us,
                route_us=route_us,
                max_abs_diff_vs_route=max_abs,
                route_declined=route_declined,
                not_applicable=None,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="*", default=None)
    parser.add_argument(
        "--use-cache-hit",
        nargs="+",
        choices=sorted(USE_CACHE_HIT_ARMS),
        default=["false"],
        help="Cake backend arms to time per row (default: false).",
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    device = torch.device("cuda")
    rows = ROWS if not args.rows else [r for r in ROWS if r[0] in set(args.rows)]
    arms = list(dict.fromkeys(args.use_cache_hit))
    results = []
    print(
        f"{'row':30s} {'uch':5s} {'items':>6s} {'route':>10s} {'splits':>6s} "
        f"{'cake_us':>9s} {'route_us':>9s} {'route/cake':>10s}"
    )
    for row in rows:
        for record in bench_row(*row, arms=arms, device=device, seed=args.seed):
            route_cell = (
                f"{record['route_us']:9.2f}"
                if record["route_us"] is not None
                else f"{'declined':>9s}"
            )
            if record["cake_us"] is None:
                print(
                    f"{record['row']:30s} {record['use_cache_hit']:5s} {record['items']:6d} "
                    f"{'n/a':>10s} {'-':>6s} {'-':>9s} {route_cell} {'-':>10s}"
                )
            else:
                ratio_cell = (
                    f"{record['route_us'] / record['cake_us']:10.3f}"
                    if record["route_us"] is not None
                    else f"{'-':>10s}"
                )
                print(
                    f"{record['row']:30s} {record['use_cache_hit']:5s} {record['items']:6d} "
                    f"{record['route']:>10s} {record['splits']:6d} "
                    f"{record['cake_us']:9.2f} {route_cell} {ratio_cell}"
                )
            results.append(record)
        torch.cuda.empty_cache()
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                dict(device=torch.cuda.get_device_name(device), rows=results), indent=1
            )
        )


if __name__ == "__main__":
    main()
