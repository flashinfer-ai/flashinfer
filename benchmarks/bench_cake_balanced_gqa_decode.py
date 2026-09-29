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

"""Benchmark the balanced paged GQA decode families on SM100/SM103.

Ragged and uniform decode batches timed with CUPTI and a cold L2 between
iterations, against ``trtllm_batch_decode_with_kv_cache(backend="trtllm-gen")``
on the same tensors (``--with-trtllm``).  The AgentX rows reproduce the ragged
KV pattern of flashinfer-ai/flashinfer#4832.  ``--production`` also times the
production ``backend="cake"`` route (``cake_batch_decode_with_kv_cache`` with a
caller-provided zero-initialized counter buffer, eager and CUDA-graph replay);
with ``--with-trtllm`` the trtllm-gen arm then receives the same buffer, so
neither arm pays a per-call zero-fill on the stream.

Usage::

    python benchmarks/bench_cake_balanced_gqa_decode.py [--with-trtllm] [--production] [--rows agentx_b16 ...]
    python benchmarks/bench_cake_balanced_gqa_decode.py --dtype fp16 --with-trtllm [--rows ...]
    python benchmarks/bench_cake_balanced_gqa_decode.py --dtype fp8|bf16q|fp16q --with-trtllm
    python benchmarks/bench_cake_balanced_gqa_decode.py --head-dim 64 --with-trtllm
    python benchmarks/bench_cake_balanced_gqa_decode.py --head-dim 256 --page-size 32 --with-trtllm

``--dtype`` selects the Q/K/V/O dtypes: ``bf16`` and ``fp16`` (Q/K/V/O in that
dtype), ``fp8`` (E4M3 Q/K/V/O), ``bf16q`` / ``fp16q`` (BF16 / FP16 Q and O over
an E4M3 K/V cache).  The E4M3 tensors are quantized per tensor with amax scales
and the route receives ``bmm1_scale = q_scale * k_scale / sqrt(head_dim)`` and
``bmm2_scale = v_scale`` (output scale 1.0); the fp8 output is allocated as
``float8_e4m3fn``.  ``--head-dim {64,128,256}`` (default 128) selects the BF16
head_dim-64 (1..16 query heads per KV head) and head_dim-256 (per-row tiles,
uniform q_len 1..8) families; ``--page-size {16,32,64}`` (default 16) sizes the
KV pages and block tables and only head_dim 256 accepts 32 / 64.

The experimental ``balanced_gqa_decode`` API is BF16 / head_dim 128 only; for
every other family the ``balanced`` column is the production ``backend="cake"``
route and ``--production`` is implied.  The default row set follows the family
(``--rows`` overrides it): the q_len-1 GQA-8 rows for fp8 / bf16q / fp16q, plus
the ``gqa16_*`` rows for head_dim 64 and the ``mtp4`` / ``mtp8`` rows for
head_dim 256.  ``route`` names the Cake component the production route selected.
"""

import argparse
import json
import math

import torch

from flashinfer.cake_fmha import (
    cake_batch_decode_with_kv_cache,
    cake_fmha_balanced_counter_bytes,
    select_cake_fmha_decode_route,
)
from flashinfer.decode import (
    prepare_balanced_batch_decode_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)
from flashinfer.experimental.balanced_gqa_decode.cake_backend import (
    balanced_gqa_decode_workspace_size,
)
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.utils import (
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)

AGENTX = [
    8193,
    57345,
    73729,
    81921,
    98305,
    106497,
    114689,
    131073,
    139265,
    147457,
    163841,
    180225,
    196609,
    212993,
    229377,
    237569,
]

E4M3_MAX = 448.0

# (query / output dtype, K / V cache dtype)
DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp16": (torch.float16, torch.float16),
    "fp8": (torch.float8_e4m3fn, torch.float8_e4m3fn),
    "bf16q": (torch.bfloat16, torch.float8_e4m3fn),
    "fp16q": (torch.float16, torch.float8_e4m3fn),
}


def _agentx(batch):
    return [AGENTX[i % len(AGENTX)] for i in range(batch)]


def _random_lengths(batch, lo, hi, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi + 1, (batch,), generator=gen).tolist()


# row -> (seq_lens, num_kv_heads, q_len, query heads per KV head)
ROWS = {
    "agentx_b16_hkv1": (_agentx(16), 1, 1, 8),
    "agentx_b32_hkv1": (_agentx(32), 1, 1, 8),
    "agentx_b64_hkv1": (_agentx(64), 1, 1, 8),
    "agentx_b128_hkv1": (_agentx(128), 1, 1, 8),
    "agentx_b192_hkv1": (_agentx(192), 1, 1, 8),
    "agentx_b256_hkv1": (_agentx(256), 1, 1, 8),
    "random_128_128k_b256_hkv1": (_random_lengths(256, 128, 131072, 1), 1, 1, 8),
    "random_128_65k_b64_hkv8": (_random_lengths(64, 128, 65536, 2), 8, 1, 8),
    "uniform_b4_hkv8_s65536": ([65536] * 4, 8, 1, 8),
    "uniform_b16_hkv8_s65536": ([65536] * 16, 8, 1, 8),
    "uniform_b64_hkv8_s32768": ([32768] * 64, 8, 1, 8),
    "uniform_b128_hkv8_s32768": ([32768] * 128, 8, 1, 8),
    "uniform_b128_hkv8_s4096": ([4096] * 128, 8, 1, 8),
    "agentx_b16_hkv1_mtp7": (_agentx(16), 1, 7, 8),
    "uniform_b8_hkv8_s4096_mtp3": ([4096] * 8, 8, 3, 8),
    "uniform_b8_hkv8_s4096_mtp7": ([4096] * 8, 8, 7, 8),
    "uniform_b1_hkv1_s60007_mtp7": ([60007], 1, 7, 8),
    "ragged_b4_hkv2_mtp7": ([300, 257, 5000, 777], 2, 7, 8),
}
HD128_ROWS = list(ROWS)
Q1_GQA8_ROWS = [
    name for name, (_, _, q_len, group) in ROWS.items() if q_len == 1 and group == 8
]
HD64_EXTRA_ROWS = {
    "gqa16_agentx_b64_hkv1": (_agentx(64), 1, 1, 16),
    "gqa16_uniform_b64_hkv2_s16384": ([16384] * 64, 2, 1, 16),
}
HD256_EXTRA_ROWS = {
    "mtp4_ragged_b16_hkv1": (_agentx(16), 1, 4, 8),
    "mtp8_uniform_b8_hkv4_s4096": ([4096] * 8, 4, 8, 8),
}
ROWS.update(HD64_EXTRA_ROWS)
ROWS.update(HD256_EXTRA_ROWS)


def default_rows(dtype, head_dim):
    """The per-family row set of the Cake design document."""

    if head_dim == 64:
        return Q1_GQA8_ROWS + list(HD64_EXTRA_ROWS)
    if head_dim == 256:
        return Q1_GQA8_ROWS + list(HD256_EXTRA_ROWS)
    if dtype in ("fp8", "bf16q", "fp16q"):
        return Q1_GQA8_ROWS
    return HD128_ROWS


def _quantize_e4m3(values):
    """Per-tensor E4M3 quantization with an amax scale: ``(quantized, scale)``."""

    scale = float(values.abs().amax()) / E4M3_MAX
    quantized = (values / scale).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return quantized, scale


def make_inputs(
    seq_lens,
    num_kv_heads,
    q_len,
    device,
    seed=0,
    dtype=torch.bfloat16,
    *,
    kv_dtype=None,
    head_dim=128,
    page_size=16,
    group=8,
):
    """Random query and paged HND K / V caches with shuffled block tables.

    E4M3 tensors are quantized per tensor with amax scales.  Returns
    ``(query, k_cache, v_cache, block_tables, seq_lens, (q_scale, k_scale, v_scale))``
    with unit scales for the non-quantized tensors.
    """

    kv_dtype = dtype if kv_dtype is None else kv_dtype
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(seq_lens)
    num_q_heads = group * num_kv_heads
    max_pages = (max(seq_lens) + page_size - 1) // page_size
    max_pages = (max_pages + 7) // 8 * 8
    num_pages = batch * max_pages

    def random_tensor(shape, target_dtype):
        values = torch.randn(shape, generator=gen, device=device)
        if target_dtype == torch.float8_e4m3fn:
            return _quantize_e4m3(values)
        return values.to(target_dtype), 1.0

    query, q_scale = random_tensor((batch * q_len, num_q_heads, head_dim), dtype)
    k_cache, k_scale = random_tensor(
        (num_pages, num_kv_heads, page_size, head_dim), kv_dtype
    )
    v_cache, v_scale = random_tensor(
        (num_pages, num_kv_heads, page_size, head_dim), kv_dtype
    )
    block_tables = (
        torch.randperm(num_pages, generator=gen, device=device)
        .to(torch.int32)
        .view(batch, max_pages)
    )
    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    scales = (q_scale, k_scale, v_scale)
    return query, k_cache, v_cache, block_tables, seq_lens_dev, scales


def _flops(seq_lens, num_q_heads, q_len, head_dim):
    tokens = sum(s - (q_len - 1 - j) for s in seq_lens for j in range(q_len))
    return 4.0 * tokens * num_q_heads * head_dim


def _bytes(seq_lens, num_kv_heads, num_q_heads, q_len, head_dim, q_dtype, kv_dtype):
    tokens = sum(seq_lens)
    kv = 2 * tokens * num_kv_heads * head_dim * kv_dtype.itemsize
    q = len(seq_lens) * q_len * num_q_heads * head_dim * q_dtype.itemsize
    return kv + 2 * q  # query read plus output write in the query dtype


def _median_ms(fn, use_cuda_graph=False):
    times = bench_gpu_time_with_cupti(
        fn, cold_l2_cache=True, use_cuda_graph=use_cuda_graph
    )
    times = sorted(times)
    return float(times[len(times) // 2])


def _cake_component(
    query,
    kv_cache,
    out,
    workspace,
    block_tables,
    seq_lens_dev,
    *,
    q_len,
    max_seq_len,
    bmm1_scale,
    bmm2_scale,
    counter_buffer,
):
    """Name of the Cake component the production route selects for the row."""

    route = select_cake_fmha_decode_route(
        query.device,
        query=query,
        key_cache=kv_cache[:, 0],
        value_cache=kv_cache[:, 1],
        out=out,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens_dev,
        batch_size=int(seq_lens_dev.numel()),
        q_len=q_len,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        o_scale=1.0,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
        multi_ctas_kv_counter_buffer=counter_buffer,
    )
    return "compat_v1" if route is None else route.component


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows",
        nargs="*",
        default=None,
        help="row names (default: the family's row set, see the module docstring)",
    )
    parser.add_argument("--with-trtllm", action="store_true")
    parser.add_argument(
        "--production",
        action="store_true",
        help="also time the production backend='cake' route (eager + graph replay)",
    )
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bf16",
        help="Q/K/V/O dtypes; every family but bf16 times the production route "
        "as the balanced column",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        choices=(64, 128, 256),
        default=128,
        help="head dimension (64 and 256 are BF16 families)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        choices=(16, 32, 64),
        default=16,
        help="tokens per KV page (32 / 64 only with --head-dim 256)",
    )
    args = parser.parse_args()
    if args.head_dim != 128 and args.dtype != "bf16":
        parser.error("--head-dim 64 / 256 are BF16 families (--dtype bf16)")
    if args.page_size != 16 and args.head_dim != 256:
        parser.error("--page-size 32 / 64 require --head-dim 256")
    q_dtype, kv_dtype = DTYPES[args.dtype]
    experimental = args.dtype == "bf16" and args.head_dim == 128
    if not experimental:
        args.production = True
    rows = args.rows if args.rows else default_rows(args.dtype, args.head_dim)
    device = torch.device("cuda", 0)
    workspace = (
        torch.empty(
            balanced_gqa_decode_workspace_size(device), dtype=torch.uint8, device=device
        )
        if experimental
        else None
    )
    trtllm_workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    cake_workspace = (
        torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        if args.production
        else None
    )
    sm_count = get_device_sm_count(device)
    results = []
    print(
        f"{torch.cuda.get_device_name(device)}, {len(rows)} rows, {args.dtype}, "
        f"head_dim {args.head_dim}, page {args.page_size}"
    )
    header = f"{'row':<30}{'balanced ms':>13}{'TFLOP/s':>9}{'GB/s':>8}"
    if args.production:
        header += f"{'cake ms':>10}{'cake graph':>11}"
    if args.with_trtllm:
        header += f"{'trtllm-gen ms':>15}{'speedup':>9}"
        if args.production:
            header += f"{'cake spd':>9}"
    if args.production:
        header += "  route"
    print(header)
    for name in rows:
        seq_lens, num_kv_heads, q_len, group = ROWS[name]
        num_q_heads = group * num_kv_heads
        max_seq_len = max(seq_lens)
        query, k_cache, v_cache, block_tables, seq_lens_dev, scales = make_inputs(
            seq_lens,
            num_kv_heads,
            q_len,
            device,
            dtype=q_dtype,
            kv_dtype=kv_dtype,
            head_dim=args.head_dim,
            page_size=args.page_size,
            group=group,
        )
        q_scale, k_scale, v_scale = scales
        bmm1_scale = q_scale * k_scale / math.sqrt(args.head_dim)
        bmm2_scale = v_scale  # output scale 1.0
        out = torch.empty_like(query)
        if experimental:
            runner = prepare_balanced_batch_decode_with_kv_cache(
                query,
                (k_cache, v_cache),
                block_tables,
                seq_lens_dev,
                workspace,
                sm_scale=bmm1_scale,
                q_len_per_req=q_len,
                out=out,
            )
            ms = _median_ms(runner)
        else:
            ms = None  # filled from the production route below
        flops = _flops(seq_lens, num_q_heads, q_len, args.head_dim)
        moved = _bytes(
            seq_lens, num_kv_heads, num_q_heads, q_len, args.head_dim, q_dtype, kv_dtype
        )
        row = dict(
            row=name,
            batch=len(seq_lens),
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            q_len=q_len,
            dtype=args.dtype,
            head_dim=args.head_dim,
            page_size=args.page_size,
        )
        line = f"{name:<30}"
        if ms is not None:
            row.update(balanced_ms=ms, tflops=flops / ms / 1e9, gbps=moved / ms / 1e6)
            line += f"{ms:>13.4f}{row['tflops']:>9.1f}{row['gbps']:>8.0f}"
        kv_cache = None
        counter_buffer = None
        if args.production or args.with_trtllm:
            kv_cache = torch.stack([k_cache, v_cache], dim=1).contiguous()
        if args.production:
            # One zero-initialized buffer for both production arms: the
            # kernels reset their counters in-kernel, so it is reused as is.
            counter_buffer = torch.zeros(
                max(
                    get_trtllm_gen_multi_ctas_kv_counter_bytes(
                        len(seq_lens), num_q_heads, sm_count
                    ),
                    cake_fmha_balanced_counter_bytes(
                        sm_count, q_len, head_dim=args.head_dim
                    ),
                ),
                dtype=torch.uint8,
                device=device,
            )
            production_out = torch.empty_like(query)

            def _production():
                cake_batch_decode_with_kv_cache(
                    query,
                    kv_cache,
                    cake_workspace,
                    block_tables,
                    seq_lens_dev,
                    max_seq_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    out=production_out,
                    kv_layout="HND",
                    q_len_per_req=q_len,
                    multi_ctas_kv_counter_buffer=counter_buffer,
                )

            _production()
            torch.cuda.synchronize()
            production_ms = _median_ms(_production)
            production_graph_ms = _median_ms(_production, use_cuda_graph=True)
            if ms is None:
                ms = production_ms
                out.copy_(production_out)
                row.update(
                    balanced_ms=ms, tflops=flops / ms / 1e9, gbps=moved / ms / 1e6
                )
                line = f"{name:<30}{ms:>13.4f}{row['tflops']:>9.1f}{row['gbps']:>8.0f}"
            row.update(
                production_ms=production_ms,
                production_graph_ms=production_graph_ms,
                max_abs_diff_production_vs_balanced=(
                    (production_out.float() - out.float()).abs().max().item()
                ),
                cake_component=_cake_component(
                    query,
                    kv_cache,
                    production_out,
                    cake_workspace,
                    block_tables,
                    seq_lens_dev,
                    q_len=q_len,
                    max_seq_len=max_seq_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    counter_buffer=counter_buffer,
                ),
            )
            line += f"{production_ms:>10.4f}{production_graph_ms:>11.4f}"
        if args.with_trtllm:
            trtllm_out = torch.empty_like(query)

            def _trtllm():
                trtllm_batch_decode_with_kv_cache(
                    query,
                    kv_cache,
                    trtllm_workspace,
                    block_tables,
                    seq_lens_dev,
                    max_seq_len,
                    bmm1_scale,
                    bmm2_scale,
                    out=trtllm_out,
                    backend="trtllm-gen",
                    q_len_per_req=q_len,
                    multi_ctas_kv_counter_buffer=counter_buffer,
                )

            _trtllm()
            torch.cuda.synchronize()
            max_diff = (trtllm_out.float() - out.float()).abs().max().item()
            trtllm_ms = _median_ms(_trtllm)
            row.update(
                trtllm_gen_ms=trtllm_ms,
                speedup=trtllm_ms / ms,
                max_abs_diff_vs_trtllm=max_diff,
            )
            line += f"{trtllm_ms:>15.4f}{trtllm_ms / ms:>9.3f}"
            if args.production:
                row["production_speedup"] = trtllm_ms / row["production_ms"]
                row["max_abs_diff_production_vs_trtllm"] = (
                    (production_out.float() - trtllm_out.float()).abs().max().item()
                )
                line += f"{row['production_speedup']:>9.3f}"
        if args.production:
            line += f"  {row['cake_component']}"
        print(line)
        results.append(row)
        # Release the row's tensors before the next row is built: a 256-request
        # 237k-token head_dim-256 row holds 62 GiB of K / V per copy.
        query = k_cache = v_cache = kv_cache = block_tables = out = None
        counter_buffer = production_out = trtllm_out = runner = None
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                dict(
                    device=torch.cuda.get_device_name(device),
                    dtype=args.dtype,
                    head_dim=args.head_dim,
                    page_size=args.page_size,
                    rows=results,
                ),
                handle,
                indent=2,
            )
    if args.with_trtllm:
        speedups = [r["speedup"] for r in results]
        print(
            f"geomean speedup vs trtllm-gen: {math.exp(sum(map(math.log, speedups)) / len(speedups)):.3f}"
        )
        if args.production:
            prod = [r["production_speedup"] for r in results]
            print(
                f"geomean production backend='cake' speedup vs trtllm-gen: "
                f"{math.exp(sum(map(math.log, prod)) / len(prod)):.3f}"
            )


if __name__ == "__main__":
    main()
