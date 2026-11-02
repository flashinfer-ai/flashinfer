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

"""Benchmark the experimental MiniMax-H3 packed-varlen attention on SM100/SM103.

Packed THD ``[T, H, 128]`` BF16 rows (56 global heads split over Ulysses
degree P = 1/2/4/8, single production-center segments of 4k-110k tokens,
ragged tails, trailing padding and multi-segment packs) timed with CUPTI and
a cold L2 between iterations.  Every variant reports the complete-call time
(the customer boundary: BF16 inputs + ``cu_seqlens`` -> BF16 output).  The
NVFP4 variants additionally report quantization-only and attention-only times
measured on the prepared runner's stage callables.  ``--with-flashinfer``
adds ``BatchPrefillWithRaggedKVCacheWrapper`` (planned once outside the timed
region, ``run()`` timed) on the same tensors as the BF16 reference route.

``--operand-layout`` selects how Q/K/V are laid out: ``contract`` (default,
contiguous ``[T, H, 128]``), ``engine`` (column chunks of the fused QKV
projection ``[T, 3 * H * 128]``, strides ``(3 * H * 128, 128, 1)``) or
``pack`` (kind slices of the Cake pre-attention pack ``[T, H, 3, 128]``,
strides ``(H * 384, 384, 1)``).  The BF16 kernel consumes the strided views in
place; for the non-contract layouts the ``copies ms`` column times the three
THD ``.contiguous()`` copies a route needed before, so the saving is visible
without a second kernel.  The NVFP4 variants keep the contiguous contract and
are skipped for the other layouts.

Usage::

    python benchmarks/bench_cake_minimax_h3_varlen_attention.py \
        [--variants bf16 nvfp4_fp8 nvfp4_fp4] [--rows center_5s_p8_m4824 ...] \
        [--operand-layout contract|engine|pack] [--with-flashinfer] [--json out.json]
"""

import argparse
import json
import math

import torch

from flashinfer.experimental.minimax_h3_varlen_attention.cake_backend import (
    HEAD_DIM,
    prepare_minimax_h3_varlen_attention,
    prepare_minimax_h3_varlen_nvfp4_attention,
)
from flashinfer.testing import bench_gpu_time_with_cupti

GLOBAL_HEADS = 56

# (cu_seqlens, Ulysses degree).  Local heads = 56 / degree.  Labels follow the
# Cake evaluation contract rows (``m<N>`` = longest segment).
ROWS = {
    "center_4s_p1_m33472": ([0, 33472], 1),
    "center_4s_p8_m4184": ([0, 4184], 8),
    "center_5s_p1_m38592": ([0, 38592], 1),
    "center_5s_p2_m19296": ([0, 19296], 2),
    "center_5s_p4_m9648": ([0, 9648], 4),
    "center_5s_p8_m4824": ([0, 4824], 8),
    "center_8s_p1_m58944": ([0, 58944], 1),
    "center_8s_p8_m7368": ([0, 7368], 8),
    "center_10s_p1_m74240": ([0, 74240], 1),
    "center_10s_p4_m18560": ([0, 18560], 4),
    "center_15s_p1_m109952": ([0, 109952], 1),
    "center_15s_p8_m13744": ([0, 13744], 8),
    "tail_5s_p1_m38531": ([0, 38531], 1),
    "tail_5s_p8_m4763": ([0, 4763], 8),
    "pad_5s_p4_used9611_total9664": ([0, 9611, 9664], 4),
    "pad_15s_p1_used109901_total109952": ([0, 109901, 109952], 1),
    "seg3_5s_p8_m4824": ([0, 4310, 4567, 4824], 8),
    "seg4_6s_p2_m24384": ([0, 12285, 16401, 20393, 24384], 2),
}
VARIANTS = ("bf16", "nvfp4_fp8", "nvfp4_fp4")
OPERAND_LAYOUTS = ("contract", "engine", "pack")


def make_inputs(cu, degree, device, seed=0, layout="contract"):
    heads = GLOBAL_HEADS // degree
    gen = torch.Generator(device=device).manual_seed(seed)
    total = cu[-1]
    shape = (total, heads, HEAD_DIM)
    if layout == "contract":
        q = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
        k = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
        v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
    elif layout == "engine":
        # Column chunks of the fused QKV projection, never copied.
        qkv = torch.randn(
            (total, 3 * heads * HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        width = heads * HEAD_DIM
        q, k, v = (
            qkv[:, i * width : (i + 1) * width].view(total, heads, HEAD_DIM)
            for i in range(3)
        )
        assert q.stride() == (3 * heads * HEAD_DIM, HEAD_DIM, 1)
    elif layout == "pack":
        # Kind slices of the Cake pre-attention pack [T, H, 3, 128], never copied.
        pack = torch.randn(
            (total, heads, 3, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        q, k, v = (pack[:, :, i, :] for i in range(3))
        assert q.stride() == (heads * 3 * HEAD_DIM, 3 * HEAD_DIM, 1)
    else:
        raise ValueError(f"unknown operand layout {layout!r}")
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens, heads


def _thd_copies(q, k, v):
    """The three THD copies a route needed before strided Q/K/V were accepted."""
    return q.contiguous(), k.contiguous(), v.contiguous()


def _flops(cu, heads):
    return sum(
        4.0 * heads * HEAD_DIM * (b - a) ** 2 for a, b in zip(cu, cu[1:], strict=False)
    )


def _median_ms(fn):
    times = sorted(bench_gpu_time_with_cupti(fn, cold_l2_cache=True))
    return float(times[len(times) // 2])


def _flashinfer_ragged(q, k, v, cu_seqlens, heads, sm_scale, workspace):
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        cu_seqlens,
        cu_seqlens,
        heads,
        heads,
        HEAD_DIM,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out = torch.empty_like(q)

    def run():
        wrapper.run(q, k, v, out=out)
        return out

    return run, out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=list(ROWS))
    parser.add_argument(
        "--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS
    )
    parser.add_argument("--operand-layout", choices=OPERAND_LAYOUTS, default="contract")
    parser.add_argument("--with-flashinfer", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    layout = args.operand_layout
    device = torch.device("cuda", 0)
    sm_scale = HEAD_DIM**-0.5
    fi_workspace = (
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        if args.with_flashinfer
        else None
    )
    results = []
    print(
        f"{torch.cuda.get_device_name(device)}, {len(args.rows)} rows x "
        f"{len(args.variants)} variants, operand layout: {layout}"
    )
    header = f"{'row':<36}{'variant':<11}{'total ms':>10}{'quant ms':>10}{'attn ms':>10}{'TFLOP/s':>9}"
    if layout != "contract":
        header += f"{'copies ms':>11}"
    if args.with_flashinfer:
        header += f"{'FI ragged ms':>14}{'speedup':>9}"
    print(header)
    for name in args.rows:
        cu, degree = ROWS[name]
        q, k, v, cu_seqlens, heads = make_inputs(cu, degree, device, layout=layout)
        flops = _flops(cu, heads)
        fi_ms = None
        if args.with_flashinfer:
            fi_q, fi_k, fi_v = _thd_copies(q, k, v)
            fi_run, _ = _flashinfer_ragged(
                fi_q, fi_k, fi_v, cu_seqlens, heads, sm_scale, fi_workspace
            )
            fi_run()
            torch.cuda.synchronize()
            fi_ms = _median_ms(fi_run)
            del fi_q, fi_k, fi_v
        copies_ms = None
        if layout != "contract":
            copies_ms = _median_ms(lambda q=q, k=k, v=v: _thd_copies(q, k, v))
        for variant in args.variants:
            if variant != "bf16" and layout != "contract":
                print(f"{name:<36}{variant:<11}  skipped (contiguous contract only)")
                continue
            out = torch.empty(
                (cu[-1], heads, HEAD_DIM), dtype=torch.bfloat16, device=device
            )
            quant_ms = attn_ms = None
            if variant == "bf16":
                runner = prepare_minimax_h3_varlen_attention(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    softmax_scale=sm_scale,
                    out=out,
                    cu_seqlens_host=cu,
                )
                runner()
                torch.cuda.synchronize()
                total_ms = _median_ms(runner)
            else:
                runner = prepare_minimax_h3_varlen_nvfp4_attention(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    pv_mode=variant.split("_")[1],
                    softmax_scale=sm_scale,
                    out=out,
                    cu_seqlens_host=cu,
                )
                runner()
                torch.cuda.synchronize()
                total_ms = _median_ms(runner)
                quant_ms = _median_ms(runner.quantize)
                attn_ms = _median_ms(runner.attention)
            row = dict(
                row=name,
                variant=variant,
                total_tokens=cu[-1],
                segments=len(cu) - 1,
                ulysses_degree=degree,
                local_heads=heads,
                operand_layout=layout,
                q_strides=list(q.stride()),
                total_ms=total_ms,
                quantize_ms=quant_ms,
                attention_ms=attn_ms,
                thd_copies_ms=copies_ms,
                tflops=flops / total_ms / 1e9,
                route_metadata=runner.route_metadata,
            )
            line = (
                f"{name:<36}{variant:<11}{total_ms:>10.4f}"
                f"{(quant_ms if quant_ms is not None else float('nan')):>10.4f}"
                f"{(attn_ms if attn_ms is not None else float('nan')):>10.4f}{row['tflops']:>9.1f}"
            )
            if copies_ms is not None:
                line += f"{copies_ms:>11.4f}"
            if fi_ms is not None:
                row.update(flashinfer_ragged_ms=fi_ms, speedup=fi_ms / total_ms)
                line += f"{fi_ms:>14.4f}{fi_ms / total_ms:>9.3f}"
            print(line)
            results.append(row)
        del q, k, v
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                dict(
                    device=torch.cuda.get_device_name(device),
                    operand_layout=layout,
                    rows=results,
                ),
                handle,
                indent=2,
            )
    if args.with_flashinfer:
        for variant in args.variants:
            speedups = [r["speedup"] for r in results if r["variant"] == variant]
            print(
                f"{variant}: geomean speedup vs FlashInfer ragged BF16 "
                f"{math.exp(sum(map(math.log, speedups)) / len(speedups)):.3f}"
            )


if __name__ == "__main__":
    main()
