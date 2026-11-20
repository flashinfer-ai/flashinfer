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

"""Benchmark the experimental Kimi-K3 vision tower on SM100/SM103.

Representative image / video requests of the Kimi-K3 media processor
(``grid_thws`` per row, 27 encoder layers, random BF16 parameters) timed with
CUPTI and a cold L2 between iterations at the complete-call boundary: packed
BF16 patch pixels + ``grid_thws`` (with the per-grid tables cached) to the
BF16 ``[N, 7168]`` output.  The baseline is the HF module chain in torch
(``F.linear`` / cuBLAS, RMSNorm, RoPE, GELU, residual adds, the 2x2 merge)
with the fastest FlashInfer ragged BF16 attention route
(``BatchPrefillWithRaggedKVCacheWrapper``, planned once per row).  Both arms
are captured into CUDA graphs and replayed (the launch-overhead-free serving
form); ``--eager`` times the direct launches instead.  ``--stages`` adds the
per-stage kernel times of the prepared runner.

Usage::

    python benchmarks/bench_cake_kimi_k3_vision_tower.py \
        [--rows img_448 video_720p_4f ...] [--layers 27] [--eager] [--stages] \
        [--attention-backends fa2 cutlass cute-dsl] [--json out.json]
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import (
    FFN,
    HEAD_DIM,
    HEADS,
    HIDDEN,
    MERGED_DIM,
    NORM_EPS,
    PATCH,
    PATCH_DIM,
    PROJECTOR_EPS,
    QKV_HIDDEN,
    QKV_N,
    SOFTMAX_SCALE,
    TEXT_HIDDEN,
    cu_seqlens_of,
    merged_tokens,
    pos_emb_rows,
    prepare_kimi_k3_vision_tower,
    prepare_kimi_k3_vision_weights,
    rope_cos_sin,
    sincos_time_table,
)
from flashinfer.testing import bench_gpu_time_with_cupti

# Contract rows (Cake ``eval_contract_kimi_k3_vision_tower``): grid_thws per request.
ROWS = {
    "img_224": [(1, 16, 16)],
    "img_336": [(1, 24, 24)],
    "img_448": [(1, 32, 32)],
    "img_640x480": [(1, 36, 46)],
    "img_800x600": [(1, 44, 58)],
    "img_1024x768": [(1, 56, 74)],
    "img_1280x720": [(1, 52, 92)],
    "img_1920x1080": [(1, 78, 138)],
    "doc_1240x1754": [(1, 126, 90)],
    "img_2560x1440": [(1, 104, 184)],
    "img_3840x2160": [(1, 156, 276)],
    "img_max_4096sq": [(1, 258, 258)],
    "batch4_1024x768": [(1, 56, 74)] * 4,
    "batch8_448": [(1, 32, 32)] * 8,
    "mixed_1080p_xga_448_336": [(1, 78, 138), (1, 56, 74), (1, 32, 32), (1, 24, 24)],
    "video_720p_4f": [(4, 52, 92)],
    "video_1080p_4f": [(4, 78, 138)],
    "video_720p_32f": [(4, 52, 92)] * 8,
    "video_480p_64f": [(4, 36, 46)] * 16,
}
ATTENTION_BACKENDS = ("fa2", "cutlass", "cute-dsl", "cudnn")


def make_weights(device, layers, seed=6230):
    g = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16

    def normal(shape, std):
        return (
            torch.empty(shape, dtype=torch.float32, device=device)
            .normal_(0.0, std, generator=g)
            .to(bf16)
        )

    def uniform(shape, lo, hi):
        return (
            torch.empty(shape, dtype=torch.float32, device=device)
            .uniform_(lo, hi, generator=g)
            .to(bf16)
        )

    weights = {
        "patch_proj": normal((HIDDEN, PATCH_DIM), 0.02),
        "pos_emb": normal((64, 64, HIDDEN), 0.02),
        "time_weight": sincos_time_table().to(device=device, dtype=bf16),
        "final_norm": uniform((HIDDEN,), 0.9, 1.1),
        "merger_proj0": normal((MERGED_DIM, MERGED_DIM), math.sqrt(2.0 / MERGED_DIM)),
        "merger_proj1": normal((TEXT_HIDDEN, MERGED_DIM), math.sqrt(2.0 / MERGED_DIM)),
        "post_norm": uniform((TEXT_HIDDEN,), 0.9, 1.1),
        "layers": [
            {
                "norm0": uniform((HIDDEN,), 0.9, 1.1),
                "wqkv": normal((QKV_N, HIDDEN), 0.02),
                "wo": normal((HIDDEN, QKV_HIDDEN), 0.02),
                "norm1": uniform((HIDDEN,), 0.9, 1.1),
                "fc0": normal((FFN, HIDDEN), math.sqrt(2.0 / HIDDEN)),
                "fc1": normal((HIDDEN, FFN), math.sqrt(2.0 / FFN)),
            }
            for _ in range(layers)
        ],
    }
    return weights


def tower_flops(grids):
    """(per-layer GEMM FLOPs, per-layer attention FLOPs, patch-embed + projector FLOPs)."""
    cu = cu_seqlens_of(grids)
    total, merged = cu[-1], merged_tokens(grids)
    gemm = 2.0 * total * (HIDDEN * QKV_N + QKV_HIDDEN * HIDDEN + 2 * HIDDEN * FFN)
    attn = float(
        sum(
            4 * HEADS * (b - a) ** 2 * HEAD_DIM
            for a, b in zip(cu, cu[1:], strict=False)
        )
    )
    merger = 2.0 * merged * (MERGED_DIM * MERGED_DIM + MERGED_DIM * TEXT_HIDDEN)
    return gemm, attn, 2.0 * total * PATCH_DIM * HIDDEN + merger


# ---------------------------------------------------------------------------
# HF torch chain with the FlashInfer ragged attention route
# ---------------------------------------------------------------------------


def _rms_norm(x, weight, eps):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * weight.float()).to(x.dtype)


def _apply_rope(q, k, cos, sin):
    def rot(x):
        xf = x.float().reshape(*x.shape[:-1], HEAD_DIM // 2, 2)
        a, b = xf[..., 0], xf[..., 1]
        c = cos.reshape(cos.shape[0], 1, HEAD_DIM // 2)
        s = sin.reshape(sin.shape[0], 1, HEAD_DIM // 2)
        return (
            torch.stack([a * c - b * s, a * s + b * c], dim=-1)
            .reshape(x.shape)
            .to(x.dtype)
        )

    return rot(q), rot(k)


def _tpool_merge(x, grids):
    outputs = []
    start = 0
    for t, h, w in grids:
        n = t * h * w
        seq = x[start : start + n]
        start += n
        nh, nw = h // 2, w // 2
        r = (
            seq.view(t, nh, 2, nw, 2, x.shape[-1])
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .mean(dim=0)
        )
        outputs.append(r.reshape(nh * nw, 4 * x.shape[-1]))
    return torch.cat(outputs, dim=0)


class RaggedAttention:
    """One planned FlashInfer ragged BF16 route writing into a private output."""

    def __init__(self, backend, cu, device, workspace):
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        self.backend = backend
        T = cu[-1]
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
        if backend == "cudnn":
            cu_seqlens = cu_seqlens * (HEADS * HEAD_DIM)
        self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace, kv_layout="NHD", backend=backend
        )
        self.wrapper.plan(
            cu_seqlens,
            cu_seqlens,
            HEADS,
            HEADS,
            HEAD_DIM,
            causal=False,
            sm_scale=SOFTMAX_SCALE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
        )
        self.out = torch.empty(
            (T, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
        )

    def __call__(self, q, k, v):
        result = self.wrapper.run(q, k, v, out=self.out)
        return self.out if result is None else result


def hf_chain(pixels, grids, weights, cos, sin, pos_rows, attention):
    T = pixels.shape[0]
    x = F.linear(pixels.view(T, PATCH_DIM), weights["patch_proj"]) + pos_rows
    for lw in weights["layers"]:
        n = _rms_norm(x, lw["norm0"], NORM_EPS)
        qkv = F.linear(n, lw["wqkv"]).view(T, 3, HEADS, HEAD_DIM)
        q, k, v = qkv.unbind(dim=1)
        q, k = _apply_rope(q, k, cos, sin)
        a = attention(q.contiguous(), k.contiguous(), v.contiguous())
        x = x + F.linear(a.reshape(T, QKV_HIDDEN), lw["wo"])
        n = _rms_norm(x, lw["norm1"], NORM_EPS)
        x = x + F.linear(F.gelu(F.linear(n, lw["fc0"]), approximate="tanh"), lw["fc1"])
    x = _rms_norm(x, weights["final_norm"], NORM_EPS)
    m = _tpool_merge(x, grids)
    y = F.linear(F.gelu(F.linear(m, weights["merger_proj0"])), weights["merger_proj1"])
    return _rms_norm(y, weights["post_norm"], PROJECTOR_EPS)


def graph_capture(run):
    """Capture ``run`` (warmed up) into a CUDA graph; the returned callable replays it."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    def replay():
        graph.replay()
        return captured

    replay._graph = graph
    return replay


def _median_ms(fn, repeat_time_ms):
    times = sorted(
        bench_gpu_time_with_cupti(fn, cold_l2_cache=True, repeat_time_ms=repeat_time_ms)
    )
    return float(times[len(times) // 2])


def _fastest_attention(cu, device, workspace, backends, q, k, v):
    best = None
    timings = {}
    for backend in backends:
        try:
            route = RaggedAttention(backend, cu, device, workspace)
            out = route(q, k, v)
            torch.cuda.synchronize()
            if bool(torch.isnan(out).any()):
                timings[backend] = "nan output"
                continue
            ms = _median_ms(lambda: route(q, k, v), 100)
        except Exception as exc:  # noqa: BLE001 - an unavailable route is recorded, not fatal
            timings[backend] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        timings[backend] = ms
        if best is None or ms < best[0]:
            best = (ms, backend, route)
    if best is None:
        raise RuntimeError(f"no FlashInfer ragged attention route available: {timings}")
    return best[1], best[2], timings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=list(ROWS))
    parser.add_argument("--layers", type=int, default=27)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="time the direct launch sequences instead of CUDA-graph replays",
    )
    parser.add_argument(
        "--stages",
        action="store_true",
        help="also time every stage of the prepared runner",
    )
    parser.add_argument(
        "--attention-backends",
        nargs="*",
        default=list(ATTENTION_BACKENDS),
        choices=ATTENTION_BACKENDS,
    )
    parser.add_argument(
        "--repeat-ms",
        type=int,
        default=300,
        help="CUPTI sampling window per arm in milliseconds",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    weights = make_weights(device, args.layers)
    prepared = prepare_kimi_k3_vision_weights(weights)
    fi_workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    results = []
    form = "eager" if args.eager else "cuda_graph"
    print(
        f"{torch.cuda.get_device_name(device)}, {len(args.rows)} rows, {args.layers} layers, {form}"
    )
    print(
        f"{'row':<26}{'T':>8}{'N':>7}{'cake ms':>10}{'hf+fi ms':>10}{'speedup':>9}{'TFLOP/s':>9}  attention"
    )
    for name in args.rows:
        grids = ROWS[name]
        cu = cu_seqlens_of(grids)
        T, N = cu[-1], merged_tokens(grids)
        gen = torch.Generator(device=device).manual_seed(1)
        pixels = torch.empty(
            (T, 3, PATCH, PATCH), dtype=torch.bfloat16, device=device
        ).uniform_(-1.0, 1.0, generator=gen)
        cos, sin = rope_cos_sin(grids, device)
        pos_rows = pos_emb_rows(weights["pos_emb"], weights["time_weight"], grids)
        out = torch.empty((N, TEXT_HIDDEN), dtype=torch.bfloat16, device=device)
        runner = prepare_kimi_k3_vision_tower(
            pixels, grids, prepared, out, pos_rows=pos_rows
        )
        runner.launch()
        torch.cuda.synchronize()
        q = torch.randn(
            (T, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device, generator=gen
        )
        backend, attention, route_timings = _fastest_attention(
            cu, device, fi_workspace, args.attention_backends, q, q, q
        )
        del q

        def chain(pixels=pixels, attention=attention):
            return hf_chain(pixels, grids, weights, cos, sin, pos_rows, attention)

        chain()
        torch.cuda.synchronize()
        cake_fn, chain_fn = runner.launch, chain
        if not args.eager:
            cake_fn, chain_fn = graph_capture(runner.launch), graph_capture(chain)
        cake_ms = _median_ms(cake_fn, args.repeat_ms)
        chain_ms = _median_ms(chain_fn, args.repeat_ms)
        gemm, attn, rest = tower_flops(grids)
        flops = (gemm + attn) * args.layers + rest
        row = dict(
            row=name,
            grid_thws=[list(g) for g in grids],
            total_tokens=T,
            merged_tokens=N,
            layers=args.layers,
            form=form,
            cake_ms=cake_ms,
            hf_chain_ms=chain_ms,
            speedup=chain_ms / cake_ms,
            tflops=flops / cake_ms / 1e9,
            attention_route=backend,
            attention_route_timings_ms=route_timings,
            route_metadata=runner.route_metadata,
        )
        if args.stages:
            row["stage_ms"] = {
                stage: _median_ms(fn, 100) for stage, fn in runner.stages.items()
            }
        print(
            f"{name:<26}{T:>8}{N:>7}{cake_ms:>10.4f}{chain_ms:>10.4f}{row['speedup']:>9.3f}{row['tflops']:>9.1f}  {backend}"
        )
        if args.stages:
            print(
                "    " + ", ".join(f"{k}={v:.4f}" for k, v in row["stage_ms"].items())
            )
        results.append(row)
        del runner, out, cake_fn, chain_fn
        torch.cuda.empty_cache()
    speedups = [r["speedup"] for r in results]
    print(
        f"geomean speedup vs HF chain + FlashInfer ragged attention: {math.exp(sum(map(math.log, speedups)) / len(speedups)):.3f}"
    )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                dict(device=torch.cuda.get_device_name(device), rows=results),
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
