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

"""Cake vs CuTe-DSL Sage-FP8 block-sparse attention on SM100/SM103 (blk64, head_dim 128).

Row A uses the CuTe kernel's own contract (batch 1, 8 heads) so both backends run one call.
Row B is the production shape (batch 8, 32 heads); the CuTe Sage path only accepts B=1/H in
(4, 8), so it runs as 32 contiguous B1/H8 sub-problems and its kernel-sum time is reported.
The quantization row compares the fused Cake quantizer with the torch recipe used by
``tests/attention/test_vsa_block_sparse.py``.
"""

import argparse
import math
import statistics

import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd
from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import (
    bsa_attn_sm100_sage_fwd_cake,
    is_cake_sage_sm100_supported,
    sage_fp8_quantize_sm100,
)
from flashinfer.testing import bench_gpu_time

HEAD_DIM = 128
BLOCK = 64
E4M3_MAX = 448.0


def _torch_quant_recipe(q, k, v):
    """Per-batch torch recipe from tests/attention/test_vsa_block_sparse.py::_quantize_fp8_sage."""
    outs = []
    for b in range(q.shape[0]):
        qb, kb, vb = q[b], k[b], v[b]
        seqlen_k, heads = kb.shape[0], kb.shape[1]
        q_scale_hs = qb.float().abs().amax(dim=-1).clamp_min(1e-6) / E4M3_MAX
        q_fp8 = (
            (qb.float() / q_scale_hs.unsqueeze(-1))
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        n_buckets = (seqlen_k + 15) // 16
        k_amax = (
            kb.float().abs().reshape(n_buckets, 16, heads, HEAD_DIM).amax(dim=(1, 3))
        )
        k_scale = (
            (k_amax.clamp_min(1e-6) / E4M3_MAX).permute(1, 0).unsqueeze(0).contiguous()
        )
        k_scale_bcast = (
            k_scale.squeeze(0).permute(1, 0).repeat_interleave(16, dim=0)[:seqlen_k]
        )
        k_fp8 = (
            (kb.float() / k_scale_bcast.unsqueeze(-1))
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        v_scale = (vb.float().abs().amax(dim=0).clamp_min(1e-6) / E4M3_MAX).contiguous()
        v_fp8 = (
            (vb.float() / v_scale.unsqueeze(0))
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        outs.append(
            (
                q_fp8,
                k_fp8,
                v_fp8,
                q_scale_hs.permute(1, 0).unsqueeze(0).contiguous(),
                k_scale,
                v_scale,
            )
        )
    return outs


def _median_ms(fn):
    times = bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True)
    return statistics.median(times)


def _kernel_sum_ms(fns):
    """Sum of per-launch median GPU times for a list of single-kernel callables.

    Used for the CuTe rows at the production shape: the CuTe Sage path only
    accepts ``batch == 1`` and ``num_head in (4, 8)``, so B8/H32 runs as 32
    sub-calls. Timing them together with ``bench_gpu_time`` would report the
    span from the first kernel start to the last kernel end and charge CuTe for
    host-side gaps between launches; timing each sub-call on its own (same
    CUPTI path, cold L2) and summing the medians keeps the comparison
    kernel-only. (torch.profiler is deliberately not used here: its CUPTI
    session slows down every later ``bench_gpu_time`` measurement in the same
    process.)
    """
    return sum(_median_ms(fn) for fn in fns)


def _inputs(batch, heads, seq, sel, device, seed=7):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        (batch, seq, heads, HEAD_DIM), device=device, generator=g
    ).bfloat16()
    k = torch.randn(
        (batch, seq, heads, HEAD_DIM), device=device, generator=g
    ).bfloat16()
    v = torch.randn(
        (batch, seq, heads, HEAD_DIM), device=device, generator=g
    ).bfloat16()
    q8, k8, v8, q_scale, k_scale, v_scale = sage_fp8_quantize_sm100(q, k, v)
    blocks = seq // BLOCK
    scores = torch.rand((batch, heads, blocks, blocks), device=device, generator=g)
    index = (
        scores.argsort(dim=-1, descending=True)[..., :sel].to(torch.int32).contiguous()
    )
    return dict(
        q=q,
        k=k,
        v=v,
        q8=q8,
        k8=k8,
        v8=v8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        index=index,
    )


def _tflops(batch, heads, seq, sel, ms):
    flops = 4.0 * batch * heads * seq * sel * BLOCK * HEAD_DIM
    return flops / ms / 1e9


def run(seq, sels, skip_registry):
    device = torch.device("cuda")
    scale = 1.0 / math.sqrt(HEAD_DIM)
    print(
        f"device={torch.cuda.get_device_name(device)} cc={torch.cuda.get_device_capability(device)} seq={seq}"
    )
    print(
        f"{'row':>10} {'B':>3} {'H':>3} {'sel/64':>7} {'cake_ms':>9} {'cake_TF':>8} {'cute_ms':>9} {'speedup':>8}"
    )
    for sel in sels:
        i = _inputs(1, 8, seq, sel, device)
        cake = lambda: bsa_attn_sm100_sage_fwd_cake(
            i["q8"],
            i["k8"],
            i["v8"],
            i["index"],
            sel,
            q_scale=i["q_scale"],
            k_scale=i["k_scale"],
            v_scale=i["v_scale"],
            softmax_scale=scale,
        )
        cute = lambda: bsa_attn_sm100_blk64_fwd(
            i["q8"],
            i["k8"],
            i["v8"],
            i["index"],
            sel,
            softmax_scale=scale,
            kv_splits=1,
            q_scale=i["q_scale"],
            k_scale=i["k_scale"],
            v_scale=i["v_scale"][0],
        )
        cake()
        cute()
        torch.cuda.synchronize()
        a, b = _median_ms(cake), _median_ms(cute)
        print(
            f"{'fi-native':>10} {1:>3} {8:>3} {sel:>7} {a:>9.4f} {_tflops(1, 8, seq, sel, a):>8.0f} {b:>9.4f} {b / a:>7.2f}x"
        )
    if skip_registry:
        return
    for sel in sels:
        i = _inputs(8, 32, seq, sel, device)
        cake = lambda: bsa_attn_sm100_sage_fwd_cake(
            i["q8"],
            i["k8"],
            i["v8"],
            i["index"],
            sel,
            q_scale=i["q_scale"],
            k_scale=i["k_scale"],
            v_scale=i["v_scale"],
            softmax_scale=scale,
        )
        subs = []
        for b in range(8):
            for h0 in range(0, 32, 8):
                hs = slice(h0, h0 + 8)
                subs.append(
                    (
                        i["q8"][b : b + 1, :, hs].contiguous(),
                        i["k8"][b : b + 1, :, hs].contiguous(),
                        i["v8"][b : b + 1, :, hs].contiguous(),
                        i["q_scale"][b : b + 1, hs].contiguous(),
                        i["k_scale"][b : b + 1, hs].contiguous(),
                        i["v_scale"][b, hs].contiguous(),
                        i["index"][b : b + 1, hs].contiguous(),
                    )
                )

        def _cute_call(q8, k8, v8, qs, ks, vs, idx):
            return lambda: bsa_attn_sm100_blk64_fwd(
                q8,
                k8,
                v8,
                idx,
                sel,
                softmax_scale=scale,
                kv_splits=1,
                q_scale=qs,
                k_scale=ks,
                v_scale=vs,
            )

        cute_calls = [_cute_call(*sub) for sub in subs]
        cake()
        for call in cute_calls:
            call()
        torch.cuda.synchronize()
        a = _median_ms(cake)
        b = _kernel_sum_ms(cute_calls)
        print(
            f"{'registry':>10} {8:>3} {32:>3} {sel:>7} {a:>9.4f} {_tflops(8, 32, seq, sel, a):>8.0f} {b:>9.4f} {b / a:>7.2f}x  (cute = sum of 32 B1/H8 calls, each timed alone)"
        )
    i = _inputs(8, 32, seq, sels[0], device)
    fused = lambda: sage_fp8_quantize_sm100(i["q"], i["k"], i["v"])
    recipe = lambda: _torch_quant_recipe(i["q"], i["k"], i["v"])
    fused()
    recipe()
    torch.cuda.synchronize()
    a, b = _median_ms(fused), _median_ms(recipe)
    print(
        f"{'quantize':>10} {8:>3} {32:>3} {'-':>7} {a:>9.4f} {'':>8} {b:>9.4f} {b / a:>7.2f}x  (cute = torch recipe, GPU span of its launches)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--sels", default="6,32,58")
    ap.add_argument("--skip-registry", action="store_true")
    args = ap.parse_args()
    if not is_cake_sage_sm100_supported():
        raise SystemExit("requires SM100 or SM103")
    run(args.seq, [int(s) for s in args.sels.split(",")], args.skip_registry)
