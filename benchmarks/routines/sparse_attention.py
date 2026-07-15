"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark routines for Minimax Sparse Attention (MSA): per-stage ops plus the
end-to-end MSAPipeline. Precision is chosen via ``--q_dtype`` / ``--kv_dtype``.
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm12x_supported

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
)

BLK_KV = 128

# Which kernel backends each MSA routine actually exposes.
_ROUTINE_BACKENDS = {
    "MSAProxyScore": ["cudsl"],
    "MSASparseAttention": ["cudsl"],
    "MSASparseDecode": ["cudsl"],
    "MSAPipeline": ["cudsl"],
}


def run_sparse_attention_test(args):
    """Route an MSA routine to its test function."""
    if args.routine == "MSAProxyScore":
        return testMSAProxyScore(args)
    elif args.routine == "MSASparseAttention":
        return testMSASparseAttention(args)
    elif args.routine == "MSASparseDecode":
        return testMSASparseDecode(args)
    elif args.routine == "MSAPipeline":
        return testMSAPipeline(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_sparse_attention_args(line, parser):
    """Parse MSA-specific command line arguments."""
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cudsl"],
        choices=["cudsl"],
        help=(
            "Kernel implementation. All MSA ops have a single CuTe-DSL impl "
            "('cudsl'); precision is chosen via --q_dtype / --kv_dtype."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of requests.")
    parser.add_argument(
        "--s_qo", type=int, default=4096, help="Query tokens per request."
    )
    parser.add_argument("--s_kv", type=int, default=4096, help="KV tokens per request.")
    parser.add_argument("--num_qo_heads", type=int, default=64)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument(
        "--head_dim", type=int, default=128, help="Head dim (MSA requires 128)."
    )
    parser.add_argument(
        "--topk", type=int, default=16, help="Selected KV blocks per query."
    )
    parser.add_argument(
        "--max_k_tiles",
        type=int,
        default=None,
        help="KV-block columns for proxy/topk (default ceil(s_kv/128)).",
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Causal masking (use --no-causal to disable).",
    )
    parser.add_argument("--q_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--kv_dtype",
        type=str,
        default="bfloat16",
        help="KV dtype: bfloat16/float16/fp8_e4m3/nvfp4 (prefill/decode/pipeline).",
    )
    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


# --------------------------------------------------------------------------- #
# input construction
# --------------------------------------------------------------------------- #
def _resolve_backends(args):
    """Drop backends the routine does not implement; warn once for each."""
    supported = _ROUTINE_BACKENDS[args.routine]
    out = []
    for b in args.backends:
        if b in supported:
            out.append(b)
        else:
            print(
                f"[WARNING] backend {b} not implemented for {args.routine}; skipping."
            )
    return out


def _cu_seqlens(batch_size, s, device):
    return torch.arange(0, (batch_size + 1) * s, s, device=device, dtype=torch.int32)


def _rand_q2k(batch_size, s_qo, s_kv, num_kv_heads, topk, device):
    """Random ascending, -1-padded block selection (msa_topk_select output format)."""
    total_q = batch_size * s_qo
    nb = -(-s_kv // BLK_KV)
    idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=device
    )
    nsel = min(topk, nb)
    for h in range(num_kv_heads):
        for qi in range(total_q):
            sel = torch.randperm(nb, device=device)[:nsel].sort().values
            idx[h, qi, :nsel] = sel.to(torch.int32)
    return idx


def _maybe_quantize_kv(k, v, kv_dtype):
    """Return (k_in, v_in, extra_kwargs) for the requested KV dtype."""
    if kv_dtype in (torch.bfloat16, torch.float16):
        return k.to(kv_dtype), v.to(kv_dtype), {}
    if kv_dtype == torch.float8_e4m3fn:
        return k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn), {}
    # nvfp4 (uint8-packed)
    from flashinfer import nvfp4_quantize

    total_k, Hkv, _ = k.shape

    def _q(x2d):
        gsf = (448.0 * 6.0) / x2d.float().abs().max()
        xq, sf = nvfp4_quantize(x2d, gsf.to(x2d.device), sf_vec_size=16)
        return xq.view(torch.uint8), sf.view(torch.uint8), float(1.0 / gsf)

    kq, ksf, kg = _q(k.reshape(-1, 128))
    vq, vsf, vg = _q(v.reshape(-1, 128))
    return (
        kq.reshape(total_k, Hkv, 64),
        vq.reshape(total_k, Hkv, 64),
        dict(k_scale=ksf, v_scale=vsf, k_global_scale=kg, v_global_scale=vg),
    )


def _common(args):
    """Shared scalar setup + SM12x gate."""
    device = get_device(args)
    if not is_sm12x_supported(device):
        print("[ERROR] MSA kernels require SM120 / SM121 (Blackwell). Exiting.")
        return None
    if args.head_dim != 128:
        print(f"[ERROR] MSA requires head_dim 128, got {args.head_dim}. Exiting.")
        return None
    if args.generate_repro_command:
        print(f"[INFO] To reproduce this test case, run: {args.repro_command}")
    return device


def _record(args, backend, times, *, tflops=0.0, tb_per_sec=0.0, total_q=0, total_kv=0):
    """Build one CSV result row from a list of per-iter times."""
    median_time = float(np.median(times))
    std_time = float(np.std(times))
    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
    cur = defaultdict(str)
    cur["routine"] = args.routine
    cur["median_time"] = median_time
    cur["std_time"] = std_time
    cur["tflops"] = tflops
    cur["tb_per_sec"] = tb_per_sec
    cur["backend"] = backend
    cur["topk"] = args.topk
    cur["max_k_tiles"] = args.max_k_tiles if args.max_k_tiles else ""
    cur["total_q"] = total_q
    cur["total_kv"] = total_kv
    cur["s_qo"] = args.s_qo
    cur["s_kv"] = args.s_kv
    cur["head_dim_qk"] = args.head_dim
    cur["head_dim_vo"] = args.head_dim
    cur["causal"] = args.causal
    cur["q_dtype"] = args.q_dtype
    cur["kv_dtype"] = args.kv_dtype
    cur["case_tag"] = args.case_tag
    return cur


# --------------------------------------------------------------------------- #
# per-op routines
# --------------------------------------------------------------------------- #
def testMSAProxyScore(args):
    """Stage 1: dense per-KV-block max logits (the MSA indexer). ``--q_dtype``
    picks the kernel: bfloat16/float16 -> ``msa_proxy_score``, nvfp4 ->
    ``msa_proxy_score_fp4`` on the same q/k quantized to NVFP4."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSAProxyScore | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import msa_proxy_score

    bs, s_qo, s_kv = args.batch_size, args.s_qo, args.s_kv
    Hq, Hkv = args.num_qo_heads, args.num_kv_heads
    total_q, total_kv = bs * s_qo, bs * s_kv
    is_fp4 = args.q_dtype == "nvfp4"
    gen_dtype = torch.bfloat16 if is_fp4 else dtype_str_to_torch_dtype(args.q_dtype)
    q = torch.randn(total_q, Hq, 128, device=device, dtype=gen_dtype) / 3
    k = torch.randn(total_kv, Hkv, 128, device=device, dtype=gen_dtype) / 3
    cu_q, cu_k = _cu_seqlens(bs, s_qo, device), _cu_seqlens(bs, s_kv, device)

    # Omitting these makes msa_proxy_score derive them via a blocking cu_seqlens D2H
    # copy, which would time host-sync overhead instead of the kernel (and is illegal
    # under CUDA-graph capture).
    max_seqlen_q = s_qo
    max_k_tiles = args.max_k_tiles or ((s_kv + 127) // 128)

    if is_fp4:
        from flashinfer.msa_ops import msa_proxy_score_fp4
        from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

        q_fp4, q_sc, q_g = _quantize_qk_to_nvfp4(q)
        k_fp4, k_sc, k_g = _quantize_qk_to_nvfp4(k)

    def run(_b=None):
        if is_fp4:
            return msa_proxy_score_fp4(
                q_fp4,
                k_fp4,
                q_sc,
                k_sc,
                q_g,
                k_g,
                cu_q,
                cu_k,
                causal=args.causal,
                max_seqlen_q=max_seqlen_q,
                max_k_tiles=max_k_tiles,
            )
        return msa_proxy_score(
            q,
            k,
            cu_q,
            cu_k,
            causal=args.causal,
            max_seqlen_q=max_seqlen_q,
            max_k_tiles=max_k_tiles,
        )

    # K-read bytes/elem: bf16 reads 2 B/elem; NVFP4 reads 0.5 (packed e2m1) + 1/16
    # (e4m3 block scale) = 0.5625 B/elem.
    b_per_elem = 0.5625 if is_fp4 else 2.0

    res = []
    for b in _resolve_backends(args):
        times = bench_gpu_time(
            fn=run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
            input_args=(b,),
        )
        median = float(np.median(times))  # milliseconds
        # Dense QK over the proxy: 2 * s_qo * s_kv * Hq * head_dim per request.
        # times are in ms, so flops / (1e9 * ms) gives TFLOPs (not 1e12).
        flops = 2 * bs * s_qo * s_kv * Hq * 128
        tflops = flops / (1e9 * median)
        # Physical HBM K bytes: the index K is one kv_head wide, read once per
        # kv_head (the qo_head re-reads within a GQA group are largely L2-served),
        # so use Hkv (not Hq) to keep this a physical-DRAM estimate.
        k_read_bytes = bs * Hkv * s_kv * 128 * b_per_elem
        tb_per_sec = k_read_bytes / (1e9 * median)
        res.append(
            _record(
                args,
                b,
                times,
                tflops=tflops,
                tb_per_sec=tb_per_sec,
                total_q=total_q,
                total_kv=total_kv,
            )
        )

    # refcheck (nvfp4 only): fp4 scores are lossy vs bf16 *by design* (4-bit index
    # K), so we report mean/max rel error against the bf16 proxy on the same q/k and
    # fail only on a gross mismatch (a real bug, not fp4 rounding). Top-k selection
    # overlap is the deployment metric (tests/msa_ops/test_proxy_fp4.py).
    if args.refcheck and is_fp4:
        ref = msa_proxy_score(
            q,
            k,
            cu_q,
            cu_k,
            causal=args.causal,
            max_seqlen_q=max_seqlen_q,
            max_k_tiles=max_k_tiles,
        ).float()
        out = run().float()
        finite = torch.isfinite(ref) & torch.isfinite(out)
        denom = ref[finite].abs().clamp_min(1e-3)
        rel = (out[finite] - ref[finite]).abs() / denom
        mean_rel, max_rel = rel.mean().item(), rel.max().item()
        print(
            f"[REFCHECK] nvfp4 vs bf16 proxy: mean rel {mean_rel:.4f}, max rel "
            f"{max_rel:.4f} (fp4 is lossy; selection overlap is the deployment metric)"
        )
        if mean_rel > 0.5 and not args.allow_output_mismatch:
            raise AssertionError(
                f"nvfp4 proxy mean rel err {mean_rel:.4f} vs bf16 (>0.5)"
            )
    return res


def testMSASparseAttention(args):
    """Stage 3: union-tile sparse prefill on the selected KV blocks."""
    if args.verbose >= 1:
        print(
            f"[INFO] Running testMSASparseAttention | FlashInfer {flashinfer.__version__}"
        )
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import msa_sparse_attention

    bs, s_qo, s_kv = args.batch_size, args.s_qo, args.s_kv
    Hq, Hkv = args.num_qo_heads, args.num_kv_heads
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    total_q, total_kv = bs * s_qo, bs * s_kv
    q = torch.randn(total_q, Hq, 128, device=device, dtype=q_dtype) / 3
    k = torch.randn(total_kv, Hkv, 128, device=device, dtype=q_dtype) / 3
    v = torch.randn(total_kv, Hkv, 128, device=device, dtype=q_dtype) / 3
    cu_q, cu_k = _cu_seqlens(bs, s_qo, device), _cu_seqlens(bs, s_kv, device)
    q2k = _rand_q2k(bs, s_qo, s_kv, Hkv, args.topk, device)
    k_in, v_in, extra = _maybe_quantize_kv(k, v, kv_dtype)
    scale = 1.0 / math.sqrt(128)

    def run(_b):
        return msa_sparse_attention(
            q,
            k_in,
            v_in,
            q2k,
            cu_q,
            cu_k,
            causal=args.causal,
            softmax_scale=scale,
            **extra,
        )

    res = []
    for b in _resolve_backends(args):
        times = bench_gpu_time(
            fn=run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
            input_args=(b,),
        )
        # Selected work: total_q * topk blocks of 128 tokens; QK + PV are two
        # GEMMs, so 4 flops per (token, column, channel).
        flops = 4 * total_q * args.topk * BLK_KV * Hq * 128
        tflops = flops / (1e9 * float(np.median(times)))
        res.append(
            _record(args, b, times, tflops=tflops, total_q=total_q, total_kv=total_kv)
        )
    return res


def testMSASparseDecode(args):
    """Stage 3: single-token decode (hot path, CUDA-graph native)."""
    if args.verbose >= 1:
        print(
            f"[INFO] Running testMSASparseDecode | FlashInfer {flashinfer.__version__}"
        )
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import msa_sparse_decode_attention

    bs, s_kv = args.batch_size, args.s_kv
    Hq, Hkv = args.num_qo_heads, args.num_kv_heads
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    total_q, total_kv = bs, bs * s_kv  # one query token per request
    q = torch.randn(total_q, Hq, 128, device=device, dtype=q_dtype) / 3
    k = torch.randn(total_kv, Hkv, 128, device=device, dtype=q_dtype) / 3
    v = torch.randn(total_kv, Hkv, 128, device=device, dtype=q_dtype) / 3
    cu_k = _cu_seqlens(bs, s_kv, device)
    q2k = _rand_q2k(bs, 1, s_kv, Hkv, args.topk, device)
    k_in, v_in, extra = _maybe_quantize_kv(k, v, kv_dtype)
    scale = 1.0 / math.sqrt(128)

    def run(_b):
        return msa_sparse_decode_attention(
            q,
            k_in,
            v_in,
            q2k,
            cu_seqlens_k=cu_k,
            seqlen_q=1,
            causal=args.causal,
            softmax_scale=scale,
            **extra,
        )

    res = []
    for b in _resolve_backends(args):
        times = bench_gpu_time(
            fn=run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
            input_args=(b,),
        )
        median = float(np.median(times))  # milliseconds
        flops = 4 * total_q * args.topk * BLK_KV * Hq * 128
        tflops = flops / (1e9 * median)
        # Sparse decode is memory-bound: each query token reads its top-k K and V
        # blocks, one kv_head wide. bf16 2 B/elem, fp8 1 B, NVFP4 0.5625 B.
        if args.kv_dtype == "nvfp4":
            b_per_elem = 0.5625
        elif kv_dtype == torch.float8_e4m3fn:
            b_per_elem = 1.0
        else:
            b_per_elem = 2.0
        kv_read_bytes = 2 * total_q * Hkv * args.topk * BLK_KV * 128 * b_per_elem
        tb_per_sec = kv_read_bytes / (1e9 * median)
        res.append(
            _record(
                args,
                b,
                times,
                tflops=tflops,
                tb_per_sec=tb_per_sec,
                total_q=total_q,
                total_kv=total_kv,
            )
        )
    return res


def testMSAPipeline(args):
    """End-to-end MSA prefill: proxy_score -> topk_select -> prefill."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSAPipeline | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import (
        msa_proxy_score,
        msa_sparse_attention,
        msa_topk_select,
    )

    bs, s_qo, s_kv = args.batch_size, args.s_qo, args.s_kv
    Hq, Hkv = args.num_qo_heads, args.num_kv_heads
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    total_q, total_kv = bs * s_qo, bs * s_kv
    q = torch.randn(total_q, Hq, 128, device=device, dtype=q_dtype) / 3
    k = torch.randn(total_kv, Hkv, 128, device=device, dtype=q_dtype) / 3
    v = torch.randn(total_kv, Hkv, 128, device=device, dtype=q_dtype) / 3
    cu_q, cu_k = _cu_seqlens(bs, s_qo, device), _cu_seqlens(bs, s_kv, device)
    k_in, v_in, extra = _maybe_quantize_kv(k, v, kv_dtype)
    scale = 1.0 / math.sqrt(128)
    # The bf16 proxy reads unquantized K; --kv_dtype applies to the attention stage.
    proxy_k = k

    def run(_b):
        max_score = msa_proxy_score(q, proxy_k, cu_q, cu_k, causal=args.causal)
        sel = msa_topk_select(max_score, args.topk)  # (total_q, Hq, topk)
        # Map the per-(qo-head) selection onto the per-kv-head q2k layout the
        # attention kernel consumes: take the first kv-group head's selection.
        q2k = sel[:, :: (Hq // Hkv), :].transpose(0, 1).contiguous().to(torch.int32)
        return msa_sparse_attention(
            q,
            k_in,
            v_in,
            q2k,
            cu_q,
            cu_k,
            causal=args.causal,
            softmax_scale=scale,
            **extra,
        )

    res = []
    for b in _resolve_backends(args):
        times = bench_gpu_time(
            fn=run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
            input_args=(b,),
        )
        res.append(_record(args, b, times, total_q=total_q, total_kv=total_kv))
    return res
