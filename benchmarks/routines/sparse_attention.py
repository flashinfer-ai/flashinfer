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

Benchmark routines for Minimax Sparse Attention (MSA) on SM120 / SM121.

MSA runs as a pipeline of stage ops rather than a single fused call:
    proxy_score -> topk_select -> build_k2q_csr(_schedule) -> kv-major prefill
                                                              \\-> decode
This module exposes one routine per stage plus an end-to-end ``MSAPipeline``
routine, all timed with CUPTI via ``bench_gpu_time``.

The ``--backends`` flag selects the kernel implementation, not a vendor library:
``cudsl`` (the default CuTe-DSL path), ``cuda`` (the retained CUDA reference,
available for ``MSATopkSelect`` / ``MSABuildCsr``), and ``cudsl_radix`` (the
multi-stage radix top-k, the CuTe-DSL default for ``MSATopkSelect``). Ops with a
single implementation ignore unsupported backends with a warning.
"""

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
    "MSATopkSelect": ["cudsl_radix", "cuda"],
    "MSABuildCsr": ["cudsl", "cuda"],
    "MSASparseAttentionKvMajor": ["cudsl"],
    "MSASparseDecode": ["cudsl"],
    "MSAPipeline": ["cudsl"],
}


def run_sparse_attention_test(args):
    """Route an MSA routine to its test function."""
    if args.routine == "MSAProxyScore":
        return testMSAProxyScore(args)
    elif args.routine == "MSATopkSelect":
        return testMSATopkSelect(args)
    elif args.routine == "MSABuildCsr":
        return testMSABuildCsr(args)
    elif args.routine == "MSASparseAttentionKvMajor":
        return testMSASparseAttentionKvMajor(args)
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
        choices=["cudsl", "cuda", "cudsl_radix"],
        help="Kernel implementation(s) to test. Default: cudsl.",
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
        "--causal", action="store_true", default=True, help="Causal masking."
    )
    parser.add_argument("--q_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--kv_dtype",
        type=str,
        default="bfloat16",
        help="KV dtype: bfloat16/float16/fp8_e4m3/nvfp4 (kvmajor/decode/pipeline).",
    )
    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


# --------------------------------------------------------------------------- #
# input construction
# --------------------------------------------------------------------------- #
def _resolve_backends(args):
    """Drop backends the routine does not implement, warning once each."""
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
    nb = s_kv // BLK_KV
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
        return k, v, {}
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
    """Stage 1: dense per-KV-block max logits (msa_proxy_score)."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSAProxyScore | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import msa_proxy_score

    bs, s_qo, s_kv = args.batch_size, args.s_qo, args.s_kv
    Hq, Hkv = args.num_qo_heads, args.num_kv_heads
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    total_q, total_kv = bs * s_qo, bs * s_kv
    q = torch.randn(total_q, Hq, 128, device=device, dtype=q_dtype) / 3
    k = torch.randn(total_kv, Hkv, 128, device=device, dtype=q_dtype) / 3
    cu_q, cu_k = _cu_seqlens(bs, s_qo, device), _cu_seqlens(bs, s_kv, device)

    def run(_b):
        return msa_proxy_score(
            q, k, cu_q, cu_k, causal=args.causal, max_k_tiles=args.max_k_tiles
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
        # dense QK over the proxy: 2 * total_q * total_kv * Hq * head_dim (per request)
        flops = 2 * bs * s_qo * s_kv * Hq * 128
        tflops = flops / (1e9 * float(np.median(times)))
        res.append(
            _record(args, b, times, tflops=tflops, total_q=total_q, total_kv=total_kv)
        )
    return res


def testMSATopkSelect(args):
    """Stage 2: select top-K KV blocks per (query, head) (msa_topk_select).

    Supports cudsl_radix / cuda backends so this routine doubles as the
    radix-vs-CUDA top-k perf comparator."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSATopkSelect | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import msa_topk_select

    bs, s_qo, s_kv = args.batch_size, args.s_qo, args.s_kv
    Hq = args.num_qo_heads
    total_q = bs * s_qo
    mkt = args.max_k_tiles or -(-s_kv // BLK_KV)
    max_score = torch.randn(Hq, mkt, total_q, device=device, dtype=torch.float32)

    def run(b):
        return msa_topk_select(max_score, args.topk, _backend=b)

    res = []
    backends = _resolve_backends(args)
    outputs = {}
    for b in backends:
        if args.refcheck:
            try:
                outputs[b] = run(b).clone()
            except Exception as e:
                print(f"[WARNING] {b} topk failed (not yet implemented?): {e}")
                continue
        times = bench_gpu_time(
            fn=run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
            input_args=(b,),
        )
        res.append(_record(args, b, times, total_q=total_q, total_kv=bs * s_kv))

    # cross-backend refcheck: compare the *scores* of the selected blocks, not
    # their indices. Exact ties at the top-k boundary are broken arbitrarily (by
    # atomic emission order) and legitimately differ across backends, but every
    # valid top-k selects the same multiset of scores, so sorted selected-scores
    # must match bit-exact.
    def _selected_scores(out):
        # out (total_q, H, topk) int32 indices (-1 padded) -> sorted scores
        msp = max_score.permute(2, 0, 1)  # (total_q, H, mkt)
        g = torch.gather(msp, 2, out.long().clamp_min(0))
        g = torch.where(out >= 0, g, torch.full_like(g, float("-inf")))
        return g.sort(dim=-1, descending=True).values

    if args.refcheck and len(outputs) > 1:
        ref_b = next(iter(outputs))
        ref_scores = _selected_scores(outputs[ref_b])
        for b, out in outputs.items():
            if b != ref_b and not torch.equal(_selected_scores(out), ref_scores):
                msg = f"[ERROR] topk backend {b} disagrees with {ref_b}"
                if args.allow_output_mismatch:
                    print(msg)
                else:
                    raise AssertionError(msg)
    return res


def testMSABuildCsr(args):
    """Stage 2: invert q->k selection into a KV-major CSR schedule."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSABuildCsr | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import msa_build_k2q_csr_schedule

    bs, s_qo, s_kv = args.batch_size, args.s_qo, args.s_kv
    Hkv = args.num_kv_heads
    cu_q, cu_k = _cu_seqlens(bs, s_qo, device), _cu_seqlens(bs, s_kv, device)
    q2k = _rand_q2k(bs, s_qo, s_kv, Hkv, args.topk, device)

    def run(b):
        return msa_build_k2q_csr_schedule(q2k, cu_q, cu_k, BLK_KV, _backend=b)

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
        res.append(_record(args, b, times, total_q=bs * s_qo, total_kv=bs * s_kv))
    return res


def testMSASparseAttentionKvMajor(args):
    """Stage 3: KV-major sparse prefill (hot path); auto-builds CSR + combine."""
    if args.verbose >= 1:
        print(
            f"[INFO] Running testMSASparseAttentionKvMajor | FlashInfer {flashinfer.__version__}"
        )
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import msa_sparse_attention_kvmajor

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
        return msa_sparse_attention_kvmajor(
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
        # selected work: ~ total_q * topk blocks of 128 tokens; QK + PV ~ 4 flops/MAC
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
        flops = 4 * total_q * args.topk * BLK_KV * Hq * 128
        tflops = flops / (1e9 * float(np.median(times)))
        res.append(
            _record(args, b, times, tflops=tflops, total_q=total_q, total_kv=total_kv)
        )
    return res


def testMSAPipeline(args):
    """End-to-end MSA prefill: proxy_score -> topk_select -> kvmajor."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSAPipeline | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    from flashinfer.msa_ops import (
        msa_proxy_score,
        msa_sparse_attention_kvmajor,
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
    # proxy uses unquantized K (the cheap proxy pass)
    proxy_k = k

    def run(_b):
        max_score = msa_proxy_score(q, proxy_k, cu_q, cu_k, causal=args.causal)
        sel = msa_topk_select(max_score, args.topk)  # (total_q, Hq, topk)
        # map per-(qo-head) selection onto the per-kv-head q2k layout the
        # attention kernel consumes: take the first kv-group head's selection.
        q2k = sel[:, :: (Hq // Hkv), :].transpose(0, 1).contiguous().to(torch.int32)
        return msa_sparse_attention_kvmajor(
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
