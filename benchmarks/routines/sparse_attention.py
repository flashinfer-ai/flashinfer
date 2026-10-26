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

Benchmark routines for sparse attention and KV selection:

* Minimax Sparse Attention (MSA): per-stage ops (MSAProxyScore, MSATopKSelect,
  MSASparseAttention, MSASparseDecode) plus the end-to-end MSAPipeline.
  Precision is chosen via ``--q_dtype`` / ``--kv_dtype``.
* paged_mqa_logits: the DSA indexer logits (``fp8_paged_mqa_logits`` /
  ``fp4_paged_mqa_logits``), precision chosen via ``--mqa_dtype``.
* BlockSparseAttentionWrapper / VariableBlockSparseAttentionWrapper: block-sparse
  (VSA-style) attention over a random block pattern of ``--block_density``.

Per-cc backend support comes from ``routine_cc_to_supported_backends``; on top of
that each MSA backend skips the dtype/layout combinations its library route does
not serve on the current cc (e.g. 10.7 only serves packed-NVFP4 paged KV).
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_compute_capability,
    get_device,
    print_perf_metrics,
)

BLK_KV = 128
# Planar NVFP4 page pool of the SM10x packed-NVFP4 paged-KV route
# (docs/design_docs/nvfp4_msa_paged_kv_layout.md): per page, all K data heads,
# then K scales, V data, V scales. head_dim 128 -> 64 packed bytes + 8 e4m3 scales.
_NVFP4_DATA_DIM = 64
_NVFP4_SCALE_DIM = 8
_E4M3_ONE = 0x38

# Which kernel backends each routine actually exposes.
_ROUTINE_BACKENDS = {
    "MSAProxyScore": ["cudsl"],
    "MSASparseAttention": ["cudsl"],
    "MSASparseDecode": ["cudsl", "trtllm-gen"],
    "MSAPipeline": ["cudsl"],
    "MSATopKSelect": ["cudsl"],
    "paged_mqa_logits": ["cute-dsl"],
    "BlockSparseAttentionWrapper": [
        "fa2",
        "fa3",
        "cutile",
        "vsa_sm100_blk128",
        "vsa_sm100_blk64",
        "vsa_sm120_blk64",
    ],
    "VariableBlockSparseAttentionWrapper": ["fa2", "fa3"],
}
# Backend used when a line passes no --backends.
_DEFAULT_BACKEND = {
    "paged_mqa_logits": "cute-dsl",
    "BlockSparseAttentionWrapper": "fa2",
    "VariableBlockSparseAttentionWrapper": "fa2",
}
_MSA_ROUTINES = (
    "MSAProxyScore",
    "MSASparseAttention",
    "MSASparseDecode",
    "MSAPipeline",
    "MSATopKSelect",
)


def run_sparse_attention_test(args):
    """Route a sparse-attention routine to its test function."""
    if args.kv_layout != "flat" and args.routine not in (
        "MSASparseAttention",
        "MSASparseDecode",
    ):
        raise ValueError(f"--kv_layout {args.kv_layout} unsupported for {args.routine}")
    if args.routine == "MSAProxyScore":
        return testMSAProxyScore(args)
    elif args.routine == "MSASparseAttention":
        return testMSASparseAttention(args)
    elif args.routine == "MSASparseDecode":
        return testMSASparseDecode(args)
    elif args.routine == "MSAPipeline":
        return testMSAPipeline(args)
    elif args.routine == "MSATopKSelect":
        return testMSATopKSelect(args)
    elif args.routine == "paged_mqa_logits":
        return testPagedMqaLogits(args)
    elif args.routine == "BlockSparseAttentionWrapper":
        return testBlockSparseAttentionWrapper(args)
    elif args.routine == "VariableBlockSparseAttentionWrapper":
        return testVariableBlockSparseAttentionWrapper(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_sparse_attention_args(line, parser):
    """Parse sparse-attention command line arguments."""
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=sorted({b for v in _ROUTINE_BACKENDS.values() for b in v}),
        help=(
            "Kernel implementation. MSA: 'cudsl' is the public msa_ops call "
            "(CuTe-DSL on SM12x, the CUDA C++ Blackwell backend on 10.0/10.3); "
            "MSASparseDecode also takes 'trtllm-gen' "
            "(trtllm_batch_decode_with_kv_cache block-sparse). "
            "Default: cudsl for MSA, cute-dsl for paged_mqa_logits, fa2 for the "
            "block-sparse wrappers."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of requests.")
    parser.add_argument(
        "--s_qo",
        type=int,
        default=None,
        help=(
            "Query tokens per request (default 4096); for MSASparseDecode the "
            "tokens per decode step (default 1)."
        ),
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
    parser.add_argument(
        "--force_fused",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "MSASparseDecode (cudsl): force (or forbid) the fused single-pass "
            "decode; default lets the library choose. The exact non-TopK16 SM10x "
            "decode routes require --force_fused."
        ),
    )
    parser.add_argument("--q_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--kv_dtype",
        type=str,
        default="bfloat16",
        help="KV dtype: bfloat16/float16/fp8_e4m3/nvfp4 (prefill/decode/pipeline).",
    )
    parser.add_argument(
        "--kv_layout",
        type=str,
        default="flat",
        choices=["flat", "paged", "packed"],
        help=(
            "KV storage for prefill/decode: flat varlen, a paged cache, or "
            "K/V views split from a paged cache that packs K|V per token "
            "(vLLM's layout, SM120/SM121 only). With --kv_dtype nvfp4, 'paged' "
            "is the planar packed-NVFP4 page pool of the 10.0/10.3/10.7 route."
        ),
    )
    parser.add_argument(
        "--mqa_dtype",
        type=str,
        default="fp8",
        choices=["fp8", "fp4"],
        help="paged_mqa_logits precision: fp8_paged_mqa_logits or fp4_paged_mqa_logits.",
    )
    parser.add_argument(
        "--next_n",
        type=int,
        default=1,
        help="paged_mqa_logits: query (draft) tokens per request.",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=64,
        help="paged_mqa_logits: KV block size of the fused paged cache.",
    )
    parser.add_argument(
        "--block_r",
        type=int,
        default=64,
        help="Block-sparse wrappers: query rows per block.",
    )
    parser.add_argument(
        "--block_c",
        type=int,
        default=64,
        help="Block-sparse wrappers: KV columns per block.",
    )
    parser.add_argument(
        "--block_density",
        type=float,
        default=0.5,
        help="Block-sparse wrappers: fraction of KV blocks attended per query block row.",
    )
    args = parser.parse_args(line)
    if args.backends is None:
        args.backends = [_DEFAULT_BACKEND.get(args.routine, "cudsl")]
    if args.s_qo is None:
        args.s_qo = 1 if args.routine == "MSASparseDecode" else 4096
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


# --------------------------------------------------------------------------- #
# input construction
# --------------------------------------------------------------------------- #
def _resolve_backends(args, device):
    """Drop backends the routine does not implement or the cc whitelist excludes."""
    supported = _ROUTINE_BACKENDS[args.routine]
    out = []
    for b in args.backends:
        if b in supported:
            out.append(b)
        else:
            print(
                f"[WARNING] backend {b} not implemented for {args.routine}; skipping."
            )
    out = filter_backends_by_compute_capability(out, args.routine, device)
    if not out:
        print("[ERROR] No backends to test. Exiting.")
    return out


def _msa_decline(args, backend, cc):
    """Reason the library route for ``backend`` does not serve this MSA case on
    ``cc``, or None. Only arch-determined dtype/layout splits are checked here;
    shape limits are left to the library so a bad testlist line still fails."""
    nvfp4 = args.kv_dtype == "nvfp4"
    if backend == "trtllm-gen":
        if args.kv_layout != "paged" or args.kv_dtype != args.q_dtype:
            return "trtllm-gen block-sparse decode is benchmarked on paged KV with kv_dtype == q_dtype"
        return None
    if cc[0] == 10:
        if nvfp4 and args.kv_layout != "paged":
            return f"cc {cc[0]}.{cc[1]} serves NVFP4 KV only through the paged planar route"
        if not nvfp4 and cc == (10, 7):
            return "cc 10.7 serves MSA only through the packed-NVFP4 paged-KV route"
        if args.kv_layout == "packed":
            return "packed K|V views are SM120/SM121 only (msa_ops.supports_packed_kv)"
    return None


def _cu_seqlens(batch_size, s, device):
    return torch.arange(0, (batch_size + 1) * s, s, device=device, dtype=torch.int32)


def _rand_q2k(
    batch_size,
    s_qo,
    s_kv,
    num_kv_heads,
    topk,
    device,
    *,
    causal_valid=False,
    per_request=False,
):
    """Random ascending, -1-padded block selection (msa_topk_select output format),
    shape ``(num_kv_heads, batch_size * s_qo, topk)``.

    Default: an independent random subset of all KV blocks per (head, token).
    ``causal_valid``: token i of a request sits at ``s_kv - s_qo + i`` and picks
    only blocks at or before its own, always including its own.
    ``per_request``: one selection per (head, request), always including the
    last block, shared by the request's ``s_qo`` decode tokens.
    """
    nb = -(-s_kv // BLK_KV)
    total_q = batch_size * s_qo
    rows = batch_size if per_request else total_q
    score = torch.rand(num_kv_heads, rows, nb, device=device)
    blk = torch.arange(nb, device=device)
    if causal_valid:
        pos = (s_kv - s_qo + torch.arange(s_qo, device=device)).repeat(batch_size)
        own = pos // BLK_KV
        score = score.masked_fill(blk[None, None, :] > own[None, :, None], -1.0)
        score = score.masked_fill(blk[None, None, :] == own[None, :, None], 2.0)
    elif per_request:
        score[..., nb - 1] = 2.0
    nsel = min(topk, nb)
    top = score.topk(nsel, dim=-1)
    idx = torch.where(top.values >= 0, top.indices, nb).sort(dim=-1).values
    idx = torch.where(idx == nb, -1, idx).to(torch.int32)
    out = torch.full((num_kv_heads, rows, topk), -1, dtype=torch.int32, device=device)
    out[..., :nsel] = idx
    if per_request:
        out = out.repeat_interleave(s_qo, dim=1)
    return out.contiguous()


def _paged_kv(bs, s_kv, num_kv_heads, q_dtype, kv_dtype, device, packed):
    """Paged K/V (contiguous, or views split from a packed K|V cache) plus the
    paged-call kwargs."""
    if kv_dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError("--kv_layout paged/packed excludes nvfp4")
    pages_per_req = -(-s_kv // BLK_KV)
    num_pages = bs * pages_per_req
    gen = torch.bfloat16 if q_dtype == torch.float8_e4m3fn else q_dtype
    cache = (
        torch.randn(num_pages, num_kv_heads, BLK_KV, 256, device=device, dtype=gen) / 3
    ).to(kv_dtype)
    k, v = cache.split(128, dim=-1)
    if not packed:
        k, v = k.contiguous(), v.contiguous()
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        bs, pages_per_req
    )
    seqused_k = torch.full((bs,), s_kv, dtype=torch.int32, device=device)
    return k, v, dict(page_table=page_table, seqused_k=seqused_k)


def _nvfp4_planar_kv(bs, s_kv, num_kv_heads, device):
    """K/V data + scale views of one planar packed-NVFP4 page pool (the SM10x
    route's cache layout) plus the paged-call kwargs. Data bytes are random (every
    E2M1 nibble is finite); every block scale is e4m3 1.0."""
    pages_per_req = -(-s_kv // BLK_KV)
    num_pages = bs * pages_per_req
    data_head = BLK_KV * _NVFP4_DATA_DIM
    scale_head = BLK_KV * _NVFP4_SCALE_DIM
    k_scale_off = num_kv_heads * data_head
    v_data_off = k_scale_off + num_kv_heads * scale_head
    v_scale_off = v_data_off + k_scale_off
    page_bytes = v_scale_off + num_kv_heads * scale_head
    pool = torch.randint(
        0, 256, (num_pages * page_bytes,), dtype=torch.uint8, device=device
    )
    data_shape = (num_pages, num_kv_heads, BLK_KV, _NVFP4_DATA_DIM)
    scale_shape = (num_pages, num_kv_heads, BLK_KV, _NVFP4_SCALE_DIM)
    data_stride = (page_bytes, data_head, _NVFP4_DATA_DIM, 1)
    scale_stride = (page_bytes, scale_head, _NVFP4_SCALE_DIM, 1)
    k = torch.as_strided(pool, data_shape, data_stride, 0)
    v = torch.as_strided(pool, data_shape, data_stride, v_data_off)
    k_scale = torch.as_strided(pool, scale_shape, scale_stride, k_scale_off)
    v_scale = torch.as_strided(pool, scale_shape, scale_stride, v_scale_off)
    k_scale.fill_(_E4M3_ONE)
    v_scale.fill_(_E4M3_ONE)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        bs, pages_per_req
    )
    seqused_k = torch.full((bs,), s_kv, dtype=torch.int32, device=device)
    # e2m1 magnitudes reach 6, so scale the dequant down to keep logits tame.
    return (
        k,
        v,
        dict(
            page_table=page_table,
            seqused_k=seqused_k,
            k_scale=k_scale,
            v_scale=v_scale,
            k_global_scale=1.0 / 6.0,
            v_global_scale=1.0 / 6.0,
        ),
    )


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


def _msa_kv(args, device, q_dtype, kv_dtype, cc):
    """K/V inputs + kwargs for MSA prefill/decode. Returns (k, v, extra, planar)."""
    bs, s_kv, Hkv = args.batch_size, args.s_kv, args.num_kv_heads
    if args.kv_dtype == "nvfp4" and args.kv_layout == "paged" and cc[0] == 10:
        k, v, extra = _nvfp4_planar_kv(bs, s_kv, Hkv, device)
        return k, v, extra, True
    if args.kv_layout != "flat":
        k, v, extra = _paged_kv(
            bs, s_kv, Hkv, q_dtype, kv_dtype, device, args.kv_layout == "packed"
        )
        return k, v, extra, False
    gen = torch.bfloat16 if q_dtype == torch.float8_e4m3fn else q_dtype
    k = torch.randn(bs * s_kv, Hkv, 128, device=device, dtype=gen) / 3
    v = torch.randn(bs * s_kv, Hkv, 128, device=device, dtype=gen) / 3
    k_in, v_in, extra = _maybe_quantize_kv(k, v, kv_dtype)
    return k_in, v_in, extra, False


def _rand_q(total_q, num_heads, device, dtype):
    """Random query; fp8 is drawn in bf16 then cast (randn has no fp8 kernel)."""
    gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    return (torch.randn(total_q, num_heads, 128, device=device, dtype=gen) / 3).to(
        dtype
    )


def _common(args):
    """Shared scalar setup. Per-cc support is the whitelist's job (_resolve_backends)."""
    device = get_device(args)
    if args.routine in _MSA_ROUTINES and args.head_dim != 128:
        print(f"[ERROR] MSA requires head_dim 128, got {args.head_dim}. Exiting.")
        return None
    if args.generate_repro_command:
        print(f"[INFO] To reproduce this test case, run: {args.repro_command}")
    return device


def _record(
    args,
    backend,
    times,
    *,
    tflops=0.0,
    tb_per_sec=0.0,
    total_q=0,
    total_kv=0,
    **fields,
):
    """Build one CSV result row from a list of per-iter times. ``fields``
    override columns a routine records differently from the MSA defaults (the
    CSV writer back-fills blank columns from argparse defaults, so a column
    that does not apply is written as 0 rather than left blank)."""
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
    cur.update(fields)
    return cur


def _time(args, fn, backend, use_cuda_graph=False):
    return bench_gpu_time(
        fn=fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
        use_cuda_graph=use_cuda_graph,
        input_args=(backend,),
    )


def _refcheck(args, outputs, atol=2e-2, rtol=2e-2):
    """Compare every backend's output against the first one's."""
    if not args.refcheck or len(outputs) < 2:
        return
    names = list(outputs)
    ref = outputs[names[0]].float()
    for name in names[1:]:
        out = outputs[name].float()
        ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
        if not ok:
            diff = (out - ref).abs()
            print(
                f"[REFCHECK] {name} vs {names[0]}: max abs diff {diff.max().item():.4f}, "
                f"mismatched {(diff > atol + rtol * ref.abs()).float().mean().item():.2%}"
            )
            if not args.allow_output_mismatch:
                raise AssertionError(f"[ERROR] Backend {name} output mismatch")


# --------------------------------------------------------------------------- #
# MSA per-op routines
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
    backends = _resolve_backends(args, device)
    if not backends:
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
    for b in backends:
        times = _time(args, run, b)
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


def testMSATopKSelect(args):
    """Stage 2: per-(token, head) top-k KV blocks from the proxy's block maxima."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSATopKSelect | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    backends = _resolve_backends(args, device)
    if not backends:
        return []
    from flashinfer.msa_ops import msa_topk_select

    total_q = args.batch_size * args.s_qo
    Hq = args.num_qo_heads
    max_k_tiles = args.max_k_tiles or -(-args.s_kv // BLK_KV)
    max_score = torch.randn(Hq, max_k_tiles, total_q, device=device)

    def run(_b):
        return msa_topk_select(max_score, args.topk)

    res = []
    for b in backends:
        times = _time(args, run, b)
        median = float(np.median(times))
        nbytes = max_score.numel() * 4 + total_q * Hq * args.topk * 4
        res.append(
            _record(
                args,
                b,
                times,
                tb_per_sec=nbytes / (1e9 * median),
                total_q=total_q,
                total_kv=args.batch_size * args.s_kv,
                page_size=BLK_KV,
            )
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
    backends = _resolve_backends(args, device)
    cc = get_compute_capability(device)
    for b in list(backends):
        reason = _msa_decline(args, b, cc)
        if reason:
            print(f"[WARNING] {b} skipped for {args.routine}: {reason}.")
            backends.remove(b)
    if not backends:
        return []
    from flashinfer.msa_ops import msa_sparse_attention

    bs, s_qo, s_kv = args.batch_size, args.s_qo, args.s_kv
    Hq, Hkv = args.num_qo_heads, args.num_kv_heads
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    total_q, total_kv = bs * s_qo, bs * s_kv
    q = _rand_q(total_q, Hq, device, q_dtype)
    cu_q, cu_k = _cu_seqlens(bs, s_qo, device), _cu_seqlens(bs, s_kv, device)
    k_in, v_in, extra, planar = _msa_kv(args, device, q_dtype, kv_dtype, cc)
    # The planar NVFP4 route takes causal-valid selections and seqused_k as the
    # sole KV length (it refuses cu_seqlens_k).
    q2k = _rand_q2k(bs, s_qo, s_kv, Hkv, args.topk, device, causal_valid=planar)
    if planar:
        cu_k = None
    # softmax_scale is left to the library default (1/sqrt(128)): the exact SM10x
    # non-TopK16 routes only match a call that does not pass one.
    n_sel = int((q2k >= 0).sum().item())

    def run(_b):
        return msa_sparse_attention(
            q,
            k_in,
            v_in,
            q2k,
            cu_q,
            cu_k,
            causal=args.causal,
            **extra,
        )

    res = []
    for b in backends:
        times = _time(args, run, b)
        # Selected work: one 128-token block per valid q2k entry (fewer than topk
        # for early causal tokens), each serving Hq/Hkv query heads; QK + PV are
        # two GEMMs, so 4 flops per (token, column, channel).
        flops = 4 * n_sel * BLK_KV * (Hq // Hkv) * 128
        tflops = flops / (1e9 * float(np.median(times)))
        res.append(
            _record(args, b, times, tflops=tflops, total_q=total_q, total_kv=total_kv)
        )
    return res


def _trtllm_block_sparse_inputs(q2k, page_table, bs, s_qo, s_kv):
    """MSA per-(kv head, request) selections -> trtllm-gen block-sparse page
    tables ``[Hkv, B, topk]`` (physical ids, packed ascending) and surviving KV
    lengths ``[Hkv, B]``."""
    sel = q2k[:, ::s_qo, :].long()  # one selection per request (shared by its tokens)
    valid = sel >= 0
    phys = torch.gather(
        page_table.long().unsqueeze(0).expand(sel.shape[0], -1, -1),
        2,
        sel.clamp_min(0),
    )
    block_tables = torch.where(valid, phys, 0).to(torch.int32).contiguous()
    blk_len = (s_kv - sel * BLK_KV).clamp(0, BLK_KV)
    seq_lens = torch.where(valid, blk_len, 0).sum(-1).to(torch.int32).contiguous()
    return block_tables, seq_lens, int(seq_lens.max().item())


def testMSASparseDecode(args):
    """Stage 3: decode of ``--s_qo`` tokens per request (hot path, CUDA-graph native)."""
    if args.verbose >= 1:
        print(
            f"[INFO] Running testMSASparseDecode | FlashInfer {flashinfer.__version__}"
        )
    device = _common(args)
    if device is None:
        return []
    backends = _resolve_backends(args, device)
    cc = get_compute_capability(device)
    for b in list(backends):
        reason = _msa_decline(args, b, cc)
        if reason:
            print(f"[WARNING] {b} skipped for {args.routine}: {reason}.")
            backends.remove(b)
    if not backends:
        return []
    from flashinfer.msa_ops import msa_sparse_decode_attention

    bs, s_kv = args.batch_size, args.s_kv
    seqlen_q = args.s_qo
    Hq, Hkv = args.num_qo_heads, args.num_kv_heads
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    total_q, total_kv = bs * seqlen_q, bs * s_kv
    q = _rand_q(total_q, Hq, device, q_dtype)
    q2k = _rand_q2k(bs, seqlen_q, s_kv, Hkv, args.topk, device, per_request=True)
    k_in, v_in, extra, planar = _msa_kv(args, device, q_dtype, kv_dtype, cc)
    if args.kv_layout == "flat":
        extra = dict(cu_seqlens_k=_cu_seqlens(bs, s_kv, device), **extra)
    scale = 1.0 / math.sqrt(128)
    n_sel = int((q2k >= 0).sum().item())
    n_sel_req = int((q2k[:, ::seqlen_q] >= 0).sum().item())

    runners = {}
    fused_kw = {} if args.force_fused is None else dict(force_fused=args.force_fused)
    if "cudsl" in backends:
        runners["cudsl"] = lambda: msa_sparse_decode_attention(
            q,
            k_in,
            v_in,
            q2k,
            seqlen_q=seqlen_q,
            causal=args.causal,
            **extra,
            **fused_kw,
        )
    if "trtllm-gen" in backends:
        block_tables, sparse_lens, max_len = _trtllm_block_sparse_inputs(
            q2k, extra["page_table"], bs, seqlen_q, s_kv
        )
        trt_ws = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device)
        runners["trtllm-gen"] = lambda: (
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=(k_in, v_in),
                workspace_buffer=trt_ws,
                block_tables=block_tables,
                seq_lens=sparse_lens,
                max_seq_len=max_len,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                kv_layout="HND",
                backend="trtllm-gen",
                q_len_per_req=seqlen_q,
                enable_block_sparse_attention=True,
            )
        )

    res = []
    outputs = {}
    for b in backends:
        if args.refcheck:
            outputs[b] = runners[b]().detach().clone()
        times = _time(args, lambda _b: runners[_b](), b)
        median = float(np.median(times))  # milliseconds
        flops = 4 * n_sel * BLK_KV * (Hq // Hkv) * 128
        tflops = flops / (1e9 * median)
        # Sparse decode is memory-bound: each request reads its selected K and V
        # blocks once per kv_head (its decode tokens share the selection).
        # bf16 2 B/elem, fp8 1 B, NVFP4 0.5625 B.
        if args.kv_dtype == "nvfp4":
            b_per_elem = 0.5625
        elif kv_dtype == torch.float8_e4m3fn:
            b_per_elem = 1.0
        else:
            b_per_elem = 2.0
        kv_read_bytes = 2 * n_sel_req * BLK_KV * 128 * b_per_elem
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
    _refcheck(args, outputs)
    return res


def testMSAPipeline(args):
    """End-to-end MSA prefill: proxy_score -> topk_select -> prefill."""
    if args.verbose >= 1:
        print(f"[INFO] Running testMSAPipeline | FlashInfer {flashinfer.__version__}")
    device = _common(args)
    if device is None:
        return []
    backends = _resolve_backends(args, device)
    if not backends:
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
    for b in backends:
        times = _time(args, run, b)
        res.append(_record(args, b, times, total_q=total_q, total_kv=total_kv))
    return res


# --------------------------------------------------------------------------- #
# DSA indexer logits
# --------------------------------------------------------------------------- #
def _mqa_inputs(mqa_dtype, bs, s_kv, next_n, num_heads, page_size, device):
    """Random finite inputs in the fused paged layouts the public API requires
    (as in benchmarks/bench_paged_mqa_logits.py): kv_fused is flat per block --
    [all value bytes][all scale bytes] -- not per-token [value|scale] rows."""
    ntb = flashinfer.min_block_table_width(s_kv, page_size)
    num_blocks = max(bs * ntb, 1)
    seq_lens = torch.full((bs,), s_kv, dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(bs * ntb, dtype=torch.int32, device=device)
        .view(bs, ntb)
        .contiguous()
    )
    weights = torch.randn(bs * next_n, num_heads, dtype=torch.float32, device=device)
    if mqa_dtype == "fp8":
        q = torch.randn(bs, next_n, num_heads, 128, device=device).to(
            torch.float8_e4m3fn
        )
        kv_vals = torch.randn(num_blocks, page_size, 128, device=device).to(
            torch.float8_e4m3fn
        )
        kv_scales = torch.rand(num_blocks, page_size, device=device) + 0.5
        kv = torch.cat(
            [
                kv_vals.view(torch.uint8).flatten(1),
                kv_scales.view(torch.uint8).flatten(1),
            ],
            dim=1,
        ).view(num_blocks, page_size, 1, 128 + 4)
        ref = dict(q=q, kv_vals=kv_vals, kv_scales=kv_scales, weights=weights, ntb=ntb)
        return (q, kv, weights, block_tables, seq_lens, s_kv), 132, ref
    # Every E2M1 nibble is finite; UE8M0 0x7F == 1.0 keeps the MMA finite.
    q = torch.randint(
        0, 256, (bs, next_n, num_heads, 64), dtype=torch.uint8, device=device
    )
    q_sf = torch.full(
        (bs, next_n, num_heads), 0x7F7F7F7F, dtype=torch.int32, device=device
    )
    kv_vals = torch.randint(
        0, 256, (num_blocks, page_size * 64), dtype=torch.uint8, device=device
    )
    kv_sf = torch.full(
        (num_blocks, page_size * 4), 0x7F, dtype=torch.uint8, device=device
    )
    kv = torch.cat([kv_vals, kv_sf], dim=1).view(num_blocks, page_size, 1, 64 + 4)
    ref = dict(q=q, kv_vals=kv_vals, kv_scales=None, weights=weights, ntb=ntb)
    return (q, q_sf, kv, weights, block_tables, seq_lens, s_kv), 68, ref


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _unpack_e2m1(packed):
    """uint8 (..., n) packed E2M1 (low nibble first) -> float (..., 2n). Every
    UE8M0 scale the inputs use is 1.0, so this is the full dequant."""
    lut = torch.tensor(_E2M1_VALUES, device=packed.device)
    nib = torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2).long()
    val = lut[nib & 0x7]
    return torch.where((nib & 0x8) != 0, -val, val)


def _mqa_ref_cosine_diff(mqa_dtype, ref, out, b, s_kv, next_n):
    """Torch reference (tests/attn_scores/test_attn_scores.py
    _ref_fp8/_ref_fp4_paged_mqa_logits) for request ``b`` over the API-defined
    causal region; returns the tests' cosine difference vs ``out``."""
    ntb = ref["ntb"]
    blocks = ref["kv_vals"][b * ntb : (b + 1) * ntb]
    if mqa_dtype == "fp8":
        k = blocks.float().reshape(-1, 128)[:s_kv]
        scale = ref["kv_scales"][b * ntb : (b + 1) * ntb].reshape(-1)[:s_kv]
        q = ref["q"][b].float()
    else:
        k = _unpack_e2m1(blocks.reshape(-1, 64))[:s_kv]
        scale = 1.0
        q = _unpack_e2m1(ref["q"][b])
    w = ref["weights"][b * next_n : (b + 1) * next_n]  # [next_n, H]
    s = torch.einsum("thd,kd->thk", q, k).relu()
    logits = (s * w[:, :, None]).sum(1) * scale  # [next_n, s_kv]
    t = torch.arange(next_n, device=out.device)
    mask = (
        torch.arange(s_kv, device=out.device)[None, :] <= (s_kv - next_n + t)[:, None]
    )
    x = out[b * next_n : (b + 1) * next_n, :s_kv].double()[mask]
    y = logits.double()[mask]
    denom = (x * x + y * y).sum()
    return 0.0 if denom == 0 else float(1 - 2 * (x * y).sum() / denom)


def testPagedMqaLogits(args):
    """DSA lightning-indexer logits over a paged FP8 / MXFP4 KV cache:
    ``logits[b*next_n+t, pos] = sum_h w * relu(q . k[pos])``. The schedule and
    output are prepared once, so only the kernel is timed."""
    if args.verbose >= 1:
        print(
            f"[INFO] Running testPagedMqaLogits | FlashInfer {flashinfer.__version__}"
        )
    device = _common(args)
    if device is None:
        return []
    backends = _resolve_backends(args, device)
    if not backends:
        return []
    if args.head_dim != 128:
        print(f"[ERROR] paged_mqa_logits requires head_dim 128, got {args.head_dim}.")
        return []

    bs, s_kv, next_n, H = args.batch_size, args.s_kv, args.next_n, args.num_qo_heads
    call_args, kv_bytes_per_token, ref = _mqa_inputs(
        args.mqa_dtype, bs, s_kv, next_n, H, args.page_size, device
    )
    seq_lens = call_args[-2]
    schedule = flashinfer.compute_paged_mqa_logits_schedule(
        seq_lens, device, next_n=next_n, variant=args.mqa_dtype
    )
    out_dtype = torch.float32 if args.mqa_dtype == "fp8" else torch.bfloat16
    out = torch.empty(
        bs * next_n, flashinfer.padded_seq_len(s_kv), dtype=out_dtype, device=device
    )
    fn = (
        flashinfer.fp8_paged_mqa_logits
        if args.mqa_dtype == "fp8"
        else flashinfer.fp4_paged_mqa_logits
    )

    def run(_b):
        return fn(*call_args, schedule_meta=schedule, out=out)

    res = []
    for b in backends:
        run(b)  # JIT + schedule-bucket compile outside timing
        if args.refcheck:
            rows = bs * next_n
            limits = s_kv - next_n + (torch.arange(rows, device=device) % next_n)
            defined = torch.arange(s_kv, device=device)[None, :] <= limits[:, None]
            if not torch.isfinite(out[:, :s_kv][defined]).all():
                raise AssertionError(
                    "[ERROR] paged_mqa_logits produced non-finite logits in the "
                    "API-defined output region"
                )
            # Value check on the first and last request (tests' threshold 0.02).
            for req in sorted({0, bs - 1}):
                diff = _mqa_ref_cosine_diff(args.mqa_dtype, ref, out, req, s_kv, next_n)
                if diff >= 0.02:
                    print(
                        f"[REFCHECK] {b} request {req}: cosine diff {diff:.3e} vs "
                        "torch reference"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(f"[ERROR] Backend {b} output mismatch")
        times = _time(args, run, b, use_cuda_graph=not args.no_cuda_graph)
        median = float(np.median(times))
        flops = 2 * bs * next_n * H * 128 * s_kv
        nbytes = bs * s_kv * kv_bytes_per_token + out.numel() * out.element_size()
        mqa_type = "fp8_e4m3" if args.mqa_dtype == "fp8" else "mxfp4"
        res.append(
            _record(
                args,
                b,
                times,
                tflops=flops / (1e9 * median),
                tb_per_sec=nbytes / (1e9 * median),
                total_q=bs * next_n,
                total_kv=bs * s_kv,
                q_dtype=mqa_type,
                kv_dtype=mqa_type,
                topk=0,
                page_size=args.page_size,
            )
        )
    return res


# --------------------------------------------------------------------------- #
# Block-sparse wrappers
# --------------------------------------------------------------------------- #
def _bsr_pattern(MB, NB, density, device):
    """Head-shared BSR pattern with the same block count in every row (the Sage
    FP8 path requires a uniform count)."""
    nnz = max(1, min(NB, round(density * NB)))
    idx = torch.rand(MB, NB, device=device).argsort(dim=-1)[:, :nnz].sort(-1).values
    indptr = torch.arange(0, (MB + 1) * nnz, nnz, dtype=torch.int32, device=device)
    return indptr, idx.reshape(-1).to(torch.int32).contiguous(), nnz


def testBlockSparseAttentionWrapper(args):
    """BlockSparseAttentionWrapper plan/run on one sequence (M = --s_qo,
    N = --s_kv) with an R x C block pattern of --block_density.
    ``--q_dtype fp8_e4m3`` selects the Sage-FP8 path of vsa_sm100_blk64."""
    if args.verbose >= 1:
        print(
            f"[INFO] Running testBlockSparseAttentionWrapper | FlashInfer {flashinfer.__version__}"
        )
    device = _common(args)
    if device is None:
        return []
    backends = _resolve_backends(args, device)
    if not backends:
        return []

    M, N, R, C = args.s_qo, args.s_kv, args.block_r, args.block_c
    Hq, Hkv, D = args.num_qo_heads, args.num_kv_heads, args.head_dim
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    sage = q_dtype == torch.float8_e4m3fn
    gen_dtype = torch.bfloat16 if sage else q_dtype
    q = torch.randn(M, Hq, D, device=device, dtype=gen_dtype)
    k = torch.randn(N, Hkv, D, device=device, dtype=gen_dtype)
    v = torch.randn(N, Hkv, D, device=device, dtype=gen_dtype)
    indptr, indices, nnz = _bsr_pattern(M // R, N // C, args.block_density, device)
    plan_kw = dict(q_data_type=q_dtype, kv_data_type=q_dtype, o_data_type=gen_dtype)
    if sage:
        # Sage scales: per-(head, token) q, per-(head, 16-token bucket) k,
        # per-(head, channel) v -- see tests/attention/test_vsa_block_sparse.py.
        plan_kw.update(
            q_scale=torch.full((1, Hq, M), 1.0 / 64, device=device),
            k_scale=torch.full((1, Hkv, -(-N // 16)), 1.0 / 64, device=device),
            v_scale=torch.full((Hkv, D), 1.0 / 64, device=device),
        )
        q, k, v = (x.to(torch.float8_e4m3fn) for x in (q, k, v))
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    res, outputs = [], {}
    for b in backends:
        if sage and b != "vsa_sm100_blk64":
            print(f"[WARNING] {b} has no Sage-FP8 path; skipping.")
            continue
        wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(workspace, backend=b)
        wrapper.plan(indptr, indices, M, N, R, C, Hq, Hkv, D, **plan_kw)
        if args.refcheck:
            outputs[b] = wrapper.run(q, k, v).detach().clone()
        times = _time(args, lambda _b, w=wrapper: w.run(q, k, v), b)
        median = float(np.median(times))
        flops = 4 * (M // R) * nnz * R * C * Hq * D
        # Planned non-causal; topk = KV blocks attended per query block row.
        res.append(
            _record(
                args,
                b,
                times,
                tflops=flops / (1e9 * median),
                total_q=M,
                total_kv=N,
                causal=False,
                topk=nnz,
                page_size=0,
            )
        )
    _refcheck(args, outputs)
    return res


def testVariableBlockSparseAttentionWrapper(args):
    """VariableBlockSparseAttentionWrapper plan/run with uniform --block_r x
    --block_c blocks (last block takes the remainder) and a random per-kv-head
    block mask of --block_density. q/k/v are HND."""
    if args.verbose >= 1:
        print(
            "[INFO] Running testVariableBlockSparseAttentionWrapper | "
            f"FlashInfer {flashinfer.__version__}"
        )
    device = _common(args)
    if device is None:
        return []
    backends = _resolve_backends(args, device)
    if not backends:
        return []

    M, N = args.s_qo, args.s_kv
    Hq, Hkv, D = args.num_qo_heads, args.num_kv_heads, args.head_dim
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)

    def sizes(total, blk):
        n = -(-total // blk)
        sz = torch.full((n,), blk, dtype=torch.int32)
        sz[-1] = total - blk * (n - 1)
        return sz.unsqueeze(0).repeat(Hkv, 1)

    block_row_sz, block_col_sz = sizes(M, args.block_r), sizes(N, args.block_c)
    mask = (
        torch.rand(Hkv, block_row_sz.shape[1], block_col_sz.shape[1])
        < args.block_density
    )
    q = torch.randn(Hq, M, D, device=device, dtype=q_dtype)
    k = torch.randn(Hkv, N, D, device=device, dtype=q_dtype)
    v = torch.randn(Hkv, N, D, device=device, dtype=q_dtype)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    # Attended (row, col) token pairs per kv head, summed over heads.
    attended = (
        block_row_sz.unsqueeze(2).float() * block_col_sz.unsqueeze(1).float() * mask
    ).sum()

    res, outputs = [], {}
    for b in backends:
        wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
            workspace, backend=b
        )
        wrapper.plan(
            block_mask_map=mask,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=Hq,
            num_kv_heads=Hkv,
            head_dim=D,
            q_data_type=q_dtype,
        )
        if args.refcheck:
            outputs[b] = wrapper.run(q, k, v).detach().clone()
        times = _time(args, lambda _b, w=wrapper: w.run(q, k, v), b)
        median = float(np.median(times))
        flops = 4 * attended.item() * (Hq // Hkv) * D
        # Planned non-causal; the per-row block count varies, so topk is n/a (0).
        res.append(
            _record(
                args,
                b,
                times,
                tflops=flops / (1e9 * median),
                total_q=M,
                total_kv=N,
                causal=False,
                topk=0,
                page_size=0,
            )
        )
    _refcheck(args, outputs)
    return res
