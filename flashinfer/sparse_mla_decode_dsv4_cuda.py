"""CUDA (tvm_ffi) fast path for DeepSeek-V4.1 sparse MLA decode on the
bf16-dense-KV ABI: bf16 varlen queries ``[T, 16, 512]``, bf16 SWA pool
``[swa_pages, 256, 512]`` and bf16 compressed pool ``[ckv_pages, 64, 512]``
gathered through one combined int32 STORAGE-OFFSET table ``[T, 640]``
(columns 0-127 address the SWA pool, 128-639 the compressed pool; ``-1``
holes and tail padding masked via ``sparse_topk_lens``), fp32 sinks
``[16]``, softmax scale ``512**-0.5`` fused with ``bmm2_scale == 1.0``.

Scope: this exact geometry, scale and encoding on SM103, for the captured
MTP/verify decode classes (``max_q_len == 6``, ``T == 6 * batch_size``,
``batch_size`` 1-13 — the allowlist JSON next to this package); everything
else stays on the stock trtllm_batch_decode_sparse_mla_dsv4 dispatch. Kill
switch: FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1.

Surface note: the "specialized" value lives on the vendor-named entry's
backend enum as a codified special-case exception to the
no-specialized-on-vendor-entries rule, because the backend-neutral public
name ``batch_decode_sparse_mla_dsv4`` is a true object-identity alias of
that entry (pinned by an upstream compatibility test), so both names
carry the backend set unchanged.

CUDA graphs: the kernel launches on the current stream re-queried per call
and allocates nothing (no workspace: every route is a single fused launch;
cluster merges run through distributed shared memory), but its FIRST call
runs one-time cudaFuncSetAttribute + occupancy probes — warm up eagerly
once before capture. Under capture the route is taken only after that
eager warm-up; dispatch is host-int based with no device syncs.

Semantics notes (documented, verified by the integration test suite):
- The kernel consumes the per-token sparse table and ``sparse_topk_lens``
  exactly; ``seq_lens`` / ``cum_seq_lens_q`` / ``max_q_len`` are accepted
  and not re-read (on this contract the table builder already encodes the
  SWA validity window, and the allowlist keys on ``max_q_len`` and the
  token count). ``workspace_buffer`` is not used by the specialized route.
- ``bmm1_scale`` / ``bmm2_scale`` must be the contract's float values
  (baked into the kernel); tensor scales fall back.

Generated-kernel provenance: agent-generated kernel (search-campaign
winner), source-vetted (overflow-safe online-max softmax with fp32 cluster
merges; terminal cluster barriers after all remote reads; runtime-guarded
token-count dispatch over a generic cluster kernel with an in-source
single-rank fallback; -1/length masking applied in-softmax) and re-timed
per workload row on GB300 against the stock trtllm-gen path of this API.
"""

import functools
import json as _json
import os as _os
from pathlib import Path as _Path

import torch

from .jit.sparse_mla_decode_dsv4 import gen_sparse_mla_decode_dsv4_module

_SCALE = 0.044194173824159216  # 512 ** -0.5 (baked into the kernel)
KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISPATCH_STATS = {"specialized": 0, "fallback": 0}
LAST_FALLBACK_REASONS: list = []
_BUILT = {"ready": False}
_WARMED: set = set()  # device indices warmed before capture


@functools.cache
def _allowlist():
    p = _Path(__file__).parent / "sparse_mla_decode_dsv4_sm103_allowlist.json"
    with open(p) as f:
        al = _json.load(f)
    return {
        (k["batch_size"], k["max_q_len"], k["num_tokens"])
        for k in al["auto_route_keys"]
    }


@functools.cache
def get_sparse_mla_decode_dsv4_module():
    mod = gen_sparse_mla_decode_dsv4_module().build_and_load()
    _BUILT["ready"] = True
    return mod


def supported_with_reasons(
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    cum_seq_lens_q,
    max_q_len,
    sinks,
    bmm1_scale,
    bmm2_scale,
    out,
    extras,
):
    """Guard order: GPU-free (kill switch, API-surface flags, dtype,
    geometry, allowlist) -> layout guards -> device guards. Returns []
    when the specialized kernel can serve the call, else the fallback
    reasons. No device syncs on this path."""
    if _os.environ.get(KILL_SWITCH_ENV, "0") == "1":
        return ["kill_switch:" + KILL_SWITCH_ENV + "=1"]
    r = []
    for name, ok in (
        ("varlen_query", cum_seq_lens_q is not None and max_q_len is not None),
        (
            "combined_table",
            sparse_indices is not None
            and sparse_topk_lens is not None
            and seq_lens is not None
            and extras.get("swa_topk_lens") is None
            and extras.get("extra_sparse_indices") is None
            and extras.get("extra_sparse_topk_lens") is None,
        ),
        (
            "storage_offset_indices",
            extras.get("sparse_indices_are_storage_offsets") is True,
        ),
        ("no_remap_buffer", extras.get("remapped_sparse_indices_buffer") is None),
        (
            "no_rope_quant",
            extras.get("dsv4_inv_rope_cos_sin_cache") is None
            and extras.get("dsv4_output_scale") is None,
        ),
        (
            "no_hca_args",
            extras.get("hca_swa_indices") is None
            and extras.get("hca_compressed_block_tables") is None
            and extras.get("hca_seq_lens") is None
            and extras.get("hca_sparse_indices_format") is None
            and not extras.get("hca_use_persistent"),
        ),
        ("kv_layout_hnd", extras.get("kv_layout") == "HND"),
        ("kv_cache_format_default", extras.get("kv_cache_format") == "fp8"),
        ("topk_lens_offset_0", extras.get("sparse_topk_lens_offset") == 0),
        (
            "float_scales",
            isinstance(bmm1_scale, float) and isinstance(bmm2_scale, float),
        ),
        ("sinks_present", sinks is not None),
    ):
        if not ok:
            r.append(name)
    if r:
        return r
    if abs(bmm1_scale - _SCALE) > 1e-7 or bmm2_scale != 1.0:
        r.append("scale_mismatch")
    if (
        query.dim() != 3
        or query.shape[0] == 0
        or query.shape[1] != 16
        or query.shape[2] != 512
        or query.dtype != torch.bfloat16
    ):
        r.append("q_not_varlen_bf16_16x512")
        return r
    T = query.shape[0]
    B = seq_lens.shape[0] if seq_lens.dim() == 1 else -1
    if (B, int(max_q_len), T) not in _allowlist():
        r.append("shape_class_not_allowlisted")
    if (
        swa_kv_cache.dim() != 3
        or swa_kv_cache.shape[1] != 256
        or swa_kv_cache.shape[2] != 512
        or swa_kv_cache.dtype != torch.bfloat16
    ):
        r.append("swa_pool_not_bf16_256x512")
    if (
        compressed_kv_cache is None
        or compressed_kv_cache.dim() != 3
        or compressed_kv_cache.shape[1] != 64
        or compressed_kv_cache.shape[2] != 512
        or compressed_kv_cache.dtype != torch.bfloat16
    ):
        r.append("compressed_pool_not_bf16_64x512")
    if (
        sparse_indices.dim() != 2
        or sparse_indices.shape[0] != T
        or sparse_indices.shape[1] != 640
        or sparse_indices.dtype != torch.int32
    ):
        r.append("indices_not_Tx640_int32")
    if (
        sparse_topk_lens.dim() != 1
        or sparse_topk_lens.shape[0] != T
        or sparse_topk_lens.dtype != torch.int32
    ):
        r.append("topk_lens_not_T_int32")
    if seq_lens.dim() != 1 or seq_lens.dtype != torch.int32:
        r.append("seq_lens_not_B_int32")
    if (
        cum_seq_lens_q.dim() != 1
        or cum_seq_lens_q.shape[0] != B + 1
        or cum_seq_lens_q.dtype != torch.int32
    ):
        r.append("cum_seq_lens_q_not_Bp1_int32")
    if sinks.dim() != 1 or sinks.shape[0] != 16 or sinks.dtype != torch.float32:
        r.append("sinks_not_16_fp32")
    if r:
        return r
    # layout guards
    for name, t in (
        ("q", query),
        ("swa_pool", swa_kv_cache),
        ("compressed_pool", compressed_kv_cache),
        ("indices", sparse_indices),
        ("topk_lens", sparse_topk_lens),
        ("seq_lens", seq_lens),
        ("cum_seq_lens_q", cum_seq_lens_q),
        ("sinks", sinks),
    ):
        if not t.is_contiguous():
            r.append(name + "_not_contiguous")
        if t.device != query.device:
            r.append(name + "_device_mismatch")
        if t.data_ptr() % 16:
            r.append(name + "_not_16B_aligned")
    if out is not None and (
        out.dtype != torch.bfloat16
        or out.data_ptr() % 16
        or out.shape != query.shape
        or not out.is_contiguous()
        or out.device != query.device
    ):
        r.append("out_not_bf16_q_shaped_contiguous")
    if r:
        return r
    # device guards
    if torch.cuda.get_device_capability(query.device) != (10, 3):
        return ["not_sm103"]
    if torch.cuda.is_current_stream_capturing() and query.device.index not in _WARMED:
        return ["not_warmed_before_graph_capture"]
    return []


def maybe_dispatch_specialized(
    backend,
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    cum_seq_lens_q,
    max_q_len,
    sinks,
    bmm1_scale,
    bmm2_scale,
    out,
    extras,
):
    """Route for backend in ("auto", "specialized"). Returns the output
    when served; None when "auto" should continue on the stock dispatch.
    An explicitly requested "specialized" backend is never silently
    swapped: if it cannot serve the call it raises."""
    reasons = supported_with_reasons(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        cum_seq_lens_q,
        max_q_len,
        sinks,
        bmm1_scale,
        bmm2_scale,
        out,
        extras,
    )
    LAST_FALLBACK_REASONS[:] = reasons
    if not reasons:
        try:
            get_sparse_mla_decode_dsv4_module()
        except Exception as e:  # pragma: no cover - environment-dependent
            if backend == "specialized":
                raise
            LAST_FALLBACK_REASONS[:] = [
                "specialized_module_unavailable:" + type(e).__name__
            ]
            DISPATCH_STATS["fallback"] += 1
            return None
        DISPATCH_STATS["specialized"] += 1
        return _run_specialized(
            query,
            swa_kv_cache,
            compressed_kv_cache,
            sparse_indices,
            sparse_topk_lens,
            seq_lens,
            cum_seq_lens_q,
            max_q_len,
            sinks,
            out,
        )
    if backend == "specialized":
        raise ValueError(
            "backend='specialized' cannot serve this call and is never "
            f"silently swapped; fallback_reasons={reasons}"
        )
    DISPATCH_STATS["fallback"] += 1
    return None


def _run_specialized(
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    cum_seq_lens_q,
    max_q_len,
    sinks,
    out,
):
    if out is None:
        out = torch.empty_like(query)
    mod = get_sparse_mla_decode_dsv4_module()
    mod.sparse_mla_decode_dsv4_run(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        cum_seq_lens_q,
        int(max_q_len),
        sinks,
        out,
    )
    if not torch.cuda.is_current_stream_capturing():
        _WARMED.add(query.device.index)
    return out
