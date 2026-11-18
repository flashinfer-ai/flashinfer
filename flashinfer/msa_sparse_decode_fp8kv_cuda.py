"""CUDA (tvm_ffi) fast path for block-sparse MSA decode: bf16 Q over a
paged fp8-E4M3 KV cache, 16 query heads sharing 1 KV head, head_dim 128,
sparse block == KV page == 128 tokens, 16 selected blocks per query token,
single-token decode (seqlen_q 1).

Scope: this exact geometry and softmax scale on SM103; everything else stays
on the stock msa_sparse_decode_attention dispatch. q2k_indices rows follow
the documented producer contract: valid block ids ascending, padded with -1
at the tail only. An interior -1 (a hole before a valid id) is out of
contract and the selection it produces is undefined; the kernels count valid
entries and consume them as a prefix, and no per-element device-side
validation pass is run on the hot path. Kill switch:
FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1. CUDA graphs: the route is taken
under capture only after the module was built eagerly (single fatbin, no
per-shape JIT); dispatch is host-int based with no syncs, and the kernel
needs no workspace (no GMEM partials: every route is single-launch).

Generated-kernel provenance: agent-generated kernel (search-campaign
winner), source-vetted (runtime-guarded shape dispatch over a general
split-K fallback family; page table honored on every route; online
max-subtracted softmax; K and V both dequantized) and re-timed per workload
row on GB300 against both existing FlashInfer forms of this op (bf16-Q and
fp8-Q fused) and a vendored Triton baseline.
"""

import math
import os as _os
import threading as _threading

import torch

from .jit.msa_sparse_decode_fp8kv import gen_msa_sparse_decode_fp8kv_module

_SCALE = 0.08838834764831845  # 128 ** -0.5 (the geometry this path serves)
KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISPATCH_STATS = {"specialized": 0, "fallback": 0}
LAST_FALLBACK_REASONS: list = []
_BUILT = {"ready": False}


_MODULE = None
_MODULE_ERROR = None  # "TypeName: message" string; deliberately traceback-free
_MODULE_LOCK = _threading.Lock()


def get_msa_sparse_decode_fp8kv_module():
    """Build-and-load exactly once per process. functools.cache would not
    remember a *failed* build, so every later backend="auto" call would
    repeat the JIT compile attempt (seconds per decode step). The failure is
    remembered as a traceback-free string (re-raising one cached exception
    object appends caller frames to its __traceback__ on every raise,
    pinning those frames' tensors) and a fresh RuntimeError is raised per
    request; the lock keeps concurrent cold callers from compiling twice."""
    global _MODULE, _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE
    if _MODULE_ERROR is not None:
        raise RuntimeError(
            "specialized module build failed previously: " + _MODULE_ERROR
        )
    with _MODULE_LOCK:
        if _MODULE is not None:
            return _MODULE
        if _MODULE_ERROR is not None:
            raise RuntimeError(
                "specialized module build failed previously: " + _MODULE_ERROR
            )
        try:
            mod = gen_msa_sparse_decode_fp8kv_module().build_and_load()
        except Exception as exc:
            _MODULE_ERROR = f"{type(exc).__name__}: {exc}"
            raise
        _MODULE = mod
    _BUILT["ready"] = True
    return _MODULE


def supported_with_reasons(
    q, k, v, q2k_indices, page_table, seqused_k, softmax_scale, out, extras
):
    """Guard order: GPU-free (kill switch, API-surface flags, dtype,
    geometry allowlist) -> layout guards -> device guards. Returns [] when
    the specialized kernel can serve the call, else fallback reasons. No
    device syncs."""
    if _os.environ.get(KILL_SWITCH_ENV, "0") == "1":
        return ["kill_switch:" + KILL_SWITCH_ENV + "=1"]
    r = []
    for name, ok in (
        ("seqlen_q_1", extras.get("seqlen_q") == 1),
        ("causal", bool(extras.get("causal"))),
        ("no_lse", not extras.get("return_softmax_lse")),
        (
            "paged_kv",
            page_table is not None
            and seqused_k is not None
            and extras.get("cu_seqlens_k") is None,
        ),
        (
            "no_kv_block_scales",
            extras.get("k_scale") is None and extras.get("v_scale") is None,
        ),
        (
            "no_global_scales",
            extras.get("k_global_scale") is None
            and extras.get("v_global_scale") is None,
        ),
        ("no_q_offset", extras.get("q_offset") is None),
        ("no_partial_dtype", extras.get("partial_dtype") is None),
        ("no_force_fused", extras.get("force_fused") is None),
        ("no_workspace", extras.get("workspace") is None),
    ):
        if not ok:
            r.append(name)
    if r:
        return r
    if (
        q.dim() != 3
        or q.shape[0] == 0
        or q.shape[1] != 16
        or q.shape[2] != 128
        or q.dtype != torch.bfloat16
        or not q.is_contiguous()
    ):
        # Return before anything below reads q.shape[0]: a wrong-rank query
        # must reach stock validation (its ValueError), not raise IndexError
        # here, and a zero-token batch must not be "served" as an empty
        # success the stock path would reject.
        r.append("q_geometry_not_bf16_16x128")
        return r
    if (
        k.dim() != 4
        or k.shape[0] == 0
        or k.shape[1] != 1
        or k.shape[2] != 128
        or k.shape[3] != 128
        or k.dtype != torch.float8_e4m3fn
        or not k.is_contiguous()
    ):
        r.append("k_not_paged_fp8e4m3_1x128x128")
    if (
        v.dim() != 4
        or v.shape != k.shape
        or v.dtype != torch.float8_e4m3fn
        or not v.is_contiguous()
    ):
        r.append("v_not_matching_fp8e4m3")
    if (
        q2k_indices.dim() != 3
        or q2k_indices.shape[0] != 1
        or q2k_indices.shape[1] != q.shape[0]
        or q2k_indices.shape[2] != 16
        or q2k_indices.dtype != torch.int32
        or not q2k_indices.is_contiguous()
    ):
        r.append("q2k_indices_not_1xBx16_int32")
    if (
        page_table.dim() != 2
        or page_table.shape[0] != q.shape[0]
        or page_table.dtype != torch.int32
        or not page_table.is_contiguous()
    ):
        r.append("page_table_geometry")
    if (
        seqused_k.dim() != 1
        or seqused_k.shape[0] != q.shape[0]
        or seqused_k.dtype != torch.int32
        or not seqused_k.is_contiguous()
    ):
        r.append("seqused_k_geometry")
    if softmax_scale is not None and (
        not math.isfinite(softmax_scale) or abs(softmax_scale - _SCALE) > 1e-7
    ):
        r.append("scale_mismatch")
    if out is not None and (
        out.dtype != torch.bfloat16
        or out.shape != q.shape
        or not out.is_contiguous()
        or out.device != q.device
    ):
        r.append("out_not_bf16_q_shaped_contiguous")
    if q.device.type != "cuda":
        return ["q_not_cuda"]
    for _name, _t in (
        ("k", k),
        ("v", v),
        ("q2k_indices", q2k_indices),
        ("page_table", page_table),
        ("seqused_k", seqused_k),
    ):
        if _t.device != q.device:
            r.append(_name + "_device_mismatch")
    for _name, _t, _align in (("q", q, 16), ("k", k, 32), ("v", v, 32)):
        if _t.data_ptr() % _align:
            r.append(_name + "_not_%dB_aligned" % _align)
    if out is not None:
        if out.data_ptr() % 16:
            r.append("out_not_16B_aligned")
        _o0 = out.data_ptr()
        _o1 = _o0 + out.numel() * out.element_size()
        for _nm, _t in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("q2k_indices", q2k_indices),
            ("page_table", page_table),
            ("seqused_k", seqused_k),
        ):
            _t0 = _t.data_ptr()
            _t1 = _t0 + _t.numel() * _t.element_size()
            if out.device == _t.device and _o0 < _t1 and _t0 < _o1:
                r.append("out_overlaps_" + _nm)
    if r:
        return r
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        return ["not_sm103"]
    # Query capture state on the device that will run the kernel: the
    # ambient current device can differ from q.device, and its capture
    # state is the wrong answer for both directions of this guard.
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing() and not _BUILT["ready"]:
            return ["module_not_built_before_graph_capture"]
    return []


def maybe_dispatch_specialized(
    backend, q, k, v, q2k_indices, page_table, seqused_k, softmax_scale, out, extras
):
    """Route for backend in ("auto", "specialized"). Returns the output when
    served; None when "auto" should continue on the stock dispatch. An
    explicitly requested "specialized" backend is never silently swapped: if
    it cannot serve the call it raises."""
    reasons = supported_with_reasons(
        q, k, v, q2k_indices, page_table, seqused_k, softmax_scale, out, extras
    )
    LAST_FALLBACK_REASONS[:] = reasons
    if not reasons:
        try:
            get_msa_sparse_decode_fp8kv_module()
        except Exception as e:  # pragma: no cover - environment-dependent
            if backend == "specialized":
                raise
            LAST_FALLBACK_REASONS[:] = [
                "specialized_module_unavailable:" + type(e).__name__
            ]
            DISPATCH_STATS["fallback"] += 1
            return None
        DISPATCH_STATS["specialized"] += 1
        return _run_specialized(q, k, v, q2k_indices, page_table, seqused_k, out)
    if backend == "specialized":
        raise ValueError(
            "backend='specialized' cannot serve this call and is never "
            f"silently swapped; fallback_reasons={reasons}"
        )
    DISPATCH_STATS["fallback"] += 1
    return None


def _run_specialized(q, k, v, q2k_indices, page_table, seqused_k, out):
    if out is None:
        out = torch.empty_like(q)
    mod = get_msa_sparse_decode_fp8kv_module()
    mod.msa_sparse_decode_fp8kv_run(q, k, v, q2k_indices, page_table, seqused_k, out)
    return out
