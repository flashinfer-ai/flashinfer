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

Specialized SM90 kernel for batch-decode MLA attention with an fp8 (e4m3)
query over an fp8 (e4m3) packed paged compressed-KV cache
(csrc/mla_decode_fp8_paged_sm90.cu,
flashinfer/jit/mla_decode_fp8_paged_sm90.py).

The stock MLA backends have no fp8-Q form on SM90 (fp8 q x fp8 KV MLA is an
SM100-class CUTLASS form). ``plan(..., q_data_type=float8_e4m3fn,
kv_data_type=float8_e4m3fn)`` on an SM90 device therefore plans the stock
backend for the exact bf16-dequantised query (bf16 holds every e4m3 value
exactly) and, when the plan matches the geometry below, prepares this
kernel. ``run()`` then dispatches it; every other fp8-Q call — and every
call while ``FLASHINFER_DISABLE_MLA_DECODE_FP8_PAGED_SM90=1`` (read at plan
and run time) — dequantises Q and takes the stock path untouched.

Supported contract (fails closed to the dequant + stock path otherwise):
8 query heads, head_dim_ckv 512 + head_dim_kpe 64 (packed 576-byte rows),
page size 64, causal, UNIFORM per-request q_len <= 6 (the measured
spec/MTP-decode widths; the serve contract is q_len 6), fp8e4m3 packed (or
zero-copy adjacent split) query and KV cache, bf16 out, per-tensor float
Q/KV descales, and the measured envelope (batch <= 128, kv_len <= 262144).
The guard is an UNSCOPED configuration guard: no per-shape competitive
carve-outs — backend choice between engines is per-deployment, not
per-shape.

PRECISION (part of the API contract): this kernel computes all-fp8
(fp8 q x fp8 KV end-to-end, the FlashMLA dense-fp8 arithmetic class), NOT
the stock wrapper's bf16-q-over-dequantised-KV arithmetic. It is
deterministic (atomics-free, bit-stable); its measured record
(allowance-row per-element error profile, fresh seeds, magnitude
perturbations, bit-determinism) accompanies the change.

All allocation, compilation and device queries happen at plan time, outside
CUDA-graph capture; run() only launches on the current stream. The scalar
Q/KV descales are written into plan-owned device buffers at run time; under
CUDA-graph capture the values present at capture are baked into the graph
(the same semantics as the baked sm_scale).
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch

from ...jit.core import logger
from ...utils import get_compute_capability

_DISABLE_ENV = "FLASHINFER_DISABLE_MLA_DECODE_FP8_PAGED_SM90"
_NUM_HEADS = 8
_HEAD_DIM_CKV = 512
_HEAD_DIM_KPE = 64
_DIMQ = _HEAD_DIM_CKV + _HEAD_DIM_KPE
_PAGE_SIZE = 64
_MAX_Q_LEN = 6
_MAX_BATCH = 128
_MAX_KV_LEN = 262144
_COMPUTE_CAPABILITY = (9, 0)

_counters: Dict[str, int] = {
    "plans_seen": 0,
    "plans_disabled": 0,
    "plans_ineligible": 0,
    "plans_prepared": 0,
    "build_failed": 0,
    "workspace_allocs": 0,
    "runs_dispatched": 0,
    "runs_dispatched_capturing": 0,
    "runs_disabled": 0,
    "runs_ineligible": 0,
    "runs_dequant_fallback": 0,
}
_module: Any = None
_module_error: Optional[str] = None
_init_devices: set = set()
_announced = False


def _disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "") not in ("", "0")


def _mla_dec_fp8q_sm90_stats() -> Dict[str, Any]:
    """Introspection for benchmarks and tests: per-process plan/run census
    and build state of the specialized SM90 fp8-Q MLA decode kernel."""
    stats: Dict[str, Any] = dict(_counters)
    stats["module_loaded"] = _module is not None
    stats["module_error"] = _module_error
    return stats


def _reset_mla_dec_fp8q_sm90_stats() -> None:
    global _announced
    for key in _counters:
        _counters[key] = 0
    _init_devices.clear()
    _announced = False


def _get_module() -> Any:
    global _module, _module_error
    if _module is not None:
        return _module
    if _module_error is not None:
        return None
    try:
        from ...jit.mla_decode_fp8_paged_sm90 import (
            gen_mla_decode_fp8_paged_sm90_module,
        )

        _module = gen_mla_decode_fp8_paged_sm90_module().build_and_load()
    except Exception as exc:  # build or load failure: stock path, say why
        _module_error = f"{type(exc).__name__}: {exc}"[:800]
        _counters["build_failed"] += 1
        logger.warning_once(
            "mla_decode_fp8_paged_sm90: JIT build failed (%s); fp8-Q calls "
            "are served by exact dequantisation into the stock MLA path",
            _module_error,
        )
        return None
    return _module


def _mla_dec_fp8q_sm90_prepare(
    wrapper: Any,
    plan_metadata: Any,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: Optional[float],
    lse_mode: Any,
    skip_softmax: Any,
    use_sinks: Any,
    explicit_output_dtype: Optional[torch.dtype],
) -> None:
    """Plan-time half: decide eligibility, JIT-build the module (first time
    per process), build the dense block table and device kv-length vector,
    and size the caller-owned split workspace and scale buffers. Everything
    that allocates, compiles or queries the device happens here, outside
    CUDA-graph capture; run() only launches."""
    counters = _counters
    wrapper._mla_dec_fp8q_sm90_state = None
    counters["plans_seen"] += 1
    if _disabled():
        counters["plans_disabled"] += 1
        logger.info_once(
            "mla_decode_fp8_paged_sm90: disabled by %s=%s; fp8-Q calls are "
            "served by exact dequantisation into the stock MLA path",
            _DISABLE_ENV,
            os.environ.get(_DISABLE_ENV, ""),
        )
        return
    device = wrapper.device if hasattr(wrapper, "device") else None
    try:
        reason = None
        if explicit_output_dtype not in (None, torch.bfloat16):
            reason = f"output_dtype={explicit_output_dtype}"
        elif num_heads != _NUM_HEADS:
            reason = f"num_heads={num_heads}"
        elif head_dim_ckv != _HEAD_DIM_CKV:
            reason = f"head_dim_ckv={head_dim_ckv}"
        elif head_dim_kpe != _HEAD_DIM_KPE:
            reason = f"head_dim_kpe={head_dim_kpe}"
        elif page_size != _PAGE_SIZE:
            reason = f"page_size={page_size}"
        elif not causal:
            reason = "causal=False"
        elif sm_scale is None:
            reason = "sm_scale=None"
        elif lse_mode not in (None, "none", False):
            reason = f"lse_mode={lse_mode}"
        elif skip_softmax:
            reason = "skip_softmax"
        elif use_sinks:
            reason = "use_sinks"
        elif device is None or device.type != "cuda":
            reason = f"device={device}"
        elif tuple(get_compute_capability(device)) != _COMPUTE_CAPABILITY:
            reason = "not SM90"
        if reason is None:
            md = plan_metadata
            if md.qo_indptr is not None:
                qo = md.qo_indptr.cpu()
                B = int(qo.numel()) - 1
                q_lens = qo[1:] - qo[:-1]
                if B < 1 or B > _MAX_BATCH:
                    reason = f"batch_size={B}"
                elif bool((q_lens != q_lens[0]).any()) or not (
                    1 <= int(q_lens[0]) <= _MAX_Q_LEN
                ):
                    reason = (
                        "q_len must be uniform and <= %d (measured decode "
                        "widths), got %s" % (_MAX_Q_LEN, q_lens.tolist()[:8])
                    )
                else:
                    QL = int(q_lens[0])
            elif md.cum_seq_lens_q is not None:
                qo = md.cum_seq_lens_q.cpu()
                B = int(qo.numel()) - 1
                q_lens = qo[1:] - qo[:-1]
                if B < 1 or B > _MAX_BATCH:
                    reason = f"batch_size={B}"
                elif bool((q_lens != q_lens[0]).any()) or not (
                    1 <= int(q_lens[0]) <= _MAX_Q_LEN
                ):
                    reason = "q_len must be uniform and <= %d" % _MAX_Q_LEN
                else:
                    QL = int(q_lens[0])
            else:
                reason = "no query metadata"
        if reason is None:
            # kv lengths + dense block table from either metadata form
            if md.kv_len_arr is not None:
                kv_lens_host = md.kv_len_arr.cpu().flatten().to(torch.int64)
            elif md.seq_lens is not None:
                kv_lens_host = md.seq_lens.cpu().flatten().to(torch.int64)
            else:
                reason = "no kv-length metadata"
        if reason is None:
            if kv_lens_host.numel() < B:
                reason = "kv-length metadata shorter than batch"
            elif int(kv_lens_host[:B].max()) > _MAX_KV_LEN:
                reason = (
                    f"kv_len={int(kv_lens_host[:B].max())} above measured "
                    "envelope"
                )
            elif int(kv_lens_host[:B].min()) < 1:
                reason = "kv_len < 1"
    except Exception as exc:  # a guard failure must never break the stock path
        reason = f"guard_error:{type(exc).__name__}: {exc}"[:300]
    if reason is not None:
        counters["plans_ineligible"] += 1
        logger.info_once(
            "mla_decode_fp8_paged_sm90: not dispatched for this plan (%s); "
            "fp8-Q calls are served by exact dequantisation into the stock "
            "MLA path",
            reason,
        )
        return
    if (
        hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    ):
        counters["runs_ineligible"] += 1
        return
    module = _get_module()
    if module is None:
        return
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    init_key = (id(module), device_index)
    if init_key not in _init_devices:
        module.init(device_index)
        _init_devices.add(init_key)

    bufs = getattr(wrapper, "_mla_dec_fp8q_sm90_bufs", None)
    if bufs is None:
        bufs = {}
        wrapper._mla_dec_fp8q_sm90_bufs = bufs

    def _buf(name, numel, dtype):
        t = bufs.get(name)
        if t is None or t.numel() < numel or t.device != device:
            t = torch.empty(numel, dtype=dtype, device=device)
            bufs[name] = t
            _counters["workspace_allocs"] += 1
        return t

    md = plan_metadata
    if md.block_tables is not None:
        bt_src = md.block_tables
        if (
            bt_src.dtype != torch.int32
            or bt_src.dim() != 2
            or bt_src.shape[0] < B
        ):
            counters["plans_ineligible"] += 1
            logger.info_once(
                "mla_decode_fp8_paged_sm90: dense block_tables form "
                "unusable (%s %s); the stock path serves this plan",
                bt_src.dtype,
                tuple(bt_src.shape),
            )
            return
        block_table = bt_src[:B].to(device=device, dtype=torch.int32)
        pps = int(block_table.shape[1])
    else:
        indptr = md.kv_indptr.cpu().to(torch.int64)
        indices = md.kv_indices
        pprs = (indptr[1 : B + 1] - indptr[:B]).tolist()
        pps = max(1, max(pprs) if pprs else 1)
        bt = bufs.get("block_table")
        if (
            bt is None
            or bt.shape[0] < B
            or bt.shape[1] != pps
            or bt.device != device
        ):
            bt = torch.zeros((B, pps), dtype=torch.int32, device=device)
            bufs["block_table"] = bt
            _counters["workspace_allocs"] += 1
        else:
            bt.zero_()
        idx_dev = indices.to(device)
        for i in range(B):
            s, e = int(indptr[i]), int(indptr[i + 1])
            if e > s:
                bt[i, : e - s].copy_(idx_dev[s:e].to(torch.int32))
        block_table = bt[:B]

    seq_lens_buf = _buf("seq_lens", B, torch.int32)[:B]
    seq_lens_buf.copy_(
        kv_lens_host[:B].to(torch.int32).to(device), non_blocking=True
    )
    # caller-owned split workspace, sized for the worst split count the
    # launch arithmetic can choose for this (B, QL, pps) plan
    ns_max = min(256, (pps + 1) // 2) if pps > 1 else 1
    ws_bytes = B * ns_max * QL * _NUM_HEADS * (_HEAD_DIM_CKV + 8)
    workspace = _buf("workspace", max(ws_bytes, 16), torch.uint8)
    q_descale = _buf("q_descale", 1, torch.float32)
    k_descale = _buf("k_descale", 1, torch.float32)

    from types import SimpleNamespace

    wrapper._mla_dec_fp8q_sm90_state = SimpleNamespace(
        batch_size=B,
        q_len=QL,
        total_q=B * QL,
        pps=pps,
        block_table=block_table,
        seq_lens=seq_lens_buf,
        workspace=workspace,
        q_descale=q_descale,
        k_descale=k_descale,
        sm_scale=float(sm_scale),
        device_index=device_index,
    )
    counters["plans_prepared"] += 1


def _packed_fp8_view(
    value: Any, widths: Tuple[int, int], name: str
) -> Optional[torch.Tensor]:
    """Return the packed fp8 (…, widths[0]+widths[1]) tensor for `value`,
    which may be packed already or a zero-copy adjacent split pair; None when
    the representation is not the packed-fp8 contract (caller falls back)."""
    w0, w1 = widths
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float8_e4m3fn or value.shape[-1] != w0 + w1:
            return None
        return value if value.is_contiguous() else None
    if not (isinstance(value, (tuple, list)) and len(value) == 2):
        return None
    a, b = value
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return None
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return None
    if a.shape[-1] != w0 or b.shape[-1] != w1:
        return None
    if a.shape[:-1] != b.shape[:-1]:
        return None
    # both views must be slices of one contiguous (..., w0 + w1) parent:
    # identical strides equal to the packed-contiguous strides, and the PE
    # view starting exactly w0 fp8 elements (bytes) after the no-PE view
    shape = tuple(a.shape[:-1]) + (w0 + w1,)
    exp_strides = []
    acc = 1
    for s in reversed(shape):
        exp_strides.append(acc)
        acc *= s
    exp_strides = tuple(reversed(exp_strides))
    if tuple(a.stride()) != exp_strides or tuple(b.stride()) != exp_strides:
        return None
    if b.data_ptr() != a.data_ptr() + w0:  # fp8 elements are 1 byte
        return None
    return a.as_strided(shape, exp_strides)


def _scale_value(scale: Any, name: str) -> Tuple[Optional[float], Optional[str]]:
    if scale is None:
        return 1.0, None
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        return float(scale), None
    if isinstance(scale, torch.Tensor) and scale.numel() == 1:
        return float(scale.reshape(-1)[0].item()), None
    return None, f"{name} must be a scalar, got {type(scale).__name__}"


def _mla_dec_fp8q_sm90_dispatch(
    wrapper: Any,
    query: Any,
    kv_cache: Any,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
    q_scale: Any,
    ckv_scale: Any,
    ckv_scale_arr: Any,
    kpe_scale: Any,
    sinks: Any,
    skip_softmax_threshold_scale_factor: Any,
) -> Tuple[Optional[torch.Tensor], Any]:
    """Run-time half. Returns (output, query): output not-None means the
    specialized kernel served the call; otherwise output is None and `query`
    is the exact bf16-dequantised query for the stock plan (which was made
    for bf16 q). Capture-safe: no sync except reading scalar scale tensors,
    no compilation; the only allocation is the output tensor when the caller
    did not pass one."""
    global _announced
    state = getattr(wrapper, "_mla_dec_fp8q_sm90_state", None)
    counters = _counters

    def _fallback(reason: str):
        counters["runs_ineligible"] += 1
        logger.info_once(
            "mla_decode_fp8_paged_sm90: not dispatched for this call (%s); "
            "the dequant + stock MLA path serves it",
            reason,
        )
        return None, _dequant_query(query, q_scale)

    if _disabled():
        counters["runs_disabled"] += 1
        logger.info_once(
            "mla_decode_fp8_paged_sm90: disabled by %s=%s; fp8-Q calls are "
            "served by exact dequantisation into the stock MLA path",
            _DISABLE_ENV,
            os.environ.get(_DISABLE_ENV, ""),
        )
        return None, _dequant_query(query, q_scale)
    if state is None:
        counters["runs_dequant_fallback"] += 1
        return None, _dequant_query(query, q_scale)
    try:
        if return_lse or lse is not None:
            return _fallback("return_lse")
        if sinks is not None:
            return _fallback("sinks")
        if skip_softmax_threshold_scale_factor is not None:
            return _fallback("skip_softmax_threshold_scale_factor")
        if ckv_scale_arr is not None:
            return _fallback("ckv_scale_arr (per-token scales)")
        q_packed = _packed_fp8_view(query, (_HEAD_DIM_CKV, _HEAD_DIM_KPE),
                                    "query")
        if q_packed is None or q_packed.dim() != 3:
            return _fallback("query is not a packed/adjacent fp8 form")
        if q_packed.shape[0] != state.total_q or q_packed.shape[1] != _NUM_HEADS:
            return _fallback(
                f"query shape {tuple(q_packed.shape)} != planned "
                f"({state.total_q}, {_NUM_HEADS}, {_DIMQ})"
            )
        kv_packed = _packed_fp8_view(kv_cache, (_HEAD_DIM_CKV, _HEAD_DIM_KPE),
                                     "kv_cache")
        if kv_packed is None or kv_packed.dim() not in (3, 4):
            return _fallback("kv_cache is not a packed/adjacent fp8 form")
        if kv_packed.dim() == 4:
            if kv_packed.shape[2] != 1:
                return _fallback("kv_cache head dim != 1")
            kv_packed = kv_packed.reshape(
                kv_packed.shape[0], kv_packed.shape[1], _DIMQ
            )
        if kv_packed.shape[1] != _PAGE_SIZE:
            return _fallback(f"kv page size {kv_packed.shape[1]}")
        qs, reason = _scale_value(q_scale, "q_scale")
        if reason is not None:
            return _fallback(reason)
        ks, reason = _scale_value(ckv_scale, "ckv_scale")
        if reason is not None:
            return _fallback(reason)
        if kpe_scale is not None:
            ps, reason = _scale_value(kpe_scale, "kpe_scale")
            if reason is not None:
                return _fallback(reason)
            if abs(ps - ks) > 1e-12:
                return _fallback(
                    "the packed cache takes one KV descale; "
                    f"kpe_scale={ps} != ckv_scale={ks}"
                )
        if out is not None and (
            out.dtype != torch.bfloat16
            or tuple(out.shape)
            != (state.total_q, _NUM_HEADS, _HEAD_DIM_CKV)
            or not out.is_contiguous()
        ):
            return _fallback(f"out {out.dtype} {tuple(out.shape)}")
        device = q_packed.device
        if device.type != "cuda" or (
            device.index is not None and device.index != state.device_index
        ):
            return _fallback(f"device {device}")
    except Exception as exc:
        return _fallback(f"guard_error:{type(exc).__name__}: {exc}"[:300])

    module = _module
    capturing = bool(
        hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    )
    if module is None:
        return _fallback("module dropped")
    if out is None:
        out = torch.empty(
            (state.total_q, _NUM_HEADS, _HEAD_DIM_CKV),
            dtype=torch.bfloat16,
            device=device,
        )
    # Scalar descales are written into plan-owned buffers; under CUDA-graph
    # capture the values present at capture are baked (like sm_scale).
    state.q_descale.fill_(qs)
    state.k_descale.fill_(ks)
    module.run(
        q_packed.view(state.batch_size, state.q_len, _NUM_HEADS, _DIMQ),
        kv_packed,
        state.block_table,
        state.seq_lens,
        state.q_descale,
        state.k_descale,
        state.sm_scale,
        out.view(state.batch_size, state.q_len, _NUM_HEADS, _HEAD_DIM_CKV),
        state.workspace,
    )
    counters["runs_dispatched"] += 1
    if capturing:
        counters["runs_dispatched_capturing"] += 1
    if not _announced:
        _announced = True
        logger.info(
            "mla_decode_fp8_paged_sm90: serving the fp8-Q paged MLA decode "
            "(batch_size=%d, q_len=%d, cuda_graph=%s)",
            state.batch_size,
            state.q_len,
            getattr(wrapper, "_use_cuda_graph", False),
        )
    return out, query


def _dequant_query(query: Any, q_scale: Any) -> Any:
    """Exact dequantisation of an fp8 (e4m3) query for the stock MLA path
    (bf16 holds every e4m3 value exactly; the scalar q descale is applied
    here and consumed)."""
    _counters["runs_dequant_fallback"] += 1
    qs, reason = _scale_value(q_scale, "q_scale")
    if reason is not None:
        raise ValueError("mla_decode_fp8_paged_sm90: " + reason)

    def _dq(t: torch.Tensor) -> torch.Tensor:
        return (t.to(torch.float32) * qs).to(torch.bfloat16)

    if isinstance(query, torch.Tensor):
        return _dq(query)
    a, b = query
    return (_dq(a), _dq(b))
