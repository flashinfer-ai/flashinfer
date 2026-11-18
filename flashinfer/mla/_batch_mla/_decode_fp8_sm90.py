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
Q/KV descales, and the measured envelope (batch <= 128, kv_len <= 262144,
and every request's kv_len >= its q_len — a shorter context leaves causal
query rows with EMPTY key windows, whose zero-output empty-row semantics
belong to the stock path). The guard is an UNSCOPED configuration guard: no
per-shape competitive carve-outs — backend choice between engines is
per-deployment, not per-shape.

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

CUDA-graph replan semantics (part of the API contract): on a
``use_cuda_graph=True`` wrapper the prepared metadata (block table,
sequence lengths) lives in wrapper-owned buffers whose ADDRESSES never
change, and a successful replan at the prepared buffer geometry (same
batch, q_len and table width) refreshes their CONTENTS in place — for BOTH
metadata forms, including a caller-provided dense ``block_tables`` tensor,
whose contents are copied through the stable buffer rather than rebound —
so a graph captured earlier replays the NEW plan, never a stale mix of old
page IDs with new lengths. A successful replan whose metadata CANNOT be
refreshed through the captured buffers (different geometry, or a plan
outside the fast-path contract, both of which leave the fast path and the
buffer contents untouched) INVALIDATES previously captured graphs: discard
them and capture anew — their replays keep computing the previous plan's
values, never a mix. A FAILED (backend-rejected) replan restores the
buffer contents, so existing captures stay valid — see the wrapper's
plan(). The kill-switch env vars are an operator override: they tear down
the fast path for eager calls while existing captures keep replaying their
captured plan.
"""

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch

from ...jit.core import logger
from ...utils import get_compute_capability
from ._contracts import (
    _adjacent_last_dim_view,
    _are_adjacent_last_dim_views,
    _parse_structural_mla_input,
)


class _GraphRealloc(Exception):
    """A CUDA-graph wrapper replan would reallocate prepared buffers whose
    addresses an already-captured graph may hold."""


_DISABLE_ENV = "FLASHINFER_DISABLE_MLA_DECODE_FP8_PAGED_SM90"
# shared specialized-kernel kill switch (#5667 convention), honored as an
# alias of the route-scoped switch above; the census records which fired
_DISABLE_ENV_SHARED = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
_FP8Q_BACKENDS = ("auto", "specialized", "dequant")
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
    "plans_selector_dequant": 0,
    "specialized_raises": 0,
    "plans_seen": 0,
    "plans_disabled": 0,
    "plans_ineligible": 0,
    "plans_prepared": 0,
    "replans_refreshed_captured": 0,
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


def _disabled_by() -> Optional[str]:
    """The kill-switch env var that fired, or None (route-scoped name and
    the shared #5667 alias are both honored)."""
    for env in (_DISABLE_ENV, _DISABLE_ENV_SHARED):
        if os.environ.get(env, "") not in ("", "0"):
            return env
    return None


def _disabled() -> bool:
    return _disabled_by() is not None


def _fp8q_selector(wrapper: Any) -> str:
    return getattr(wrapper, "_fp8q_backend", "auto")


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
    use_profiler: bool = False,
) -> None:
    """Plan-time half: decide eligibility, JIT-build the module (first time
    per process), build the dense block table and device kv-length vector,
    and size the caller-owned split workspace and scale buffers. Everything
    that allocates, compiles or queries the device happens here, outside
    CUDA-graph capture; run() only launches."""
    counters = _counters
    wrapper._mla_dec_fp8q_sm90_state = None
    counters["plans_seen"] += 1
    selector = _fp8q_selector(wrapper)
    if selector == "dequant":
        counters["plans_selector_dequant"] += 1
        logger.info_once(
            "mla_decode_fp8_paged_sm90: fp8q_backend='dequant'; the fast "
            "path is never prepared, fp8-Q calls are served by exact "
            "dequantisation into the stock MLA path"
        )
        return
    fired = _disabled_by()
    if fired is not None:
        counters["plans_disabled"] += 1
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "the kill switch %s=%s disables the fast path (an "
                "explicitly selected backend is never silently swapped)"
                % (fired, os.environ.get(fired, ""))
            )
        logger.info_once(
            "mla_decode_fp8_paged_sm90: disabled by %s=%s; fp8-Q calls are "
            "served by exact dequantisation into the stock MLA path",
            fired,
            os.environ.get(fired, ""),
        )
        return
    device = wrapper.device if hasattr(wrapper, "device") else None
    # Under a CUDA-graph wrapper an already-captured graph binds the prepared
    # buffers' ADDRESSES, so a replan never reallocates or rebinds them: a
    # refreshable replan rewrites their CONTENTS in place (replays then
    # compute the NEW plan), and a replan that cannot be refreshed through
    # them leaves the buffers untouched and — by the documented contract —
    # INVALIDATES previously captured graphs (their replays keep computing
    # the previous plan's values, never a mix; callers re-capture).
    use_graph = bool(getattr(wrapper, "_use_cuda_graph", False))
    captured = bool(getattr(wrapper, "_mla_dec_fp8q_sm90_scales_captured", False))
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
        elif not (math.isfinite(float(sm_scale)) and float(sm_scale) > 0.0):
            # The kernels reduce the per-row score maximum BEFORE applying
            # the fused qk scale (max(score) * qk_scale), which is only
            # order-preserving for a positive scale; a zero scale would also
            # turn -inf score masks into NaN. The stock path serves these.
            reason = f"sm_scale={sm_scale} is not a finite positive value"
        elif use_profiler:
            # The specialized dispatch does not receive profiler_buffer and
            # would produce no profiler records; profiled plans stay on the
            # stock path, which implements the profiling workflow.
            reason = "use_profiler"
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
                reason = f"kv_len={int(kv_lens_host[:B].max())} above measured envelope"
            elif int(kv_lens_host[:B].min()) < QL:
                # A request with kv_len < q_len has causal query rows whose
                # visible-key window is EMPTY. The kernels clamp every row's
                # key limit to >= 1 (they would attend such rows to key 0 and
                # emit a value), while the contract's empty-row semantics are
                # zero output rows — which the stock path implements — so
                # these plans decline to the exact dequant + stock path.
                reason = (
                    "kv_len=%d < q_len=%d leaves causal rows with zero "
                    "visible keys (empty-row semantics belong to the stock "
                    "path)" % (int(kv_lens_host[:B].min()), QL)
                )
    except Exception as exc:  # a guard failure must never break the stock path
        reason = f"guard_error:{type(exc).__name__}: {exc}"[:300]
    if reason is not None:
        counters["plans_ineligible"] += 1
        if captured:
            # The captured buffers stay untouched (this return precedes every
            # buffer write), so existing captures keep replaying the PREVIOUS
            # plan's values intact; by the documented contract this replan
            # invalidates them and callers must re-capture.
            logger.warning(
                "mla_decode_fp8_paged_sm90: this replan is outside the "
                "fast-path contract (%s) and cannot refresh the metadata "
                "buffers a captured graph reads; previously captured graphs "
                "are invalidated (they keep computing the previous plan) — "
                "discard them and capture anew",
                reason,
            )
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "this plan is outside the measured contract (%s); an "
                "explicitly selected backend is never silently swapped — "
                "use fp8q_backend='auto' for guarded fallback" % reason
            )
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
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "plan() ran under CUDA-graph capture, where the fast path "
                "cannot be prepared (allocation and compilation are plan-time "
                "work); plan outside capture first"
            )
        return
    module = _get_module()
    if module is None:
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "the kernel module failed to build or load (%s); an "
                "explicitly selected backend is never silently swapped"
                % (_module_error,)
            )
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
            if use_graph and t is not None:
                raise _GraphRealloc(
                    f"{name} buffer would be reallocated "
                    f"({t.numel()} -> {numel} elements)"
                )
            t = torch.empty(numel, dtype=dtype, device=device)
            bufs[name] = t
            _counters["workspace_allocs"] += 1
        return t

    try:
        md = plan_metadata
        if md.block_tables is not None:
            bt_src = md.block_tables
            if bt_src.dtype != torch.int32 or bt_src.dim() != 2 or bt_src.shape[0] < B:
                counters["plans_ineligible"] += 1
                logger.info_once(
                    "mla_decode_fp8_paged_sm90: dense block_tables form "
                    "unusable (%s %s); the stock path serves this plan",
                    bt_src.dtype,
                    tuple(bt_src.shape),
                )
                return
            pps = int(bt_src.shape[1])
            if use_graph:
                # The captured graph binds the stable wrapper-owned buffer's
                # address, so the new table's CONTENTS are copied through it
                # (never rebind the state to the caller's tensor: a replay
                # would keep reading the previous plan's page IDs while the
                # refreshed sequence lengths change under it).
                bt = bufs.get("block_table")
                if (
                    bt is None
                    or bt.shape[0] < B
                    or bt.shape[1] != pps
                    or bt.device != device
                ):
                    if bt is not None:
                        raise _GraphRealloc(
                            "block_table geometry changed "
                            f"({tuple(bt.shape)} -> ({B}, {pps}))"
                        )
                    bt = torch.zeros((B, pps), dtype=torch.int32, device=device)
                    bufs["block_table"] = bt
                    _counters["workspace_allocs"] += 1
                bt[:B].copy_(bt_src[:B])
                block_table = bt[:B]
            else:
                block_table = bt_src[:B].to(device=device, dtype=torch.int32)
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
                if use_graph and bt is not None:
                    raise _GraphRealloc(
                        "block_table geometry changed "
                        f"({tuple(bt.shape)} -> ({B}, {pps}))"
                    )
                bt = torch.zeros((B, pps), dtype=torch.int32, device=device)
                bufs["block_table"] = bt
                _counters["workspace_allocs"] += 1
            else:
                bt.zero_()
            # Build the dense table with one masked scatter instead of one
            # small copy_ per request: plan() runs on every decode step, so
            # the launch count must not scale with the batch size. The CSR
            # segments are contiguous in `indices`, and the row-major order
            # of the mask matches their concatenation.
            idx_dev = indices.to(device=device, dtype=torch.int32)
            counts = torch.tensor(pprs, dtype=torch.int64, device=device)
            cols = torch.arange(pps, device=device)
            mask = cols.unsqueeze(0) < counts.unsqueeze(1)
            start = int(indptr[0])
            total = int(indptr[B]) - start
            bt[:B][mask] = idx_dev[start : start + total]
            block_table = bt[:B]

        seq_lens_buf = _buf("seq_lens", B, torch.int32)[:B]
        seq_lens_buf.copy_(
            kv_lens_host[:B].to(torch.int32).to(device), non_blocking=True
        )
        # caller-owned split workspace, sized for the worst split count the
        # launch arithmetic can choose for this (B, QL, pps) plan. The
        # repaired launcher's B>=5 compact-dual path consumes ONE page per
        # split (its pps_split is not forced even), so ns can reach pps; the
        # other branches stay under this bound too, and every branch is
        # capped at 256 splits in the launch arithmetic itself. The earlier
        # two-page-only bound min(256, ceil(pps/2)) undersized exactly the
        # ragged short-kv B>=5 class and run() refused fail-closed.
        ns_max = min(256, pps) if pps > 1 else 1
        ws_bytes = B * ns_max * QL * _NUM_HEADS * (_HEAD_DIM_CKV + 8)
        workspace = _buf("workspace", max(ws_bytes, 16), torch.uint8)
        q_descale = _buf("q_descale", 1, torch.float32)
        k_descale = _buf("k_descale", 1, torch.float32)
    except _GraphRealloc as exc:
        counters["plans_ineligible"] += 1
        if captured:
            logger.warning(
                "mla_decode_fp8_paged_sm90: this CUDA-graph replan cannot be "
                "refreshed through the prepared buffers (%s); previously "
                "captured graphs are invalidated (they keep computing the "
                "previous plan) — discard them and capture anew",
                exc,
            )
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "this CUDA-graph replan cannot keep the captured buffer "
                "addresses stable (%s); use a new wrapper for the new "
                "capture shape" % exc
            ) from None
        logger.info_once(
            "mla_decode_fp8_paged_sm90: not prepared for this CUDA-graph "
            "replan (%s); fp8-Q calls are served by exact dequantisation "
            "into the stock MLA path",
            exc,
        )
        return

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
        scales_cached=None,
        # The descale buffers live in wrapper-owned `bufs` and survive
        # replans, so a graph captured against an earlier state keeps
        # rewriting them on every replay. The "a graph holds these buffers"
        # fact must therefore persist on the wrapper, not on the per-plan
        # state, or a replan would let eager calls trust the scales cache
        # that a replay can invalidate behind its back.
        graph_captured=bool(
            getattr(wrapper, "_mla_dec_fp8q_sm90_scales_captured", False)
        ),
    )
    if captured and use_graph:
        # The stable buffers' contents were refreshed in place, so replays of
        # earlier captures now compute THIS plan.
        counters["replans_refreshed_captured"] += 1
    counters["plans_prepared"] += 1


def _packed_fp8_view(
    value: Any, widths: Tuple[int, int], name: str
) -> Optional[torch.Tensor]:
    """Return the packed fp8 (…, widths[0]+widths[1]) tensor for `value`,
    which may be any structural form the shared contract accepts — packed,
    a zero-copy adjacent split pair, or the (redundant) dual forms; None when
    the representation is not the packed-fp8 contract (caller falls back).
    `value` is normalized through `_parse_structural_mla_input`, the SAME
    parser the dequant fallback uses, so the two gates accept and reject the
    same structural forms (a bare list is rejected by both; the dual forms
    are served by both through their packed leaf)."""
    w0, w1 = widths
    try:
        packed, split = _parse_structural_mla_input(value, name=name)
    except (TypeError, ValueError):
        return None
    if packed is not None:
        # Packed leaf (alone, or the packed side of a dual form — the dequant
        # fallback likewise trusts the packed leaf of a dual form).
        if packed.dtype != torch.float8_e4m3fn or packed.shape[-1] != w0 + w1:
            return None
        return packed if packed.is_contiguous() else None
    assert split is not None
    a, b = split
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
    rev_strides = []
    acc = 1
    for s in reversed(shape):
        rev_strides.append(acc)
        acc *= s
    exp_strides = tuple(reversed(rev_strides))
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
        # Reading a CUDA tensor's value is a device-to-host sync; under
        # CUDA-graph capture that read is forbidden and would otherwise
        # poison the capture with an opaque runtime error (and the dequant
        # fallback needs the same host value, so it cannot serve the call
        # either). Reject it with a documented error instead.
        if (
            scale.is_cuda
            and hasattr(torch.cuda, "is_current_stream_capturing")
            and torch.cuda.is_current_stream_capturing()
        ):
            return None, (
                f"{name} is a CUDA tensor, whose value cannot be read "
                "during CUDA-graph capture; pass a Python float (or a CPU "
                "scalar tensor) when capturing"
            )
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
    for bf16 q). Capture-safe: no sync except reading scalar scale tensors
    eagerly — under CUDA-graph capture a CUDA-tensor scale cannot be read
    and is rejected with a clear ValueError (pass float scales, or CPU
    scalar tensors, when capturing) — no compilation; the only allocation
    is the output tensor when the caller did not pass one."""
    global _announced
    state = getattr(wrapper, "_mla_dec_fp8q_sm90_state", None)
    counters = _counters

    selector = _fp8q_selector(wrapper)

    def _fallback(reason: str):
        counters["runs_ineligible"] += 1
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "this call cannot be served by the fast path (%s); an "
                "explicitly selected backend is never silently swapped" % reason
            )
        logger.info_once(
            "mla_decode_fp8_paged_sm90: not dispatched for this call (%s); "
            "the dequant + stock MLA path serves it",
            reason,
        )
        return None, _dequant_query(query, q_scale)

    fired = _disabled_by()
    if fired is not None:
        counters["runs_disabled"] += 1
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "the kill switch %s=%s disables the fast path"
                % (fired, os.environ.get(fired, ""))
            )
        logger.info_once(
            "mla_decode_fp8_paged_sm90: disabled by %s=%s; fp8-Q calls are "
            "served by exact dequantisation into the stock MLA path",
            fired,
            os.environ.get(fired, ""),
        )
        return None, _dequant_query(query, q_scale)
    if state is None:
        if selector == "specialized":
            counters["specialized_raises"] += 1
            raise RuntimeError(
                "mla_decode_fp8_paged_sm90: fp8q_backend='specialized' but "
                "no prepared fast-path state exists for this plan; an "
                "explicitly selected backend is never silently swapped"
            )
        counters["runs_dequant_fallback"] += 1
        return None, _dequant_query(query, q_scale)
    # The guard computes only a reason string; _fallback runs ONCE, outside
    # the try block. _fallback itself raises under fp8q_backend='specialized'
    # and runs the dequant fallback (which can raise, e.g. for a CUDA-tensor
    # q_scale under capture); were it called inside the try, the except arm
    # would catch that raise and call it a second time, double-counting
    # specialized_raises / runs_ineligible / runs_dequant_fallback and
    # nesting the user-facing message inside a truncated guard_error.
    q_packed = kv_packed = None
    qs = ks = 0.0
    device = None

    def _guard() -> Optional[str]:
        nonlocal q_packed, kv_packed, qs, ks, device
        if return_lse or lse is not None:
            return "return_lse"
        if sinks is not None:
            return "sinks"
        if skip_softmax_threshold_scale_factor is not None:
            return "skip_softmax_threshold_scale_factor"
        if ckv_scale_arr is not None:
            return "ckv_scale_arr (per-token scales)"
        q_packed = _packed_fp8_view(query, (_HEAD_DIM_CKV, _HEAD_DIM_KPE), "query")
        if q_packed is None or q_packed.dim() != 3:
            return "query is not a packed/adjacent fp8 form"
        # The kernels read the packed query with 16-byte vectorized loads
        # (uint4) from its base address; a contiguous offset view whose data
        # pointer is not 16-byte aligned would read misaligned. This also
        # covers packed parents reconstructed from adjacent split views,
        # whose base is the first view's data pointer.
        if q_packed.data_ptr() % 16 != 0:
            return "query base address is not 16-byte aligned"
        if q_packed.shape[0] != state.total_q or q_packed.shape[1] != _NUM_HEADS:
            return (
                f"query shape {tuple(q_packed.shape)} != planned "
                f"({state.total_q}, {_NUM_HEADS}, {_DIMQ})"
            )
        kv_packed = _packed_fp8_view(
            kv_cache, (_HEAD_DIM_CKV, _HEAD_DIM_KPE), "kv_cache"
        )
        if kv_packed is None or kv_packed.dim() not in (3, 4):
            return "kv_cache is not a packed/adjacent fp8 form"
        if kv_packed.dim() == 4:
            if kv_packed.shape[2] != 1:
                return "kv_cache head dim != 1"
            kv_packed = kv_packed.reshape(kv_packed.shape[0], kv_packed.shape[1], _DIMQ)
        if kv_packed.shape[1] != _PAGE_SIZE:
            return f"kv page size {kv_packed.shape[1]}"
        # The kv base feeds 16-byte cp.async.cg loads (MT <= 2) and
        # cuTensorMapEncodeTiled (MT == 3), both of which require a 16-byte
        # aligned base; the native entry checks only strides, so a contiguous
        # view at an odd storage offset must be refused here, like the query.
        if kv_packed.data_ptr() % 16 != 0:
            return "kv_cache base address is not 16-byte aligned"
        qs, reason = _scale_value(q_scale, "q_scale")
        if reason is not None:
            return reason
        ks, reason = _scale_value(ckv_scale, "ckv_scale")
        if reason is not None:
            return reason
        if kpe_scale is not None:
            ps, reason = _scale_value(kpe_scale, "kpe_scale")
            if reason is not None:
                return reason
            if not (math.isfinite(ps) and abs(ps - ks) <= 1e-12):
                return (
                    "the packed cache takes one finite KV descale; "
                    f"kpe_scale={ps} != ckv_scale={ks}"
                )
        # The kernel's masked-score arithmetic multiplies -inf sentinels by
        # the fused qk scale, so a zero, negative, or nonfinite descale would
        # turn masked positions into NaN/+inf; the exact dequant path handles
        # those values instead.
        for sv, sname in ((qs, "q_scale"), (ks, "ckv_scale")):
            if not (math.isfinite(sv) and sv > 0.0):
                return f"{sname}={sv} is not a finite positive scalar descale"
        if out is not None and (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != (state.total_q, _NUM_HEADS, _HEAD_DIM_CKV)
            or not out.is_contiguous()
        ):
            return f"out {out.dtype} {tuple(out.shape)}"
        device = q_packed.device
        if device.type != "cuda" or (
            device.index is not None and device.index != state.device_index
        ):
            return f"device {device}"
        return None

    try:
        guard_reason = _guard()
    except Exception as exc:
        guard_reason = f"guard_error:{type(exc).__name__}: {exc}"[:300]
    if guard_reason is not None:
        return _fallback(guard_reason)

    module = _module
    capturing = bool(
        hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    )
    if module is None:
        return _fallback("module dropped")
    # The kernel writes `out` through vectorized (__nv_bfloat162 / uint4)
    # stores from its base address, and the stock path's output store is also
    # vectorized — so a misaligned caller buffer cannot be served by falling
    # back with the same `out` either. Stage the result through a fresh
    # (allocator-aligned) buffer and copy into the caller's view; the copy is
    # capture-safe and runs only for this pathological input.
    staged_user_out = None
    if out is not None and out.data_ptr() % 16 != 0:
        staged_user_out, out = out, None
    if out is None:
        out = torch.empty(
            (state.total_q, _NUM_HEADS, _HEAD_DIM_CKV),
            dtype=torch.bfloat16,
            device=device,
        )
    # Scalar descales are written into plan-owned buffers only when they
    # change (per-tensor serve scales are constant across steps, so the
    # steady-state hot path launches no extra kernels). Under CUDA-graph
    # capture the writes are always recorded into the graph, whatever the
    # eager cache says: every replay then rewrites the buffers to the
    # capture-time values (truly baked, like sm_scale), so a later eager
    # call with different scales cannot retroactively change what a replay
    # computes. The eager cache is invalidated so the next eager call
    # refills the buffers.
    if capturing:
        state.q_descale.fill_(qs)
        state.k_descale.fill_(ks)
        state.graph_captured = True
        wrapper._mla_dec_fp8q_sm90_scales_captured = True
    elif state.graph_captured or state.scales_cached != (qs, ks):
        # Once any graph holds these buffers, a replay rewrites them to the
        # captured values behind the cache's back, so eager calls must always
        # refill instead of trusting the cache.
        state.q_descale.fill_(qs)
        state.k_descale.fill_(ks)
        state.scales_cached = (qs, ks)
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
    if staged_user_out is not None:
        staged_user_out.copy_(out)
        out = staged_user_out
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
    here and consumed). The structural form of `query` is preserved through
    the shared structural parser: an adjacent split pair (and the redundant
    dual form) is dequantised once through its packed parent, so the result
    still satisfies a packed-layout plan zero-copy; a dual form never
    reaches the per-leaf path (where its nested tuple would not be a
    tensor), and independent split pairs stay independent."""
    _counters["runs_dequant_fallback"] += 1
    qs, reason = _scale_value(q_scale, "q_scale")
    if reason is not None:
        raise ValueError("mla_decode_fp8_paged_sm90: " + reason)

    def _dq(t: torch.Tensor) -> torch.Tensor:
        return (t.to(torch.float32) * qs).to(torch.bfloat16)

    packed, split = _parse_structural_mla_input(query, name="query")
    if packed is not None:
        dq_packed = _dq(packed)
        if split is None:
            return dq_packed
        w0 = int(split[0].shape[-1])
        return (dq_packed, (dq_packed[..., :w0], dq_packed[..., w0:]))
    assert split is not None
    a, b = split
    if _are_adjacent_last_dim_views(a, b):
        parent = _adjacent_last_dim_view(a, b, assume_adjacent=True)
        assert parent is not None
        dq = _dq(parent)
        w0 = int(a.shape[-1])
        return (dq[..., :w0], dq[..., w0:])
    return (_dq(a), _dq(b))
