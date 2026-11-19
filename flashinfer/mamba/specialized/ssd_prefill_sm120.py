"""Specialized Mamba2 SSD chunk-scan prefill for SM120 (RTX PRO 6000):
single-sequence, zero-initial-state, chunk_size=128, headdim=64, dstate=128,
ngroups=8, bf16 io with fp32 state/accumulation, ``dt_softplus``, no z, no
final-state output.

This is an agent-generated kernel (CUDA, tvm-ffi destination-passing entry,
JIT-built for ``sm_120a``): a cooperative persistent-CTA design that keeps the
SSM state in registers (bf16 ``mma.sync`` with fp32 accumulation) and runs the
whole varlen scan in one launch. ``SSDCombined`` refuses SM120 outright today
("requires datacenter Blackwell ... for tcgen05 MMA"); this backend is the
first FlashInfer-resident SSD prefill path on that arch.

Dispatch follows the specialized-backend pattern (PR #5667): it serves
``SSDCombined`` through ``backend="auto"`` (routes here on SM120 only when
the constructor geometry and the per-call form match the dispatch surface in
``ssd_prefill_sm120_workload.json``) or an explicit ``backend="specialized"``;
an explicitly selected other backend is never swapped, and off-surface
``auto`` keeps today's refusal (fail-closed — no composable in-tree path
exists for this op on SM120). ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``
(read at call time) disables the backend entirely. Guard order: GPU-free
checks (kill switch, dtype, geometry, contiguity, call form) -> device guards
(compute capability, CUDA-graph warm state). Every refusal is counted in
``fallback_reasons``.

The sequence length is a runtime value (the kernel is varlen/ragged by
construction): the dispatch surface constrains geometry and call form, never
an exact-T list.
"""

import functools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec
from ...utils import get_compute_capability

logger = logging.getLogger(__name__)

# Kill switch shared by FlashInfer's specialized kernels: read at CALL time.
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

_WORKLOAD_FILE = "ssd_prefill_sm120_workload.json"

_warmed: set = set()  # (device_index, nheads) pairs that ran eagerly once
# The native module binds per-kernel launch state (shared-memory opt-in,
# cooperative residency) to the first CUDA device that runs it; the guard
# refuses any other device instead of launching against that state.
_native_device: Optional[int] = None
_allowlist_cache: Optional[dict] = None

_stats: Dict[str, Any] = {
    "calls": 0,
    "specialized_dispatches": 0,
    "fallback_dispatches": 0,
    "fallback_reasons": {},
}


def _bump_fallback(reason: str) -> str:
    _stats["fallback_dispatches"] += 1
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    return reason


def _load_allowlist() -> Optional[dict]:
    """Load the dispatch surface; ``None`` => never dispatch."""
    global _allowlist_cache
    if _allowlist_cache is not None:
        return _allowlist_cache
    try:
        payload = json.loads((Path(__file__).parent / _WORKLOAD_FILE).read_text())
        geo = payload["geometry"]
        _allowlist_cache = {
            "min_cc": tuple(int(v) for v in payload["min_compute_capability"]),
            "max_cc": tuple(int(v) for v in payload["max_compute_capability"]),
            "chunk_size": int(geo["chunk_size"]),
            "headdim": int(geo["headdim"]),
            "dstate": int(geo["dstate"]),
            "ngroups": int(geo["ngroups"]),
            "nheads": frozenset(int(v) for v in geo["nheads_allowlist"]),
        }
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            "flashinfer.mamba specialized SSD prefill: workload allowlist "
            "unavailable, backend disabled: %s",
            exc,
        )
        _allowlist_cache = None
    return _allowlist_cache


def specialized_ssd_prefill_stats() -> dict:
    """Diagnostics: dispatch counters and the dispatch surface."""
    allow = _load_allowlist()
    return {
        **_stats,
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        # Shallow copy: handing out the live dict would let a diagnostics
        # consumer mutate the dispatch surface the guards read (its values
        # are immutable, so a shallow copy suffices).
        "allowlist": dict(allow) if allow is not None else None,
        "warmed": sorted(_warmed),
        "kill_switch_env": _DISABLE_ENV,
    }


def gen_specialized_ssd_prefill_sm120_module() -> JitSpec:
    src_dir = jit_env.FLASHINFER_CSRC_DIR / "specialized_mamba2_ssd_prefill_sm120"
    sources = sorted(Path(src_dir).glob("*.cu"))
    return gen_jit_spec(
        "specialized_mamba2_ssd_prefill_sm120",
        sources,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--pre-include=cooperative_groups.h,mutex,tvm/ffi/extra/cuda/base.h",
            # single-token form REQUIRED: cpp_ext only treats "-gencode=..."
            # as a module arch override; the kernel's dynamic-smem sizing and
            # cooperative residency math assume the sm_120a target.
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def _get_cuda_module():
    return gen_specialized_ssd_prefill_sm120_module().build_and_load()


def _kernel_entry():
    """The DPS kernel entry (JIT/module load happens here — never under
    CUDA-graph capture, which the guard's warmed rule already prevents)."""
    return _get_cuda_module().run


def construction_supported(
    *,
    chunk_size: int,
    nheads: int,
    headdim: int,
    dstate: int,
    ngroups: int,
    io_dtype: torch.dtype,
    has_d: bool,
    d_has_hdim: bool,
    has_z: bool,
    has_initial_states: bool = False,
) -> Optional[str]:
    """Constructor-time (GPU-free) half of the guard: ``None`` when the
    specialized backend can serve instances with this geometry, else the
    refusal reason."""
    allow = _load_allowlist()
    if allow is None:
        return "allowlist_unavailable"
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        return "kill_switch"
    if chunk_size != allow["chunk_size"]:
        return "chunk_size"
    if headdim != allow["headdim"] or dstate != allow["dstate"]:
        return "head_geometry"
    if ngroups != allow["ngroups"]:
        return "ngroups"
    if nheads not in allow["nheads"]:
        return "nheads_off_allowlist"
    if io_dtype != torch.bfloat16:
        return "io_dtype"
    if not has_d or d_has_hdim:
        return "d_form"
    if has_z:
        return "z_unsupported"
    if has_initial_states:
        return "initial_states_unsupported"
    return None


def _guard_call(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    dt_limit: Tuple[float, float],
    initial_states: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    return_final_states: bool,
    num_seqs: Optional[int],
    cu_seqlens: Optional[torch.Tensor],
    extras_none: bool,
    geo: Tuple[int, int, int, int],
) -> Optional[str]:
    """Per-call half of the guard; ``None`` when servable."""
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        return "kill_switch"
    allow = _load_allowlist()
    if allow is None:
        return "allowlist_unavailable"
    chunk_size, nheads, headdim, dstate = geo
    # --- GPU-free call-form checks ---
    if not extras_none:
        return "unsupported_call_arguments"
    if z is not None or initial_states is not None:
        return "z_or_initial_states"
    if not dt_softplus:
        return "dt_softplus_required"
    if not (dt_limit[0] == 0.0 and dt_limit[1] == float("inf")):
        return "dt_limit"
    if return_final_states:
        return "final_states_unsupported"
    if D is None or dt_bias is None:
        return "d_dt_bias_required"
    if x.ndim != 4 or x.shape[0] != 1:
        return "batch_form"
    if num_seqs not in (None, 1):
        return "multi_sequence"
    if cu_seqlens is not None:
        if cu_seqlens.numel() != 2:
            return "multi_sequence"
        # Metadata-only checks (never a device sync): require the documented
        # device int32 offset-vector form.  The two offset values themselves
        # are trusted as [0, T] — the single-sequence caller precondition of
        # ``SSDCombined.run`` — because validating them would synchronize.
        if (
            cu_seqlens.dim() != 1
            or cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != x.device
        ):
            return "cu_seqlens_form"
    T = x.shape[1]
    if T < 1:
        return "empty_sequence"
    if x.shape[2] != nheads or x.shape[3] != headdim:
        return "x_geometry"
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return "x_form"
    if (
        dt.shape != (1, T, nheads)
        or dt.dtype != torch.bfloat16
        or not dt.is_contiguous()
    ):
        return "dt_form"
    if A.shape != (nheads,) or A.dtype != torch.float32 or not A.is_contiguous():
        return "a_form"
    ng = allow["ngroups"]
    if (
        B.shape != (1, T, ng, dstate)
        or B.dtype != torch.bfloat16
        or not B.is_contiguous()
    ):
        return "b_form"
    if (
        C.shape != (1, T, ng, dstate)
        or C.dtype != torch.bfloat16
        or not C.is_contiguous()
    ):
        return "c_form"
    if D.shape != (nheads,) or D.dtype != torch.bfloat16 or not D.is_contiguous():
        return "d_form"
    if (
        dt_bias.shape != (nheads,)
        or dt_bias.dtype != torch.bfloat16
        or not dt_bias.is_contiguous()
    ):
        return "dt_bias_form"
    if out is not None and (
        out.shape != (1, T, nheads, headdim)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.device != x.device
    ):
        return "out_form"
    # --- GPU-free pointer checks (placement, alignment, aliasing) ---
    if x.device.type != "cuda":
        return "device_mismatch"
    for t in (dt, A, B, C, D, dt_bias):
        if t.device != x.device:
            return "device_mismatch"
    # The kernel uses 16-byte vector memory operations and TMA tensor maps;
    # a storage-offset view can defeat the allocator's natural alignment.
    for t in (x, B, C):
        if t.data_ptr() % 16 != 0:
            return "alignment"
    if out is not None:
        if out.data_ptr() % 16 != 0:
            return "alignment"
        # Output stores must never land in memory an input still reads.
        o0 = out.data_ptr()
        o1 = o0 + out.numel() * out.element_size()
        for t in (x, dt, A, B, C, D, dt_bias):
            t0 = t.data_ptr()
            t1 = t0 + t.numel() * t.element_size()
            if o0 < t1 and t0 < o1:
                return "out_overlap"
    # --- device guards ---
    cc = tuple(get_compute_capability(x.device))
    if not (allow["min_cc"] <= cc <= allow["max_cc"]):
        return "compute_capability"
    # The NH128 path is one cooperative launch of ``nheads + 8`` CTAs whose
    # shared-memory footprint caps residency at one CTA per SM, so a device
    # with fewer SMs can never hold the whole grid: refuse here instead of
    # letting the native residency check raise on every call.  (The NH64
    # path is a plain launch with no residency requirement.)  The native
    # kernel still enforces the exact occupancy envelope at first launch.
    if (
        nheads >= 96
        and torch.cuda.get_device_properties(x.device).multi_processor_count
        < nheads + 8
    ):
        return "insufficient_sm_count"
    if _native_device is not None and x.device.index != _native_device:
        return "native_state_other_device"
    # CUDA-graph rule: the first call JIT-compiles and initializes the
    # cooperative-launch state on the host — never under capture.  Query the
    # capture state on the operand's device: the launch runs on that
    # device's current stream, not on the host-current device's.
    with torch.cuda.device(x.device):
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing and (x.device.index or 0, nheads) not in _warmed:
        return "capture_unwarmed"
    return None


def maybe_run_specialized(
    *,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    dt_limit: Tuple[float, float],
    initial_states: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    return_final_states: bool,
    num_seqs: Optional[int],
    cu_seqlens: Optional[torch.Tensor],
    extras_none: bool,
    geo: Tuple[int, int, int, int],
) -> Tuple[Optional[Tuple[torch.Tensor, None]], Optional[str]]:
    """Dispatch to the specialized kernel when the guard passes.

    Returns ``((out, None), None)`` when served (token-major
    ``[1, seqlen, nheads, headdim]`` output, no final states), else
    ``(None, reason)`` with the reason recorded in ``fallback_reasons``.
    """
    _stats["calls"] += 1
    reason = _guard_call(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        dt_limit,
        initial_states,
        out,
        return_final_states,
        num_seqs,
        cu_seqlens,
        extras_none,
        geo,
    )
    if reason is not None:
        return None, _bump_fallback(reason)
    run = _kernel_entry()
    T, nheads, headdim = x.shape[1], x.shape[2], x.shape[3]
    if out is None:
        out = torch.empty(
            (1, T, nheads, headdim), dtype=torch.bfloat16, device=x.device
        )
    run(
        x.view(T, nheads, headdim),
        dt.view(T, nheads),
        A,
        B.view(T, B.shape[2], B.shape[3]),
        C.view(T, C.shape[2], C.shape[3]),
        D,
        dt_bias,
        out.view(T, nheads, headdim),
    )
    global _native_device
    _native_device = x.device.index
    _warmed.add((x.device.index or 0, nheads))
    _stats["specialized_dispatches"] += 1
    return (out, None), None


def warmup(nheads: int, device=None) -> None:
    """Pre-build the module and run one eager call per nheads variant
    (CUDA-graph rule: an unwarmed variant never dispatches under capture)."""
    global _native_device
    allow = _load_allowlist()
    if allow is None or nheads not in allow["nheads"]:
        return
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        return
    if isinstance(device, int):  # 0 is a device index, never falsy
        device = torch.device("cuda", device)
    elif device is not None:
        device = torch.device(device)
        if device.type != "cuda":
            return
    if not torch.cuda.is_available():
        return
    if device is None or device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if _native_device is not None and device.index != _native_device:
        return
    cc = tuple(get_compute_capability(device))
    if not (allow["min_cc"] <= cc <= allow["max_cc"]):
        return
    # Mirror the dispatch guard's cooperative-residency refusal: warming the
    # NH128 variant on a device that cannot hold the whole grid would raise.
    if (
        nheads >= 96
        and torch.cuda.get_device_properties(device).multi_processor_count < nheads + 8
    ):
        return
    run = _kernel_entry()
    hd, ds, ng = allow["headdim"], allow["dstate"], allow["ngroups"]
    T = allow["chunk_size"]
    x = torch.zeros(T, nheads, hd, dtype=torch.bfloat16, device=device)
    dt = torch.zeros(T, nheads, dtype=torch.bfloat16, device=device)
    A = -torch.ones(nheads, dtype=torch.float32, device=device)
    B = torch.zeros(T, ng, ds, dtype=torch.bfloat16, device=device)
    C = torch.zeros(T, ng, ds, dtype=torch.bfloat16, device=device)
    D = torch.zeros(nheads, dtype=torch.bfloat16, device=device)
    db = torch.zeros(nheads, dtype=torch.bfloat16, device=device)
    y = torch.empty(T, nheads, hd, dtype=torch.bfloat16, device=device)
    run(x, dt, A, B, C, D, db, y)
    torch.cuda.synchronize(device)
    _native_device = device.index
    _warmed.add((device.index or 0, nheads))
