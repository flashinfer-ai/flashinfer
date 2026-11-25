# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Input contracts and cuDNN execution for FlashInfer batch-invariant GEMMs."""

from __future__ import annotations

import functools
import logging
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

from ...batch_invariant.support import (
    _check_device_arch as check_device,
    byte_range,
    compute_capability,
    register_self_served,
    unsupported,
)
from ...utils import device_support_pdl
from .generator import Knobs
from .heuristic import family_of
from .schedule import chunks_for, fp4_class_for, mx_class_for
from .spec import BLOCK_K, NVFP4_FMTS, SCALE_BLOCK_N, k_granule, sf_bytes

logger = logging.getLogger(__name__)

CLASS = "gemm.bf16"
#: The class family of :func:`flashinfer.bmm_fp8` (its two formats are the classes ``gemm.e4m3`` and ``gemm.e5m2``).
CLASS_FP8 = "gemm.fp8"
#: The class family of :func:`flashinfer.mm_mxfp8`: one class per block-scaled instruction, ``gemm.mxfp8.*`` on SM100 /
#: SM103 and ``gemm.mxfp8.*64`` on SM107 (`schedule.mx_class_for`).
CLASS_MXFP8 = "gemm.mxfp8"
#: The class family of :func:`flashinfer.mm_fp4`: the NVFP4 classes ``gemm.nvfp4.*`` (and ``gemm.nvfp4.*128`` on SM107 for
#: the unsplit weight shapes, `schedule.fp4_class_for`) and the MXFP4 classes ``gemm.mxfp4.*`` / ``gemm.mxfp4.*128``.
CLASS_FP4 = "gemm.fp4"
#: The class family of :func:`flashinfer.mm_mxfp8_mxfp4`: ``gemm.mxfp4w8a.*`` on SM100 / SM103 and ``gemm.mxfp4w8a.*64`` on SM107 (`schedule.mx_class_for`).
CLASS_MXW48 = "gemm.mxfp4w8a"
register_self_served(
    CLASS_MXW48, "mm_mxfp8_mxfp4"
)  # both flag values run the class: its errors advise no fallback
#: The compute capabilities the classes are stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
MM_BF16_ARCHS = ((10, 0), (10, 3), (10, 7))
BMM_FP8_ARCHS = MM_BF16_ARCHS
MM_MXFP8_ARCHS = MM_BF16_ARCHS
MM_FP4_ARCHS = MM_BF16_ARCHS
MM_MXW48_ARCHS = MM_BF16_ARCHS
_ALIGN = 8  # bf16 / fp16 elements per 16 bytes
_ALIGN_FP8 = 16  # fp8 elements per 16 bytes
_SF_BLOCK = 32  # elements per UE8M0 block scale of the MXFP8 classes: K is a whole number of blocks
_ALIGN_BYTES = 16  # the operand base alignment the kernels assume (TMA)
_PACKED_FP4_DTYPES = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", None))
FP8_FMT_OF_DTYPE = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}
OUT_FMT_OF_DTYPE = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
# Public quantized GEMMs return BF16 or FP16. Low-level diagnostics may request
# FP32, which also requires a cuDNN engine supporting that output type.
BMM_FP8_OUT_DTYPES = (torch.bfloat16, torch.float16)
MM_FP8_OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
MM_MXFP8_API_OUT_DTYPES = (torch.bfloat16, torch.float16)
MM_MXFP8_OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
MM_FP4_API_OUT_DTYPES = MM_MXFP8_API_OUT_DTYPES
MM_FP4_OUT_DTYPES = MM_MXFP8_OUT_DTYPES
MM_MXW48_API_OUT_DTYPES = MM_MXFP8_API_OUT_DTYPES
MM_MXW48_OUT_DTYPES = MM_MXFP8_OUT_DTYPES
#: The block-scale fp8 class of :func:`flashinfer.gemm.gemm_fp8_nt_groupwise` (``scale_granularity_mnk = (1, 128, 128)``).
CLASS_FP8BS = "gemm.e4m3.kblock128"
GEMM_FP8BS_ARCHS = MM_BF16_ARCHS
GEMM_FP8BS_API_OUT_DTYPES = MM_MXFP8_API_OUT_DTYPES
GEMM_FP8BS_OUT_DTYPES = MM_MXFP8_OUT_DTYPES


def _layout_of_b(b: torch.Tensor, K: int, N: int, cls: str = CLASS) -> str:
    """The layout of a (K, N) weight: ``"n"`` row-major, ``"k"`` the transpose of a row-major (N, K) tensor."""
    s0, s1 = b.stride()
    if s1 == 1 and s0 == N:
        return "n"
    if s0 == 1 and s1 == K:
        return "k"
    raise unsupported(
        cls,
        f"B with strides {tuple(b.stride())}",
        "B (K, N) row-major or the transpose of a row-major (N, K) tensor",
    )


def _device_key(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def sm_count_of(device: torch.device) -> int:
    return _sm_count_cached(_device_key(device))


# the devices' answers to the occupancy query, keyed (device index, cluster size). A failed query is retried on the
# next call up to _QUERY_ATTEMPTS times; then the part's nominal capacity is kept in its place (logged once), so a device
# whose query cannot succeed is neither asked again nor re-ranked on every call.
_CO_RESIDENT: Dict[Tuple[int, int], int] = {}
_QUERY_FAILURES: Dict[Tuple[int, int], int] = {}
_QUERY_ATTEMPTS = 3
_REGISTERED_DEVICES: set = set()


def _query_co_resident_clusters(device_index: int, cluster_size: int) -> int:
    """The CuTe DSL's GPC-aware occupancy query for a one-CTA-per-SM body, with the device's primary context made
    current on the calling thread for its duration (the query needs one; the previous current context is restored).
    No stream work and no synchronisation: legal during CUDA graph capture."""
    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cu  # the DSL's own driver binding

    def ok(result):
        err, *values = result
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{err}")
        return values[0] if values else None

    ok(cu.cuInit(0))
    dev = ok(cu.cuDeviceGet(device_index))
    prev = ok(cu.cuCtxGetCurrent())
    ctx = ok(cu.cuDevicePrimaryCtxRetain(dev))
    try:
        ok(cu.cuCtxSetCurrent(ctx))
        n = int(
            cutlass_utils.HardwareInfo(device_id=device_index).get_max_active_clusters(
                cluster_size
            )
        )
    finally:
        ok(cu.cuCtxSetCurrent(prev))
        ok(cu.cuDevicePrimaryCtxRelease(dev))
    if n <= 0:
        raise RuntimeError(f"the occupancy query returned {n}")
    return n


def _co_resident_clusters(
    device_index: int, cluster_size: int, sm_count: int, family: str
) -> int:
    """How many clusters of ``cluster_size`` CTAs the device can host at once: the device's answer, asked once and kept.
    A failed query (logged) gives the part's nominal capacity; the next call asks again, up to ``_QUERY_ATTEMPTS`` times
    in all, after which the nominal capacity is kept for the process like an answer (a launch parameter: the bits never
    depend on it)."""
    from .heuristic import capacity_for

    key = (device_index, cluster_size)
    if key not in _CO_RESIDENT:
        try:
            _CO_RESIDENT[key] = _query_co_resident_clusters(device_index, cluster_size)
        except Exception as e:  # noqa: BLE001
            nominal = capacity_for(family, sm_count).get(cluster_size) or max(
                1, sm_count // cluster_size
            )
            _QUERY_FAILURES[key] = _QUERY_FAILURES.get(key, 0) + 1
            if _QUERY_FAILURES[key] >= _QUERY_ATTEMPTS:
                _CO_RESIDENT[key] = nominal
                logger.warning(
                    "batch_invariant: the co-resident cluster query failed %d times on device %d (%s: %s); the part's "
                    "nominal cluster capacity (%d clusters of %d CTAs) is kept for this process",
                    _QUERY_FAILURES[key],
                    device_index,
                    type(e).__name__,
                    e,
                    nominal,
                    cluster_size,
                )
            else:
                logger.warning(
                    "batch_invariant: the co-resident cluster query failed on device %d (%s: %s); the part's nominal "
                    "cluster capacity is used for this call",
                    device_index,
                    type(e).__name__,
                    e,
                )
            return nominal
    return _CO_RESIDENT[key]


def _register_device(device_index: int, sm_count: int, family: str) -> None:
    """Once per device (again after a failed query, until its retries are spent): the device's L2 size and the
    co-resident cluster counts of the cluster sizes the space uses, into the heuristic's tables; a query's answer, or
    the nominal capacity kept after the retries, is recorded."""
    if device_index in _REGISTERED_DEVICES:
        return
    from .generator import CLUSTERS
    from .heuristic import register_capacity, register_l2

    register_l2(
        family, sm_count, torch.cuda.get_device_properties(device_index).L2_cache_size
    )
    complete = True
    for size in sorted({cm * cn for cm, cn in CLUSTERS}):
        n = _co_resident_clusters(device_index, size, sm_count, family)
        if (device_index, size) in _CO_RESIDENT:
            register_capacity(family, sm_count, size, n)
        else:
            complete = False
    if complete:
        _REGISTERED_DEVICES.add(device_index)


def persistent_clusters(device: torch.device, cfg: Knobs) -> int:
    """The persistent grid in clusters for ``cfg`` on ``device``: the device's co-resident cluster count for the knobs'
    cluster shape (cached per device and cluster size), so that the compiled grid is what the device can run at once;
    a launch parameter that decides which CTA computes which tile and nothing else (the bits never depend on it). A
    grid of ``SMs // size`` would leave its surplus clusters to run as a serial tail under the static persistent
    assignment whenever a GPC cannot host a whole cluster of every SM."""
    if cfg.grid_clusters:
        return cfg.grid_clusters
    index = device.index if device.index is not None else torch.cuda.current_device()
    sm, fam = part_of(device)
    return _co_resident_clusters(index, cfg.cluster_m * cfg.cluster_n, sm, fam)


def part_of(device: torch.device) -> Tuple[int, str]:
    """The part the dispatcher models for ``device``: its SM count and architecture family (`heuristic.family_of`); the
    first call for a device also records its L2 size and co-resident cluster capacity for the heuristic."""
    index = _device_key(device)
    sm, fam = _part_cached(index)
    _register_device(index, sm, fam)
    return sm, fam


# Both are device properties, fixed for the process, and are read on every call: cached per device index.
@functools.lru_cache(maxsize=None)
def _sm_count_cached(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _part_cached(index: int) -> Tuple[int, str]:
    return _sm_count_cached(index), family_of(
        compute_capability(torch.device("cuda", index))
    )


def prepare_operands(
    a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str, int, int, int]:
    """The bf16 class's typed checks and operand normalisation, shared by the direct entry and the public entry: device
    and architecture, dtypes, 2-D shapes with matching K, 16-byte TMA alignment of K and N, A's row placement, B's layout
    (``"n"``: (K, N) row-major; ``"k"``: the transpose of a row-major (N, K) weight), the output's dtype / shape / layout
    (allocated when absent). Returns ``(a, out, b_major, M, N, K)``; raises :class:`BatchInvariantUnsupportedError`."""
    check_device(a.device, CLASS, MM_BF16_ARCHS)
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise unsupported(CLASS, f"dtypes ({a.dtype}, {b.dtype})", "bf16 A and B")
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            CLASS, f"{a.dim()}-D A and {b.dim()}-D B", "2-D operands (M, K) and (K, N)"
        )
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise unsupported(CLASS, f"A (M, {K}) with B ({K2}, N)", "matching K")
    if b.device != a.device:
        raise unsupported(
            CLASS, "operands on different devices", "A and B on one device"
        )
    if K == 0 or N == 0:
        raise unsupported(
            CLASS, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % _ALIGN or N % _ALIGN:
        raise unsupported(
            CLASS,
            f"K = {K}, N = {N}",
            f"K and N multiples of {_ALIGN} (16-byte TMA rows)",
        )
    a = _aligned_rows(a, M, K, _ALIGN)
    b_major = _layout_of_b(b, K, N)
    _check_base(b, "B", CLASS)
    out = _prepare_out(out, M, N, torch.bfloat16, a.device, CLASS)
    return a, out, b_major, M, N, K


def _aligned_rows(a: torch.Tensor, M: int, K: int, align: int) -> torch.Tensor:
    """A's row placement: the kernel reads 16-byte-aligned rows from a 16-byte-aligned base (a single row's row stride is
    immaterial); a placement, not numerics."""
    s0, s1 = a.stride()
    aligned_base = a.data_ptr() % _ALIGN_BYTES == 0
    if s1 == 1 and s0 % align == 0 and aligned_base:
        return a
    if M == 1 and s1 == 1 and aligned_base:
        return a.as_strided((1, K), (K, 1), a.storage_offset())
    if a.dtype == _PACKED_FP4_DTYPES[1]:
        a = a.view(torch.uint8)
    return a.clone(memory_format=torch.contiguous_format)


def _check_base(
    t: torch.Tensor, name: str, cls: str, what: Optional[str] = None
) -> None:
    """The operand base alignment the kernels assume (TMA): a placement check, not numerics."""
    if t.data_ptr() % _ALIGN_BYTES:
        raise unsupported(
            cls,
            f"{name} at a base address that is not 16-byte aligned",
            f"a 16-byte-aligned {what or name}",
        )


def _prepare_out(
    out: Optional[torch.Tensor],
    M: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    cls: str,
) -> torch.Tensor:
    if out is None:
        return torch.empty(M, N, dtype=dtype, device=device)
    if out.dtype != dtype or out.shape != (M, N):
        raise unsupported(
            cls,
            f"out {tuple(out.shape)} {out.dtype}",
            f"a {OUT_FMT_OF_DTYPE.get(dtype, dtype)} (M, N) = ({M}, {N}) output",
        )
    if out.device != device:
        raise unsupported(cls, "out on another device", "the output on A's device")
    _check_base(out, "out", cls, "output")
    s0, s1 = out.stride()
    if s1 != 1 or (M > 1 and s0 != N):
        raise unsupported(
            cls,
            f"out with strides {(s0, s1)}",
            "a row-major contiguous output",
        )
    if M == 1 and s0 != N:
        out = out.as_strided(
            (1, N), (N, 1), out.storage_offset()
        )  # the same elements, an aligned row stride
    return out


def _require_disjoint(cls: str, writes, reads) -> None:
    """Check the full byte span of each buffer, including gaps between strided rows."""
    written = [(byte_range(t), name) for t, name in writes]
    read = [(byte_range(t), name) for t, name in reads]
    for i, ((lo, hi), name) in enumerate(written):
        for (other_lo, other_hi), other in read + written[i + 1 :]:
            if lo < other_hi and other_lo < hi:
                raise unsupported(
                    cls, f"{other} overlapping {name}", f"{other} disjoint from {name}"
                )


def _check_scale(
    s: torch.Tensor, name: str, device: torch.device, cls: str
) -> torch.Tensor:
    """A per-tensor scale: a 1-element fp32 tensor on the operands' device, as a (1,) view."""
    if not isinstance(s, torch.Tensor):
        raise unsupported(
            cls, f"{name} of type {type(s).__name__}", f"{name} a torch.Tensor"
        )
    if s.dtype != torch.float32 or s.numel() != 1:
        raise unsupported(
            cls,
            f"{name} {tuple(s.shape)} {s.dtype}",
            f"a 1-element float32 {name} (one per-tensor scale)",
        )
    if s.device != device:
        raise unsupported(cls, f"{name} on another device", f"{name} on A's device")
    return s if s.shape == (1,) else s.reshape(1)


def prepare_bias(
    bias: Optional[torch.Tensor], N: int, a: torch.Tensor
) -> Optional[torch.Tensor]:
    """The class's typed checks of the bias (what the operator's backends accept): a bf16 vector of ``N`` elements on
    A's device, contiguous, at a 16-byte-aligned base; ``None`` is the unbiased chain. Raises
    :class:`BatchInvariantUnsupportedError`; never copies (a bias is an operand the kernel reads as given)."""
    if bias is None:
        return None
    if not isinstance(bias, torch.Tensor):
        raise unsupported(
            CLASS, f"a bias of type {type(bias).__name__}", "a bf16 torch.Tensor bias"
        )
    if bias.dtype != torch.bfloat16:
        raise unsupported(CLASS, f"a {bias.dtype} bias", "a bf16 bias")
    if bias.dim() != 1 or bias.shape[0] != N:
        raise unsupported(
            CLASS, f"a bias of shape {tuple(bias.shape)}", f"a bias of shape ({N},)"
        )
    if bias.device != a.device:
        raise unsupported(CLASS, "a bias on another device", "the bias on A's device")
    if bias.stride(0) != 1:
        raise unsupported(
            CLASS, f"a bias with stride {bias.stride(0)}", "a contiguous bias"
        )
    if bias.data_ptr() % _ALIGN_BYTES:
        raise unsupported(
            CLASS,
            "a bias at a base address that is not 16-byte aligned",
            "a 16-byte-aligned bias",
        )
    return bias


def _check_plan_options(cls, config, chunks, n, k, fmt):
    if config is not None:
        raise unsupported(
            cls, "a FlashInfer kernel config", "cuDNN batch-invariant plan selection"
        )
    if chunks is not None and chunks != chunks_for(n, k, fmt):
        raise unsupported(
            cls,
            f"chunks = {chunks}",
            "the cuDNN batch-invariant chunk count for this weight shape",
        )


def _checked_format(cls, requested, expected):
    if requested is not None and requested != expected:
        raise unsupported(
            cls,
            f"in_fmt={requested!r}",
            f"cuDNN's {expected!r} class for this device and weight",
        )
    return expected


def mm_bf16_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    bias: Optional[torch.Tensor] = None,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run BF16 A @ B with optional BF16 bias using the cuDNN batch-invariant plan."""
    from .cudnn_gemm import bf16_gemm

    return bf16_gemm(a, b, out, bias, config, chunks, workspace)


def _fp8_fmt(a: torch.Tensor, b: torch.Tensor) -> str:
    in_fmt = FP8_FMT_OF_DTYPE.get(a.dtype)
    if in_fmt is None or b.dtype != a.dtype:
        raise unsupported(
            CLASS_FP8,
            f"dtypes ({a.dtype}, {b.dtype})",
            "A and B both float8_e4m3fn or both float8_e5m2",
        )
    return in_fmt


def prepare_fp8_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str, int, int, int
]:
    """The fp8 classes' typed checks for one product: device and architecture, one fp8 dtype for A and B (e4m3 or e5m2:
    the class), 2-D shapes with matching K, 16-byte TMA alignment (K a multiple of 16; N of 8, or of 16 for a row-major
    (K, N) weight), A's row placement, B's layout (``"k"``: the transpose of a row-major (N, K) weight; ``"n"``: (K, N)
    row-major), the two per-tensor scales, the output's dtype / shape / layout (allocated when absent). Returns
    ``(a, a_scale, b_scale, out, in_fmt, b_major, M, N, K)``."""
    check_device(a.device, CLASS_FP8, BMM_FP8_ARCHS)
    in_fmt = _fp8_fmt(a, b)
    cls = f"gemm.{in_fmt}"
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            cls, f"{a.dim()}-D A and {b.dim()}-D B", "2-D operands (M, K) and (K, N)"
        )
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise unsupported(cls, f"A (M, {K}) with B ({K2}, N)", "matching K")
    if b.device != a.device:
        raise unsupported(cls, "operands on different devices", "A and B on one device")
    if K == 0 or N == 0:
        raise unsupported(
            cls, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % _ALIGN_FP8 or N % _ALIGN:
        raise unsupported(
            cls,
            f"K = {K}, N = {N}",
            f"K a multiple of {_ALIGN_FP8} and N of {_ALIGN} (16-byte TMA rows)",
        )
    a = _aligned_rows(a, M, K, _ALIGN_FP8)
    b_major = _layout_of_b(b, K, N, cls)
    _check_base(b, "B", cls)
    if b_major == "n" and N % _ALIGN_FP8:
        raise unsupported(
            cls,
            f"N = {N} with a row-major (K, N) weight",
            f"N a multiple of {_ALIGN_FP8} (16-byte TMA rows of the weight)",
        )
    a_scale = _check_scale(a_scale, "a_scale", a.device, cls)
    b_scale = _check_scale(b_scale, "b_scale", a.device, cls)
    if out_dtype not in MM_FP8_OUT_DTYPES:
        raise unsupported(cls, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    out = _prepare_out(out, M, N, out_dtype, a.device, cls)
    return a, a_scale, b_scale, out, in_fmt, b_major, M, N, K


def mm_fp8_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one per-tensor FP8 product with FP32 scales."""
    out_fresh = out is None
    caller_a = a
    a, a_scale, b_scale, out, in_fmt, _, M, N, K = prepare_fp8_operands(
        a, b, a_scale, b_scale, out, out_dtype
    )
    cls = f"gemm.{in_fmt}"
    if M == 0:
        return out  # no rows: nothing to launch
    if config is not None or chunks is not None:
        _check_plan_options(cls, config, chunks, N, K, in_fmt)
    from flashinfer.gemm.batch_invariant.cudnn_gemm import quantized_gemm

    return quantized_gemm(
        cls,
        a,
        b,
        a_scale,
        b_scale,
        out,
        out_fresh,
        caller_a=caller_a,
        workspace=workspace,
    )


def _layout_of_B_batched(B: torch.Tensor, K: int, N: int, cls: str) -> str:
    """The per-element layout of a (b, K, N) batch of weights: column-major (the transpose of a row-major (b, N, K)
    weight, ``bmm_fp8``'s documented layout) is the kernel's ``k`` layout; row-major is ``n``."""
    if B.stride(1) == 1 and B.stride(2) == K:
        return "k"
    if B.stride(2) == 1 and B.stride(1) == N:
        return "n"
    raise unsupported(
        cls,
        f"B with strides {tuple(B.stride())}",
        "B (b, K, N) column-major (the transpose of a row-major (b, N, K) weight) or row-major",
    )


def prepare_bmm_fp8_operands(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str, int, int, int, int
]:
    """The batched fp8 classes' typed checks, shared by the direct and public entries: device and architecture,
    one fp8 dtype for A and B, 3-D shapes (b, M, K) and (b, K, N), 16-byte TMA alignment (K a multiple of 16, N of 8, or of
    16 for a row-major B; the batch strides too), A's row placement, B's per-element layout, the two per-tensor scales,
    the output (b, M, N) in bf16 or fp16 (allocated when absent). Returns
    ``(A, A_scale, B_scale, out, in_fmt, b_major, b, M, N, K)``."""
    check_device(A.device, CLASS_FP8, BMM_FP8_ARCHS)
    in_fmt = _fp8_fmt(A, B)
    cls = f"gemm.{in_fmt}"
    if A.dim() != 3 or B.dim() != 3:
        raise unsupported(
            cls,
            f"{A.dim()}-D A and {B.dim()}-D B",
            "3-D operands (b, M, K) and (b, K, N)",
        )
    b, M, K = A.shape
    b2, K2, N = B.shape
    if K != K2 or b != b2:
        raise unsupported(
            cls, f"A {tuple(A.shape)} with B {tuple(B.shape)}", "matching batch and K"
        )
    if B.device != A.device:
        raise unsupported(cls, "operands on different devices", "A and B on one device")
    if K == 0 or N == 0:
        raise unsupported(
            cls, f"K = {K}, N = {N}", "K > 0 and N > 0 (only b and M may be empty)"
        )
    if K % _ALIGN_FP8 or N % _ALIGN:
        raise unsupported(
            cls,
            f"K = {K}, N = {N}",
            f"K a multiple of {_ALIGN_FP8} and N of {_ALIGN} (16-byte TMA rows)",
        )
    if (
        A.stride(2) != 1
        or A.stride(1) % _ALIGN_FP8
        or A.stride(0) % _ALIGN_FP8
        or A.data_ptr() % _ALIGN_BYTES
    ):
        A = A.clone(
            memory_format=torch.contiguous_format
        )  # a placement: the kernel reads 16-byte-aligned rows and batch slices from a 16-byte-aligned base
    b_major = _layout_of_B_batched(B, K, N, cls) if b else "k"
    _check_base(B, "B", cls)
    if b_major == "n" and N % _ALIGN_FP8:
        raise unsupported(
            cls,
            f"N = {N} with a row-major (b, K, N) weight",
            f"N a multiple of {_ALIGN_FP8} (16-byte TMA rows of the weight)",
        )
    if b and B.stride(0) % _ALIGN_FP8:
        raise unsupported(
            cls,
            f"B with batch stride {B.stride(0)}",
            "16-byte-aligned batch slices of B",
        )
    A_scale = _check_scale(A_scale, "A_scale", A.device, cls)
    B_scale = _check_scale(B_scale, "B_scale", A.device, cls)
    if dtype not in BMM_FP8_OUT_DTYPES:
        raise unsupported(cls, f"dtype={dtype}", "bf16 or fp16 output")
    if out is None:
        out = torch.empty(b, M, N, dtype=dtype, device=A.device)
    else:
        if out.dtype != dtype or tuple(out.shape) != (b, M, N):
            raise unsupported(
                cls,
                f"out {tuple(out.shape)} {out.dtype}",
                f"a {OUT_FMT_OF_DTYPE[dtype]} (b, M, N) = ({b}, {M}, {N}) output",
            )
        if out.device != A.device:
            raise unsupported(cls, "out on another device", "the output on A's device")
        _check_base(out, "out", cls, "output")
        if (
            out.stride(2) != 1
            or (M > 1 and out.stride(1) != N)
            or (b > 1 and (out.stride(0) % _ALIGN or out.stride(0) < M * N))
        ):
            raise unsupported(
                cls,
                f"out with strides {tuple(out.stride())}",
                "a row-major (b, M, N) output with 16-byte-aligned batch slices",
            )
    return A, A_scale, B_scale, out, in_fmt, b_major, b, M, N, K


def bmm_fp8_batch_invariant(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    *,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a batch of per-tensor FP8 products with shared FP32 scales."""
    out_fresh = out is None
    caller_a = A
    A, A_scale, B_scale, out, in_fmt, _, b, M, N, K = prepare_bmm_fp8_operands(
        A, B, A_scale, B_scale, dtype, out
    )
    cls = f"gemm.{in_fmt}"
    if b == 0 or M == 0:
        return out  # nothing to launch
    if config is not None or chunks is not None:
        _check_plan_options(cls, config, chunks, N, K, in_fmt)
    from flashinfer.gemm.batch_invariant.cudnn_gemm import quantized_gemm

    return quantized_gemm(
        cls,
        A,
        B,
        A_scale,
        B_scale,
        out,
        out_fresh,
        caller_a=caller_a,
        workspace=workspace,
    )


def _swizzled_scale(
    sf: torch.Tensor,
    rows: int,
    K: int,
    name: str,
    device: torch.device,
    cls: str,
    in_fmt: str = "mxfp8",
) -> torch.Tensor:
    """Check the F8_128x4 byte storage. cuDNN's descriptor supplies its logical geometry."""
    if not isinstance(sf, torch.Tensor) or not (
        sf.dtype == torch.uint8
        or (in_fmt in NVFP4_FMTS and sf.dtype == torch.float8_e4m3fn)
    ):
        raise unsupported(
            cls,
            f"{name} of type {type(sf).__name__}"
            + (f" / dtype {sf.dtype}" if isinstance(sf, torch.Tensor) else ""),
            f"{name} a uint8 tensor of UE8M0 scale factors in the 128x4 swizzled layout "
            "(mxfp8_quantize(..., is_sf_swizzled_layout=True))",
        )
    if sf.device != device:
        raise unsupported(cls, f"{name} on another device", f"{name} on A's device")
    if not sf.is_contiguous() and (sf.dim() != 2 or sf.stride() != (1, sf.size(0))):
        raise unsupported(
            cls,
            f"{name} with strides {tuple(sf.stride())}",
            f"{name} a contiguous scale-factor buffer",
        )
    need = sf_bytes(rows, K, in_fmt)
    if sf.numel() < need:
        raise unsupported(
            cls,
            f"{name} of {sf.numel()} bytes for a ({rows}, {K}) operand",
            f"at least {need} bytes: the 128x4 swizzled layout holds ceil(rows / 128) * 128 rows x 4 * ceil(K / (4 x block)) blocks",
        )
    _check_base(sf, name, cls)
    return sf


def prepare_mxfp8_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_dtypes: Tuple[torch.dtype, ...] = MM_MXFP8_OUT_DTYPES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """The MXFP8 class's typed checks for one product, shared by the direct and public entries: device and
    architecture, e4m3 A and B, 2-D shapes with matching K, K a multiple of 32 (one whole scale block, hence 16-byte TMA
    rows) and N of 8, A's row placement, B the transpose of a row-major (N, K) weight (the block-scaled instruction takes
    K-major operands only), the two scale-factor buffers, the output's dtype / shape / layout (allocated when absent).
    Returns ``(a, sfa, sfb, out, M, N, K)`` with the original scale buffers. K is a multiple of 32; where it is not a multiple of the class's step (64 on SM107) the last step is partial: its
    padded values are TMA zero fill, its scales are read from the scale buffers' padding, so those bytes must be finite
    (the quantizer writes zeros; a NaN or Inf there poisons the row)."""
    cls = CLASS_MXFP8
    check_device(a.device, cls, MM_MXFP8_ARCHS)
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise unsupported(
            cls,
            f"dtypes ({a.dtype}, {b.dtype})",
            "float8_e4m3fn A and B (the output of mxfp8_quantize)",
        )
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            cls, f"{a.dim()}-D A and {b.dim()}-D B", "2-D operands (M, K) and (K, N)"
        )
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise unsupported(cls, f"A (M, {K}) with B ({K2}, N)", "matching K")
    if b.device != a.device:
        raise unsupported(cls, "operands on different devices", "A and B on one device")
    if K == 0 or N == 0:
        raise unsupported(
            cls, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % _SF_BLOCK or N % _ALIGN:
        raise unsupported(
            cls,
            f"K = {K}, N = {N}",
            f"K a multiple of {_SF_BLOCK} (one whole block scale) and N of {_ALIGN} (16-byte rows of the output)",
        )
    a = _aligned_rows(a, M, K, _ALIGN_FP8)
    if _layout_of_b(b, K, N, cls) != "k":
        raise unsupported(
            cls,
            "a row-major (K, N) weight",
            "the transpose of a row-major (N, K) weight (the block-scaled instruction takes K-major operands only)",
        )
    _check_base(b, "B", cls)
    sfa = _swizzled_scale(a_descale, M, K, "a_descale", a.device, cls)
    sfb = _swizzled_scale(b_descale, N, K, "b_descale", a.device, cls)
    if out_dtype not in out_dtypes:
        raise unsupported(cls, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    out = _prepare_out(out, M, N, out_dtype, a.device, cls)
    return a, sfa, sfb, out, M, N, K


def mx_format_for(device: torch.device, in_fmt: Optional[str] = None) -> str:
    family = family_of(compute_capability(device))
    return _checked_format(CLASS_MXFP8, in_fmt, mx_class_for("mxfp8", family))


def mm_mxfp8_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
    in_fmt: Optional[str] = None,
) -> torch.Tensor:
    """Run MXFP8 A @ B with swizzled UE8M0 scales."""
    out_fresh = out is None
    caller_a = a
    a, b, sfa, sfb, _, out, M, N, K, oversized_scales = _prepare_block_scaled_gemm(
        a, b, a_descale, b_descale, None, out, out_dtype, "mxfp8"
    )
    if in_fmt is not None or config is not None or chunks is not None:
        in_fmt = mx_format_for(a.device, in_fmt)
        _check_plan_options(CLASS_MXFP8, config, chunks, N, K, in_fmt)
    if M == 0:
        return out
    from flashinfer.gemm.batch_invariant.cudnn_gemm import quantized_gemm

    return quantized_gemm(
        CLASS_MXFP8,
        a,
        b,
        sfa,
        sfb,
        out,
        out_fresh,
        caller_a=caller_a,
        block_size=32,
        workspace=workspace,
        oversized_scales=oversized_scales,
    )


def _packed_fp4(t: torch.Tensor, name: str, cls: str = CLASS_FP4) -> torch.Tensor:
    """Accept the byte storage of two FP4 elements in either supported torch dtype."""
    if t.dtype in _PACKED_FP4_DTYPES:
        return t
    raise unsupported(
        cls,
        f"{name} of dtype {t.dtype}",
        "packed fp4 operands (uint8, or float4_e2m1fn_x2: two e2m1 values per byte)",
    )


def prepare_fp4_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    use_nvfp4: bool,
    out_dtypes: Tuple[torch.dtype, ...] = MM_FP4_OUT_DTYPES,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    int,
    int,
    int,
]:
    """The 4-bit classes' typed checks for one product, shared by the direct and public entries: device and
    architecture, packed-fp4 A ``(M, K / 2)`` row-major and B ``(K / 2, N)`` as the transpose of a row-major ``(N, K / 2)``
    weight (the block-scaled instruction takes K-major operands only), K a multiple of 32 (whole block scales of either
    format, 16-byte rows) and N of 8, the two scale-factor buffers (e4m3 per 16 elements for NVFP4, UE8M0 per 32 for
    MXFP4, the 128x4 swizzled layout), ``alpha`` a 1-element fp32 tensor (None: 1.0; a per-token alpha is unsupported), the output's dtype / shape / layout (allocated when absent). Returns ``(a, b, sfa, sfb, alpha, out, M, N, K)``
    with the original packed operand and scale dtypes."""
    cls = CLASS_FP4
    check_device(a.device, cls, MM_FP4_ARCHS)
    a = _packed_fp4(a, "A")
    b = _packed_fp4(b, "B")
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            cls,
            f"{a.dim()}-D A and {b.dim()}-D B",
            "2-D operands (M, K / 2) and (K / 2, N)",
        )
    M, Kh = a.shape
    Kh2, N = b.shape
    if Kh != Kh2:
        raise unsupported(cls, f"A (M, {Kh}) with B ({Kh2}, N)", "matching K")
    if b.device != a.device:
        raise unsupported(cls, "operands on different devices", "A and B on one device")
    K = 2 * Kh
    if K == 0 or N == 0:
        raise unsupported(
            cls, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % _SF_BLOCK or N % _ALIGN:
        raise unsupported(
            cls,
            f"K = {K}, N = {N}",
            f"K a multiple of {_SF_BLOCK} (whole block scales, 16-byte rows) and N of {_ALIGN} (16-byte rows of the output)",
        )
    a = _aligned_rows(a, M, Kh, _ALIGN_FP8)  # 16 bytes of packed values per row
    if _layout_of_b(b, Kh, N, cls) != "k":
        raise unsupported(
            cls,
            "a row-major (K, N) weight",
            "the transpose of a row-major (N, K) weight (the block-scaled instruction takes K-major operands only)",
        )
    _check_base(b, "B", cls)
    base = "nvfp4" if use_nvfp4 else "mxfp4"
    sfa = _swizzled_scale(a_descale, M, K, "a_descale", a.device, cls, base)
    sfb = _swizzled_scale(b_descale, N, K, "b_descale", a.device, cls, base)
    alpha = _alpha_tensor(alpha, a.device, cls)
    if out_dtype not in out_dtypes:
        raise unsupported(cls, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    out = _prepare_out(out, M, N, out_dtype, a.device, cls)
    return a, b, sfa, sfb, alpha, out, M, N, K


def _alpha_tensor(
    alpha: Optional[torch.Tensor], device: torch.device, cls: str
) -> Optional[torch.Tensor]:
    if alpha is None:
        return None
    if not isinstance(alpha, torch.Tensor) or alpha.dtype != torch.float32:
        raise unsupported(
            cls,
            f"alpha of type {type(alpha).__name__}"
            + (f" / dtype {alpha.dtype}" if isinstance(alpha, torch.Tensor) else ""),
            "a float32 tensor alpha",
        )
    if alpha.numel() != 1:
        raise unsupported(
            cls,
            f"an alpha of {alpha.numel()} elements",
            "one scalar alpha (a per-token alpha is not a class of the mode)",
        )
    if alpha.device != device:
        raise unsupported(cls, "alpha on another device", "alpha on A's device")
    return alpha


def fp4_format_for(
    device: torch.device, use_nvfp4: bool, N: int, K: int, in_fmt: Optional[str] = None
) -> str:
    family = family_of(compute_capability(device))
    expected = (
        fp4_class_for(N, K, family) if use_nvfp4 else mx_class_for("mxfp4", family)
    )
    return _checked_format(CLASS_FP4, in_fmt, expected)


# Cache validated geometries only when the input and output tensors can be bound
# unchanged. Addresses and aliasing are checked by the cuDNN execution plan on
# every call. The low address bits participate because normalization requires a
# 16-byte-aligned base even when the shape and strides did not change.
_VALIDATED_BLOCK_SCALED_LAYOUTS: OrderedDict[tuple, tuple[int, int, int]] = (
    OrderedDict()
)
_BLOCK_SCALED_LAYOUT_LOCK = threading.Lock()


def _binding_layout(t):
    return type(t), t.shape, t.stride(), t.dtype, t.device, t.data_ptr() % _ALIGN_BYTES


def _scale_binding_layout(t):
    # Any contiguous shape with enough bytes is accepted; transposed buffers
    # are also accepted, with their exact dimensions and strides checked.
    layout = t.numel() if t.is_contiguous() else (t.shape, t.stride())
    return type(t), layout, t.dtype, t.device, t.data_ptr() % _ALIGN_BYTES


def _prepare_block_scaled_gemm(a, b, sa, sb, alpha, out, dtype, fmt):
    try:
        key = (
            fmt,
            dtype,
            _binding_layout(a),
            _binding_layout(b),
            _scale_binding_layout(sa),
            _scale_binding_layout(sb),
            None
            if alpha is None
            else (type(alpha), alpha.numel(), alpha.dtype, alpha.device),
            None if out is None else _binding_layout(out),
        )
        geometry = _VALIDATED_BLOCK_SCALED_LAYOUTS.get(key)
    except (AttributeError, RuntimeError, TypeError):
        # Leave symbolic shapes and malformed buffers to the full validator.
        key = geometry = None
    if geometry is not None:
        m, n, k = geometry
        if out is None:
            out = torch.empty((m, n), dtype=dtype, device=a.device)
        return a, b, sa, sb, alpha, out, m, n, k, False

    if fmt == "mxfp8":
        x, sfa, sfb, result, m, n, k = prepare_mxfp8_operands(a, b, sa, sb, out, dtype)
        prepared = x, b, sfa, sfb, None, result, m, n, k
    elif fmt == "mxfp4w8a":
        prepared = prepare_mxfp4w8a_operands(a, b, sa, sb, alpha, out, dtype)
    else:
        prepared = prepare_fp4_operands(a, b, sa, sb, alpha, out, dtype, fmt == "nvfp4")
    m, n, k = prepared[6:]
    # The engine consumes only the declared scale extent. Preserve FlashInfer's
    # disjointness contract for any extra bytes in the caller's scale tensors.
    if sa.numel() > sf_bytes(m, k, fmt) or sb.numel() > sf_bytes(n, k, fmt):
        return (*prepared, True)
    unchanged = all(
        x is y for x, y in zip(prepared[:5], (a, b, sa, sb, alpha), strict=True)
    )
    if key is not None and unchanged and (out is None or prepared[5] is out):
        with _BLOCK_SCALED_LAYOUT_LOCK:
            _VALIDATED_BLOCK_SCALED_LAYOUTS[key] = prepared[6:]
            if len(_VALIDATED_BLOCK_SCALED_LAYOUTS) > 512:
                _VALIDATED_BLOCK_SCALED_LAYOUTS.popitem(last=False)
    return (*prepared, False)


def mm_fp4_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    *,
    block_size: int = 16,
    use_nvfp4: bool = True,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
    in_fmt: Optional[str] = None,
) -> torch.Tensor:
    """Run NVFP4 or MXFP4 A @ B with swizzled block scales and optional FP32 alpha."""
    if block_size != (16 if use_nvfp4 else 32):
        raise unsupported(
            CLASS_FP4,
            f"block_size={block_size} with use_nvfp4={use_nvfp4}",
            "block_size 16 for NVFP4 and 32 for MXFP4 (the formats' block scales)",
        )
    out_fresh = out is None
    caller_a = a
    a, b, sfa, sfb, alpha, out, M, N, K, oversized_scales = _prepare_block_scaled_gemm(
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out,
        out_dtype,
        "nvfp4" if use_nvfp4 else "mxfp4",
    )
    if in_fmt is not None or config is not None or chunks is not None:
        in_fmt = fp4_format_for(a.device, use_nvfp4, N, K, in_fmt)
        _check_plan_options(CLASS_FP4, config, chunks, N, K, in_fmt)
    if M == 0:
        return out
    from flashinfer.gemm.batch_invariant.cudnn_gemm import quantized_gemm

    return quantized_gemm(
        CLASS_FP4,
        a,
        b,
        sfa,
        sfb,
        out,
        out_fresh,
        caller_a=caller_a,
        block_size=block_size,
        alpha=alpha,
        workspace=workspace,
        oversized_scales=oversized_scales,
    )


def prepare_mxfp4w8a_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_dtypes: Tuple[torch.dtype, ...] = MM_MXW48_OUT_DTYPES,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    int,
    int,
    int,
]:
    """The mixed classes' typed checks for one product: device and architecture, e4m3 A ``(M, K)`` row-major
    (``mxfp8_quantize``'s output), the packed-e2m1 weight B ``(K / 2, N)`` as the transpose of a row-major ``(N, K / 2)`` tensor
    (``mxfp4_quantize``'s output; uint8 or ``float4_e2m1fn_x2``; the block-scaled instruction takes K-major operands only), K a
    multiple of 32 (one whole UE8M0 block of either operand; the sm_100 class's 128-element runs are checked by the entry once
    the class is known) and N of 8, the two UE8M0 scale buffers in the 128x4 swizzled layout, ``alpha`` a 1-element fp32 tensor
    (None: 1.0), the output's dtype / shape / layout (allocated when absent). Returns ``(a, b, sfa, sfb, alpha, out, M, N, K)``
    with the original packed weight and scales."""
    cls = CLASS_MXW48
    check_device(a.device, cls, MM_MXW48_ARCHS)
    if a.dtype != torch.float8_e4m3fn:
        raise unsupported(
            cls,
            f"A of dtype {a.dtype}",
            "float8_e4m3fn A (the output of mxfp8_quantize)",
        )
    b = _packed_fp4(b, "B", CLASS_MXW48)
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            cls,
            f"{a.dim()}-D A and {b.dim()}-D B",
            "2-D operands (M, K) and (K / 2, N)",
        )
    M, K = a.shape
    Kh, N = b.shape
    if 2 * Kh != K:
        raise unsupported(
            cls,
            f"A (M, {K}) with B ({Kh}, N)",
            "matching K (B holds K / 2 bytes per row)",
        )
    if b.device != a.device:
        raise unsupported(cls, "operands on different devices", "A and B on one device")
    if K == 0 or N == 0:
        raise unsupported(
            cls, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % _SF_BLOCK or N % _ALIGN:
        raise unsupported(
            cls,
            f"K = {K}, N = {N}",
            f"K a multiple of {_SF_BLOCK} (one whole block scale, 16-byte rows of the weight) and N of {_ALIGN} (16-byte rows of the output)",
        )
    a = _aligned_rows(a, M, K, _ALIGN_FP8)
    if _layout_of_b(b, Kh, N, cls) != "k":
        raise unsupported(
            cls,
            "a row-major (K / 2, N) weight",
            "the transpose of a row-major (N, K / 2) weight (the block-scaled instruction takes K-major operands only)",
        )
    _check_base(b, "B", cls)
    sfa = _swizzled_scale(a_descale, M, K, "a_descale", a.device, cls, "mxfp4w8a")
    sfb = _swizzled_scale(b_descale, N, K, "b_descale", a.device, cls, "mxfp4w8a")
    alpha = _alpha_tensor(alpha, a.device, cls)
    if out_dtype not in out_dtypes:
        raise unsupported(cls, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    out = _prepare_out(out, M, N, out_dtype, a.device, cls)
    return a, b, sfa, sfb, alpha, out, M, N, K


def mxfp4w8a_format_for(device: torch.device, in_fmt: Optional[str] = None) -> str:
    family = family_of(compute_capability(device))
    return _checked_format(CLASS_MXW48, in_fmt, mx_class_for("mxfp4w8a", family))


@functools.lru_cache(maxsize=256)
def validate_mixed_k(device: torch.device, K: int, in_fmt: Optional[str] = None) -> str:
    """Return the mixed-operand instruction class supported by this K and device."""
    in_fmt = mxfp4w8a_format_for(device, in_fmt)
    cls = f"gemm.{in_fmt}"
    gran = k_granule(in_fmt)
    if K % gran:
        raise unsupported(
            cls,
            f"K = {K}",
            f"K a multiple of {gran} (the sm_100 / sm_103 instruction reads the e2m1 weight through the TMA's unpacking format "
            "in 128-element runs of the row; the SM107 class takes multiples of 32)",
        )
    return in_fmt


def mm_mxfp8_mxfp4_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    *,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
    in_fmt: Optional[str] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Run MXFP8 activations against packed MXFP4 weights with optional FP32 alpha and PDL."""
    out_fresh = out is None
    caller_a = a
    a, b, sfa, sfb, alpha, out, M, N, K, oversized_scales = _prepare_block_scaled_gemm(
        a, b, a_descale, b_descale, alpha, out, out_dtype, "mxfp4w8a"
    )
    if enable_pdl is None:
        enable_pdl = device_support_pdl(a.device)
    in_fmt = validate_mixed_k(a.device, K, in_fmt)
    cls = f"gemm.{in_fmt}"
    if M == 0:
        return out  # no rows: nothing to launch
    if config is not None or chunks is not None:
        _check_plan_options(cls, config, chunks, N, K, in_fmt)
    from flashinfer.gemm.batch_invariant.cudnn_gemm import quantized_gemm

    return quantized_gemm(
        cls,
        a,
        b,
        sfa,
        sfb,
        out,
        out_fresh,
        caller_a=caller_a,
        block_size=32,
        alpha=alpha,
        enable_pdl=enable_pdl,
        workspace=workspace,
        oversized_scales=oversized_scales,
    )


def _blockscale_scale_view(
    s: torch.Tensor,
    name: str,
    rows: int,
    KB: int,
    major: str,
    device: torch.device,
) -> torch.Tensor:
    """The (rows, KB) view of a block-scale tensor: ``major == "K"`` takes it as ``(rows, KB)``, ``"MN"`` as ``(KB, rows)``
    (:func:`flashinfer.gemm.gemm_fp8_nt_groupwise`'s ``scale_major_mode``); fp32 on the operands' device, any strides (the
    kernel's producer warp stages a block's scales in smem from the tensor's own layout; the ``K`` layout's staging is
    the callers' choice, :func:`gemm_fp8_nt_groupwise_batch_invariant`)."""
    if not isinstance(s, torch.Tensor) or s.dtype != torch.float32:
        raise unsupported(
            CLASS_FP8BS,
            f"{name} of type {type(s).__name__}"
            if not isinstance(s, torch.Tensor)
            else f"{name} {s.dtype}",
            f"a float32 {name}",
        )
    if s.device != device:
        raise unsupported(
            CLASS_FP8BS, f"{name} on another device", f"{name} on A's device"
        )
    want = (rows, KB) if major == "K" else (KB, rows)
    if tuple(s.shape) != want:
        raise unsupported(
            CLASS_FP8BS,
            f"{name} {tuple(s.shape)}",
            f"{name} of shape {want} for scale_major_mode={major!r} (one fp32 scale per 1x{BLOCK_K} block of A, per "
            f"{SCALE_BLOCK_N}x{BLOCK_K} block of B)",
        )
    return s if major == "K" else s.t()


def prepare_fp8_blockscale_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[str],
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int
]:
    """The block-scale class's typed checks: device and architecture, e4m3 A and B, 2-D shapes with matching K, K a whole
    number of 128-element scale blocks, N a multiple of 8, A's row placement, B a row-major (N, K) weight (the NT
    operator: K contiguous, 16-byte rows), the scale tensors of the layout's shapes (handed over with their own strides),
    the output's dtype / shape / layout (allocated when absent). Returns
    ``(a, b, sfa, sfb, out, M, N, K)`` with ``sfa`` (M, K / 128) and ``sfb`` (ceil(N / 128), K / 128) views."""
    cls = CLASS_FP8BS
    check_device(a.device, cls, GEMM_FP8BS_ARCHS)
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise unsupported(
            cls, f"dtypes ({a.dtype}, {b.dtype})", "A and B both float8_e4m3fn"
        )
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            cls, f"{a.dim()}-D A and {b.dim()}-D B", "2-D operands (M, K) and (N, K)"
        )
    M, K = a.shape
    N, K2 = b.shape
    if K != K2:
        raise unsupported(cls, f"A (M, {K}) with B (N, {K2})", "matching K")
    if b.device != a.device:
        raise unsupported(cls, "operands on different devices", "A and B on one device")
    if K == 0 or N == 0:
        raise unsupported(
            cls, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % BLOCK_K:
        raise unsupported(
            cls,
            f"K = {K}",
            f"K a multiple of {BLOCK_K} (a whole number of scale blocks: the class has no K tail)",
        )
    if N % _ALIGN:
        raise unsupported(
            cls, f"N = {N}", f"N a multiple of {_ALIGN} (16-byte output rows)"
        )
    if scale_major_mode not in ("K", "MN"):
        raise unsupported(
            cls,
            f"scale_major_mode={scale_major_mode!r}",
            'scale_major_mode "K" or "MN"',
        )
    a = _aligned_rows(a, M, K, _ALIGN_FP8)
    if b.stride(1) != 1 or (N > 1 and (b.stride(0) < K or b.stride(0) % _ALIGN_FP8)):
        raise unsupported(
            cls,
            f"B with strides {tuple(b.stride())}",
            "a row-major (N, K) weight with K contiguous and 16-byte rows",
        )
    _check_base(b, "B", cls)
    KB, NB = K // BLOCK_K, -(-N // SCALE_BLOCK_N)
    sfa = _blockscale_scale_view(a_scale, "a_scale", M, KB, scale_major_mode, a.device)
    sfb = _blockscale_scale_view(b_scale, "b_scale", NB, KB, scale_major_mode, a.device)
    if out_dtype not in GEMM_FP8BS_OUT_DTYPES:
        raise unsupported(cls, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    out = _prepare_out(out, M, N, out_dtype, a.device, cls)
    return a, b, sfa, sfb, out, M, N, K


def gemm_fp8_nt_groupwise_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[str] = "K",
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    config: Optional[Knobs] = None,
    native_k: Optional[bool] = None,
) -> torch.Tensor:
    """Run the (1, 128, 128) groupwise FP8 product with caller-provided FP32 scale layouts."""
    out_fresh = out is None
    caller_a = a
    a, b, sfa, sfb, out, M, N, K = prepare_fp8_blockscale_operands(
        a, b, a_scale, b_scale, scale_major_mode, out, out_dtype
    )
    if M == 0:
        return out  # no rows: nothing to launch
    if config is not None:
        raise unsupported(
            CLASS_FP8BS,
            "a FlashInfer kernel config",
            "cuDNN batch-invariant plan selection",
        )
    if native_k is not None:
        raise unsupported(
            CLASS_FP8BS, f"native_k={native_k}", "cuDNN selection of the scale staging"
        )
    from flashinfer.gemm.batch_invariant.cudnn_gemm import quantized_gemm

    return quantized_gemm(
        CLASS_FP8BS,
        a,
        b.T,
        sfa,
        sfb.T,
        out,
        out_fresh,
        caller_a=caller_a,
        block_size=128,
    )
