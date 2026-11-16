# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entries of the GEMM classes.

``mm_bf16_batch_invariant`` is what :func:`flashinfer.mm_bf16` runs under ``batch_invariant=True``: C = A @ B (+ bias)
in bf16 with fp32 accumulation; A is (M, K) row-major, B is (K, N) either row-major or the transpose of a row-major
(N, K) tensor (FlashInfer's documented "column-major" weight), the optional bias a bf16 (N,) vector added once in fp32
to the reduced accumulator before the single rounding, C is (M, N) row-major bf16.

``bmm_fp8_batch_invariant`` is what :func:`flashinfer.bmm_fp8` runs under ``batch_invariant=True``: per batch element
C[i] = cvt((A[i] @ B[i]) * mul.rn(A_scale, B_scale)) for fp8 A (b, M, K) row-major and B (b, K, N) column-major (the
transpose of a row-major (b, N, K) weight, FlashInfer's documented layout) or row-major, the two per-tensor fp32
scales multiplied once on the device, bf16 or fp16 output. ``mm_fp8_batch_invariant`` is the same class for one
product (2-D operands; the tests' entry: it forces configurations and chunk counts and exposes the fp32 output).

``mm_mxfp8_batch_invariant`` is what :func:`flashinfer.mm_mxfp8` runs under ``batch_invariant=True``: C = cvt(A @ B)
with the UE8M0 block scales (one per 32 elements, the 128x4 swizzled layout of :func:`flashinfer.mxfp8_quantize`)
applied inside the block-scaled MMA instruction, for e4m3 A (M, K) row-major and B (K, N) the transpose of a row-major
(N, K) weight (the instruction takes K-major operands only), bf16 or fp16 output; the class is the device
architecture's (`schedule.mx_class_for`: the sm_107 instruction folds 64 elements per step, the sm_100 / sm_103 one 32).

``gemm_fp8_nt_groupwise_batch_invariant`` is what :func:`flashinfer.gemm_fp8_nt_groupwise` runs under
``batch_invariant=True``: the DeepSeek-style block-scale class ``gemm.e4m3.kblock128`` -- e4m3 A (M, K) and B (N, K)
row-major, fp32 scales per 1x128 block of A and per 128x128 block of B in the operator's ``K`` or ``MN`` layout, each
128-element K block folded by the 32-wide fp8 step from +0 and the block partials folded in block order with
``fma.rn(mul.rn(s_a, s_b), P, acc)`` by the epilogue warps, bf16 or fp16 output; one launch, no split.
``mm_fp4_batch_invariant`` is what :func:`flashinfer.mm_fp4` runs under ``batch_invariant=True``: C = cvt(mul.rn(A @ B, alpha))
with the block scales (NVFP4: one e4m3 per 16 elements; MXFP4: one UE8M0 per 32; the 128x4 swizzled layout of
:func:`flashinfer.nvfp4_quantize` / :func:`flashinfer.mxfp4_quantize`) applied inside the instruction and the caller's scalar
alpha applied once after the whole fold, for packed-fp4 A (M, K / 2) row-major and B (K / 2, N) the transpose of a row-major
(N, K / 2) weight, bf16 or fp16 output; the class is the device architecture's and, for NVFP4 on SM107, the weight shape's
(`schedule.fp4_class_for`: the 128-wide step for the weight shapes its chunk rule leaves unsplit, the 64-wide step's class
otherwise). Every unsupported input raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls
back."""

from __future__ import annotations

import functools
import logging
from typing import Dict, Optional, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from .generator import Knobs
from .schedule import chunks_for, default_config, fp4_class_for, mx_class_for
from .spec import (
    BLOCK_K,
    ELEM_BITS_OF,
    FP8_BLOCKSCALE_FMTS,
    MMA_K_OF,
    MXFP4_FMTS,
    NVFP4_FMTS,
    SCALE_BLOCK_N,
    chunk_unit,
    is_scaled,
    sf_bytes,
)

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
#: The compute capabilities the classes are stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
MM_BF16_ARCHS = ((10, 0), (10, 3), (10, 7))
BMM_FP8_ARCHS = MM_BF16_ARCHS
MM_MXFP8_ARCHS = MM_BF16_ARCHS
MM_FP4_ARCHS = MM_BF16_ARCHS
_ALIGN = 8  # bf16 / fp16 elements per 16 bytes
_ALIGN_FP8 = 16  # fp8 elements per 16 bytes
_SF_BLOCK = 32  # elements per UE8M0 block scale of the MXFP8 classes: K is a whole number of blocks
_ALIGN_BYTES = 16  # the operand base alignment the kernels assume (TMA)
FP8_FMT_OF_DTYPE = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}
OUT_FMT_OF_DTYPE = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
#: The fp8 classes' output dtypes: ``bmm_fp8``'s bf16 / fp16; fp32 on the one-product entry exposes the chain's input for tests.
BMM_FP8_OUT_DTYPES = (torch.bfloat16, torch.float16)
MM_FP8_OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
#: The MXFP8 class's output dtypes: ``mm_mxfp8``'s bf16 / fp16; fp32 on the direct entry exposes the chain's input for tests.
MM_MXFP8_API_OUT_DTYPES = (torch.bfloat16, torch.float16)
MM_MXFP8_OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
MM_FP4_API_OUT_DTYPES = MM_MXFP8_API_OUT_DTYPES
MM_FP4_OUT_DTYPES = MM_MXFP8_OUT_DTYPES
#: The block-scale fp8 class of :func:`flashinfer.gemm_fp8_nt_groupwise` (``scale_granularity_mnk = (1, 128, 128)``).
CLASS_FP8BS = "gemm.e4m3.kblock128"
GEMM_FP8BS_ARCHS = MM_BF16_ARCHS
GEMM_FP8BS_API_OUT_DTYPES = MM_MXFP8_API_OUT_DTYPES
GEMM_FP8BS_OUT_DTYPES = MM_MXFP8_OUT_DTYPES
_ONES: Dict[
    int, torch.Tensor
] = {}  # a 1-element fp32 1.0 per device: the alpha of an ``mm_fp4`` call without one


def _layout_of_b(b: torch.Tensor, K: int, N: int, cls: str = CLASS) -> str:
    """The layout of a (K, N) weight: ``"n"`` row-major, ``"k"`` the transpose of a row-major (N, K) tensor."""
    if b.stride(1) == 1 and b.stride(0) == N:
        return "n"
    if b.stride(0) == 1 and b.stride(1) == K:
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
    from ...batch_invariant.support import compute_capability
    from .heuristic import family_of

    return _sm_count_cached(index), family_of(
        compute_capability(torch.device("cuda", index))
    )


def prepare_operands(
    a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str, int, int, int]:
    """The bf16 class's typed checks and operand normalisation, shared by the direct entry and the autotuner entry: device
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
    if a.stride(1) != 1 or a.stride(0) % align or a.data_ptr() % _ALIGN_BYTES:
        a = (
            a.as_strided((1, K), (K, 1), a.storage_offset())
            if (M == 1 and a.stride(1) == 1 and a.data_ptr() % _ALIGN_BYTES == 0)
            else a.clone(memory_format=torch.contiguous_format)
        )
    return a


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
    if out.dtype != dtype or tuple(out.shape) != (M, N):
        raise unsupported(
            cls,
            f"out {tuple(out.shape)} {out.dtype}",
            f"a {OUT_FMT_OF_DTYPE.get(dtype, dtype)} (M, N) = ({M}, {N}) output",
        )
    if out.device != device:
        raise unsupported(cls, "out on another device", "the output on A's device")
    _check_base(out, "out", cls, "output")
    if out.stride(1) != 1 or (M > 1 and out.stride(0) != N):
        raise unsupported(
            cls,
            f"out with strides {tuple(out.stride())}",
            "a row-major contiguous output",
        )
    if M == 1 and out.stride(0) != N:
        out = out.as_strided(
            (1, N), (N, 1), out.storage_offset()
        )  # the same elements, an aligned row stride
    return out


def _check_workspace(
    workspace: torch.Tensor, C: int, M: int, N: int, device: torch.device, cls: str
) -> torch.Tensor:
    if (
        workspace.dtype != torch.float32
        or workspace.device != device
        or workspace.numel() < C * M * N
        or not workspace.is_contiguous()
        or workspace.data_ptr() % 16
    ):
        raise unsupported(
            cls,
            "the given workspace",
            f"a contiguous, 16-byte-aligned fp32 tensor on A's device of at least {C * M * N} elements",
        )
    return workspace.reshape(-1)[: C * M * N].view(C, M, N)


def _byte_span(t: torch.Tensor) -> Tuple[int, int]:
    """The half-open byte range from a tensor's first to its last element. The overlap test built on it is conservative:
    a buffer placed in the gaps of a strided tensor counts as overlapping."""
    if t.numel() == 0:
        return 0, 0
    last = sum((n - 1) * s for n, s in zip(t.shape, t.stride(), strict=True) if n > 1)
    lo = t.data_ptr()
    return lo, lo + (last + 1) * t.element_size()


def _overlaps(x: torch.Tensor, y: torch.Tensor) -> bool:
    (a0, a1), (b0, b1) = _byte_span(x), _byte_span(y)
    return a0 < b1 and b0 < a1


def _require_disjoint(cls: str, writes, reads) -> None:
    """The one aliasing policy of every launch path, applied before the first launch: every buffer a launch writes (the
    output; the workspace of the split) is disjoint from every buffer it reads (the operands, the scales, the bias, alpha)
    and from the other written buffers. A read overlapping a write makes the bits depend on which CTA runs first.
    ``writes`` / ``reads`` are ``(tensor, name)`` pairs; byte spans, so a buffer inside a strided tensor's gaps counts as
    overlapping (conservative)."""
    for i, (w, wname) in enumerate(writes):
        for r, rname in reads:
            if _overlaps(w, r):
                raise unsupported(
                    cls,
                    f"{rname} overlapping {wname}",
                    f"{rname} disjoint from {wname}, which the launch writes while it reads {rname} (byte spans: a buffer "
                    "inside a strided tensor's gaps is rejected too)",
                )
        for w2, w2name in writes[i + 1 :]:
            if _overlaps(w, w2):
                raise unsupported(
                    cls,
                    f"{w2name} overlapping {wname}",
                    f"{w2name} disjoint from {wname} (the partials are written to the workspace before the fixup reads them "
                    "and writes the output)",
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
    return s.reshape(1)


def _run(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    out: torch.Tensor,
    b_major: str,
    M: int,
    N: int,
    K: int,
    C: int,
    config: Knobs,
    in_fmt: str,
    out_fmt: str,
    scales: Tuple[torch.Tensor, ...],
    workspace: Optional[torch.Tensor],
    cls: str,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The launch shared by the classes for one (M, K) x (K, N) product: ``b_nk`` is the (N, K) view of the weight (strides
    (1, N) for the ``n`` layout, (K, 1) for ``k``); ``scales`` the fp8 classes' two per-tensor scale tensors (empty for
    bf16); ``bias`` the bf16 class's (N,) bias (the biased chain: added once after the whole fold, in the epilogue of a
    direct store or in the fixup of the workspace split, whose partials stay bias-free)."""
    from .blocks.compile import get_compiled_gemm
    from .blocks.operands import can_implement
    from .fixup import MAX_GRID_Y, get_fixup

    if not can_implement(config, M, N, K, b_major, C, in_fmt):
        raise unsupported(
            cls,
            f"the configuration {config.describe()} at (M, N, K) = ({M}, {N}, {K}) with {C} chunks and a "
            f"{'K' if b_major == 'k' else 'N'}-major weight",
            "a configuration of the enumerated space for the shape (the dispatcher's, or the tuner's candidates)",
        )
    scaled = is_scaled(in_fmt)
    direct = C == 1 or config.cta_chunks
    kernel = get_compiled_gemm(
        config,
        b_major,
        C,
        persistent_clusters(a.device, config),
        in_fmt,
        out_fmt,
        family=part_of(a.device)[1],
        bias=bias is not None and direct,
    )
    extra = (scales if scaled else ()) + ((bias,) if bias is not None else ())
    # the kernel sees 3-D (m, k, l) / (n, k, l) / (m, n, l) tensors: l = the K chunk (1 for the unsplit class and the in-CTA
    # fold's output); every stride handed over is a multiple of the declared 16-byte alignment, the size-1 l stride included
    Kc = K // C
    a3 = a.as_strided((M, Kc, C), (a.stride(0), 1, Kc))
    b3 = b_nk.as_strided(
        (N, Kc, C), (b_nk.stride(0), b_nk.stride(1), Kc * b_nk.stride(1))
    )
    reads = [(a, "A"), (b_nk, "B"), *((sc, "a scale") for sc in scales)]
    if bias is not None:
        reads.append((bias, "the bias"))
    if direct:
        _require_disjoint(cls, [(out, "the output")], reads)
        c3 = out.as_strided((M, N, 1), (out.stride(0), 1, N))
        kernel(a3, b3, c3, *extra)  # the chain in the epilogue
        return out
    # the fixed-chunk split with the partials in an fp32 workspace, folded in chunk order by the fixup kernel (the chain
    # applied there, once)
    if workspace is None:
        ws = torch.empty(C, M, N, dtype=torch.float32, device=a.device)
        _require_disjoint(cls, [(out, "the output")], reads)
    else:
        ws = _check_workspace(workspace, C, M, N, a.device, cls)
        _require_disjoint(cls, [(ws, "the workspace"), (out, "the output")], reads)
    ws = ws.permute(1, 2, 0)  # chunk-major storage (C, M, N) viewed as (M, N, C)
    kernel(a3, b3, ws)  # the unscaled, unbiased body: raw fp32 partials
    rows = min(M, MAX_GRID_Y)
    get_fixup(C, out_fmt, scaled, bias is not None)(ws, out, *extra, M, N, rows)
    return out


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
    """``C = A @ B`` (``+ bias`` when given) by the class ``gemm.bf16``: ``kfold`` when the weight shape's chunk rule
    gives 1, else ``kchunk[C]`` with ``C = chunks_for(N, K)``, a function of the weight shape only, so a row's bits never
    depend on the batch. ``bias`` (bf16, (N,)) selects the biased chain: ``cvt(add(fold, bias[n]))``, the add once in
    fp32 after the whole fold (in the epilogue, or in the fixup for the workspace split). ``config`` forces a
    configuration (a member of the enumerated space: the bits never depend on it); ``chunks`` forces the class's
    chunk count (tests); ``workspace`` (fp32, at least ``chunks * M * N`` elements) is the split class's partials
    buffer, otherwise allocated per call from the caching allocator (stream-ordered)."""
    a, out, b_major, M, N, K = prepare_operands(a, b, out)
    bias = prepare_bias(bias, N, a)
    if M == 0:
        return out  # no rows: nothing to launch
    C = chunks if chunks is not None else chunks_for(N, K)
    if C < 1:
        raise unsupported(CLASS, f"chunks = {C}", "a positive chunk count")
    if C > 1 and K % (MMA_K_OF["bf16"] * C):
        raise unsupported(
            CLASS,
            f"chunks = {C} at K = {K}",
            "each K chunk a whole number of 16-wide MMA steps",
        )
    if config is None:
        config = default_config(M, N, K, C, b_major, *part_of(a.device))
    return _run(
        a,
        b.t(),
        out,
        b_major,
        M,
        N,
        K,
        C,
        config,
        "bf16",
        "bf16",
        (),
        workspace,
        CLASS,
        bias=bias,
    )


# ----------------------------------------------------------------------------- the fp8 classes


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
    """One product of the fp8 class ``gemm.e4m3`` / ``gemm.e5m2`` (A's dtype): ``C = cvt((A @ B) * mul.rn(a_scale, b_scale))``,
    ``kfold.scaled`` when the weight shape's chunk rule gives 1, else ``kchunk[C].scaled`` with ``C = chunks_for(N, K,
    fmt)``, a function of the weight shape and the format only. ``b`` is the (K, N) weight, the transpose of a row-major
    (N, K) tensor (:func:`flashinfer.bmm_fp8`'s column-major layout) or row-major; the scales are 1-element fp32 tensors;
    ``out_dtype`` bf16 / fp16 (fp32 exposes the fp32 chain input for tests). ``config``, ``chunks`` and ``workspace`` as in
    :func:`mm_bf16_batch_invariant`. :func:`bmm_fp8_batch_invariant` is the batched entry behind the FlashInfer API."""
    a, a_scale, b_scale, out, in_fmt, b_major, M, N, K = prepare_fp8_operands(
        a, b, a_scale, b_scale, out, out_dtype
    )
    cls = f"gemm.{in_fmt}"
    if M == 0:
        return out  # no rows: nothing to launch
    C = chunks if chunks is not None else chunks_for(N, K, in_fmt)
    if C < 1:
        raise unsupported(cls, f"chunks = {C}", "a positive chunk count")
    if C > 1 and K % (MMA_K_OF[in_fmt] * C):
        raise unsupported(
            cls,
            f"chunks = {C} at K = {K}",
            "each K chunk a whole number of 32-wide MMA steps",
        )
    if config is None:
        config = default_config(M, N, K, C, b_major, *part_of(a.device), in_fmt)
    return _run(
        a,
        b.t(),
        out,
        b_major,
        M,
        N,
        K,
        C,
        config,
        in_fmt,
        OUT_FMT_OF_DTYPE[out_dtype],
        (a_scale, b_scale),
        workspace,
        cls,
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
    """The batched fp8 classes' typed checks, shared by the direct entry and the autotuner entry: device and architecture,
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
    """``C[i] = cvt((A[i] @ B[i]) * mul.rn(A_scale, B_scale))`` by the class ``gemm.e4m3`` / ``gemm.e5m2`` (A's dtype):
    :func:`flashinfer.bmm_fp8`'s convention, the product of the two per-tensor scales formed once per CTA on the device.
    The chunk rule takes the per-element weight shape, so every batch element is the same class; the unsplit class runs
    as ONE launch whose third operand mode is the batch, the split classes run the batch elements one after another
    (bit-equal by construction: each element is one launch of the class). ``config``, ``chunks`` and ``workspace`` (the
    split class's partials buffer, reused per element) as in :func:`mm_bf16_batch_invariant`."""
    A, A_scale, B_scale, out, in_fmt, b_major, b, M, N, K = prepare_bmm_fp8_operands(
        A, B, A_scale, B_scale, dtype, out
    )
    cls = f"gemm.{in_fmt}"
    if b == 0 or M == 0:
        return out  # nothing to launch
    C = chunks if chunks is not None else chunks_for(N, K, in_fmt)
    if C < 1:
        raise unsupported(cls, f"chunks = {C}", "a positive chunk count")
    if C > 1 and K % (MMA_K_OF[in_fmt] * C):
        raise unsupported(
            cls,
            f"chunks = {C} at K = {K}",
            "each K chunk a whole number of 32-wide MMA steps",
        )
    if config is None:
        config = default_config(M, N, K, C, b_major, *part_of(A.device), in_fmt)
    out_fmt = OUT_FMT_OF_DTYPE[dtype]
    if C == 1:
        from .blocks.compile import get_compiled_gemm
        from .blocks.operands import can_implement

        if not can_implement(config, M, N, K, b_major, 1, in_fmt):
            raise unsupported(
                cls,
                f"the configuration {config.describe()} at (M, N, K) = ({M}, {N}, {K}) with a "
                f"{'K' if b_major == 'k' else 'N'}-major weight",
                "a configuration of the enumerated space for the shape (the dispatcher's, or the tuner's candidates)",
            )
        kernel = get_compiled_gemm(
            config,
            b_major,
            1,
            persistent_clusters(A.device, config),
            in_fmt,
            out_fmt,
            True,
            family=part_of(A.device)[1],
        )
        # (m, k, l) / (n, k, l) / (m, n, l) views with l = the batch element
        kernel(
            A.permute(1, 2, 0),
            B.permute(2, 1, 0),
            out.permute(1, 2, 0),
            A_scale,
            B_scale,
        )
        return out
    if workspace is None:
        # one fp32 workspace for every batch element, reused in stream order (one element's launches complete before the
        # next element's are issued on the same stream), instead of an allocation per element
        workspace = torch.empty(C * M * N, dtype=torch.float32, device=A.device)
    for i in range(b):  # the split classes: one launch of the class per batch element
        _run(
            A[i],
            B[i].t(),
            out[i],
            b_major,
            M,
            N,
            K,
            C,
            config,
            in_fmt,
            out_fmt,
            (A_scale, B_scale),
            workspace,
            cls,
        )
    return out


# ----------------------------------------------------------------------------- the MXFP8 classes


def _flat_sf(
    sf: torch.Tensor,
    rows: int,
    K: int,
    name: str,
    device: torch.device,
    cls: str,
    in_fmt: str = "mxfp8",
) -> torch.Tensor:
    """The scale-factor storage of a ``(rows, K)`` operand as a flat uint8 view (no copy): a contiguous buffer, or the
    transposed view of one, holding at least the bytes of the 128x4 swizzled layout (``spec.sf_bytes``), 16-byte aligned,
    on the operands' device. Only the layout and the extent are checked; the bytes are inputs."""
    if (
        isinstance(sf, torch.Tensor)
        and sf.dtype == torch.float8_e4m3fn
        and in_fmt in NVFP4_FMTS
    ):
        sf = sf.view(
            torch.uint8
        )  # the NVFP4 scales as the quantizer may type them: the same bytes (e4m3)
    if not isinstance(sf, torch.Tensor) or sf.dtype != torch.uint8:
        raise unsupported(
            cls,
            f"{name} of type {type(sf).__name__}"
            + (f" / dtype {sf.dtype}" if isinstance(sf, torch.Tensor) else ""),
            f"{name} a uint8 tensor of UE8M0 scale factors in the 128x4 swizzled layout "
            "(mxfp8_quantize(..., is_sf_swizzled_layout=True))",
        )
    if sf.device != device:
        raise unsupported(cls, f"{name} on another device", f"{name} on A's device")
    if sf.is_contiguous():
        flat = sf.reshape(-1)
    elif sf.dim() == 2 and sf.t().is_contiguous():
        flat = sf.t().reshape(-1)
    else:
        raise unsupported(
            cls,
            f"{name} with strides {tuple(sf.stride())}",
            f"{name} a contiguous scale-factor buffer",
        )
    need = sf_bytes(rows, K, in_fmt)
    if flat.numel() < need:
        raise unsupported(
            cls,
            f"{name} of {flat.numel()} bytes for a ({rows}, {K}) operand",
            f"at least {need} bytes: the 128x4 swizzled layout holds ceil(rows / 128) * 128 rows x 4 * ceil(K / (4 x block)) blocks",
        )
    _check_base(flat, name, cls)
    return flat


def prepare_mxfp8_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out_dtypes: Tuple[torch.dtype, ...] = MM_MXFP8_OUT_DTYPES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """The MXFP8 class's typed checks for one product, shared by the direct entry and the autotuner entry: device and
    architecture, e4m3 A and B, 2-D shapes with matching K, K a multiple of 32 (one whole scale block, hence 16-byte TMA
    rows) and N of 8, A's row placement, B the transpose of a row-major (N, K) weight (the block-scaled instruction takes
    K-major operands only), the two scale-factor buffers, the output's dtype / shape / layout (allocated when absent).
    Returns ``(a, sfa, sfb, out, M, N, K)`` with the scale buffers as flat uint8 views. K is a multiple of 32; where it is not a multiple of the class's step (64 on SM107) the last step is partial: its
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
    sfa = _flat_sf(a_descale, M, K, "a_descale", a.device, cls)
    sfb = _flat_sf(b_descale, N, K, "b_descale", a.device, cls)
    if out_dtype not in out_dtypes:
        raise unsupported(cls, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    out = _prepare_out(out, M, N, out_dtype, a.device, cls)
    return a, sfa, sfb, out, M, N, K


def mx_format_for(device: torch.device, in_fmt: Optional[str] = None) -> str:
    """The MXFP8 class a device runs: the architecture's (`schedule.mx_class_for`), or ``in_fmt`` when given (tests: the
    sm_100 instruction's class also runs on SM107; the sm_107 instruction exists nowhere else)."""
    family = part_of(device)[1]
    if in_fmt is None:
        return mx_class_for("mxfp8", family)
    if in_fmt not in ("mxfp8", "mxfp8_k64") or (
        in_fmt == "mxfp8_k64" and family != "sm_107"
    ):
        raise unsupported(
            CLASS_MXFP8,
            f"the class of in_fmt={in_fmt!r} on {family}",
            "mxfp8 on SM100 / SM103 / SM107, mxfp8_k64 on SM107",
        )
    return in_fmt


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
    """``C = cvt(A @ B)`` with the UE8M0 block scales applied inside the MMA instruction, by the MXFP8 class of the device's
    architecture (``gemm.mxfp8.kfold64`` on SM107, ``gemm.mxfp8.kfold`` on SM100 / SM103; the fingerprint carries the
    architecture): ``kfold`` when the weight shape's chunk rule gives 1, else ``kchunk[C]`` with ``C = chunks_for(N, K,
    fmt)``, a function of the weight shape and the format only (chunks of whole 128-element granules: MMA steps and
    scale-factor atom columns). ``b`` is the (K, N) weight as the transpose of a row-major (N, K) tensor
    (:func:`flashinfer.mm_mxfp8`'s column-major layout); the scales are the 1-D swizzled uint8 buffers of
    :func:`flashinfer.mxfp8_quantize`; ``out_dtype`` bf16 / fp16 (fp32 exposes the fp32 chain input for tests).
    ``config``, ``chunks`` and ``workspace`` as in :func:`mm_bf16_batch_invariant`; ``in_fmt`` forces the class (tests)."""
    a, sfa, sfb, out, M, N, K = prepare_mxfp8_operands(
        a, b, a_descale, b_descale, out, out_dtype
    )
    in_fmt = mx_format_for(a.device, in_fmt)
    cls = f"gemm.{in_fmt}"
    if M == 0:
        return out  # no rows: nothing to launch
    C = chunks if chunks is not None else chunks_for(N, K, in_fmt)
    if C < 1:
        raise unsupported(cls, f"chunks = {C}", "a positive chunk count")
    if C > 1 and K % (chunk_unit(in_fmt) * C):
        raise unsupported(
            cls,
            f"chunks = {C} at K = {K}",
            f"each K chunk a whole number of {chunk_unit(in_fmt)}-element granules (MMA steps and scale-factor atom columns)",
        )
    if config is None:
        config = default_config(M, N, K, C, "k", *part_of(a.device), in_fmt)
    return _run_mx(
        a,
        b.t(),
        sfa,
        sfb,
        out,
        M,
        N,
        K,
        C,
        config,
        in_fmt,
        OUT_FMT_OF_DTYPE[out_dtype],
        workspace,
        cls,
    )


def _run_mx(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    M: int,
    N: int,
    K: int,
    C: int,
    config: Knobs,
    in_fmt: str,
    out_fmt: str,
    workspace: Optional[torch.Tensor],
    cls: str,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The launch of the block-scaled classes for one product: ``b_nk`` the (N, K) row-major view of the weight (in bytes
    for the 4-bit formats: (N, K / 2)), the scale buffers flat, ``alpha`` the 4-bit classes' scalar. Four placements of one
    class: the direct store (the unsplit class; the in-CTA chunk fold and the in-cluster split of the split class), the
    swapped decode record (the weights as the kernel's A, the tokens as its B, the output transposed: the block-scaled step
    is symmetric in its two operands) and the workspace split with the dense module's fixup (the fp32 partials folded in
    chunk order from +0, then the chain)."""
    from .blocks.compile_blockscaled import get_compiled_blockscaled
    from .blocks.operands import can_implement
    from .fixup import MAX_GRID_Y, get_fixup

    if not can_implement(config, M, N, K, "k", C, in_fmt):
        raise unsupported(
            cls,
            f"the configuration {config.describe()} at (M, N, K) = ({M}, {N}, {K}) with {C} chunks",
            "a configuration of the enumerated space for the shape (the dispatcher's, or the tuner's candidates)",
        )
    kernel = get_compiled_blockscaled(
        config,
        C,
        persistent_clusters(a.device, config),
        in_fmt,
        out_fmt,
        family=part_of(a.device)[1],
    )
    Kb = (
        K * ELEM_BITS_OF[in_fmt] // 8
    )  # the operands' K extent in their own units (bytes for the 4-bit formats)
    extra = (alpha,) if alpha is not None else ()
    if config.swap_ab:
        # the decode specialisation: C^T = B^T A^T with the operands exchanged; the kernel's A is the weight (N rows), its B
        # the tokens (any count, one tile), its C the M-major (N, M) view of the output
        _require_disjoint(
            cls,
            [(out, "the output")],
            [(a, "A"), (b_nk, "B"), (sfa, "A's scales"), (sfb, "B's scales")]
            + ([(alpha, "alpha")] if alpha is not None else []),
        )
        kernel(
            b_nk.as_strided((N, Kb, 1), (b_nk.stride(0), 1, Kb)),
            a.as_strided((M, Kb, 1), (a.stride(0), 1, Kb)),
            sfb,
            sfa,
            out.as_strided((N, M, 1), (1, out.stride(0), M * N)),
            *extra,
        )
        return out
    # the kernel sees 3-D (m, k, l) / (n, k, l) / (m, n, l) tensors: l = the K chunk of the split views (1 for the unsplit class,
    # the in-cluster split, which folds whole operands, and the in-CTA fold's output); every stride handed over is a multiple
    # of the declared 16-byte alignment, the size-1 l stride included
    L = 1 if config.cluster_k > 1 else C
    KL = Kb // L
    a3 = a.as_strided((M, KL, L), (a.stride(0), 1, KL))
    b3 = b_nk.as_strided((N, KL, L), (b_nk.stride(0), 1, KL))
    reads = [(a, "A"), (b_nk, "B"), (sfa, "A's scales"), (sfb, "B's scales")]
    if alpha is not None:
        reads.append((alpha, "alpha"))
    if C == 1 or config.cta_chunks or config.cluster_k > 1:
        _require_disjoint(cls, [(out, "the output")], reads)
        c3 = out.as_strided((M, N, 1), (out.stride(0), 1, N))
        kernel(a3, b3, sfa, sfb, c3, *extra)  # the chain in the epilogue
        return out
    # the fixed-chunk split with the partials in an fp32 workspace, folded in chunk order by the fixup kernel (the chain
    # applied there, once)
    if workspace is None:
        ws = torch.empty(C, M, N, dtype=torch.float32, device=a.device)
        _require_disjoint(cls, [(out, "the output")], reads)
    else:
        ws = _check_workspace(workspace, C, M, N, a.device, cls)
        _require_disjoint(cls, [(ws, "the workspace"), (out, "the output")], reads)
    ws = ws.permute(1, 2, 0)  # chunk-major storage (C, M, N) viewed as (M, N, C)
    kernel(a3, b3, sfa, sfb, ws)  # the unscaled body: raw fp32 partials
    rows = min(M, MAX_GRID_Y)
    get_fixup(C, out_fmt, alpha_given=alpha is not None)(ws, out, *extra, M, N, rows)
    return out


# ----------------------------------------------------------------------------- the 4-bit classes


def _packed_fp4(t: torch.Tensor, name: str) -> torch.Tensor:
    """A packed-fp4 operand (two e2m1 values per byte) as its uint8 view, no copy."""
    if t.dtype == torch.uint8:
        return t
    fp4x2 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4x2 is not None and t.dtype == fp4x2:
        return t.view(torch.uint8)
    raise unsupported(
        CLASS_FP4,
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
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
]:
    """The 4-bit classes' typed checks for one product, shared by the direct entry and the autotuner entry: device and
    architecture, packed-fp4 A ``(M, K / 2)`` row-major and B ``(K / 2, N)`` as the transpose of a row-major ``(N, K / 2)``
    weight (the block-scaled instruction takes K-major operands only), K a multiple of 32 (whole block scales of either
    format, 16-byte rows) and N of 8, the two scale-factor buffers (e4m3 per 16 elements for NVFP4, UE8M0 per 32 for
    MXFP4, the 128x4 swizzled layout), ``alpha`` a 1-element fp32 tensor (None: 1.0; a per-token alpha is not a class of
    the mode), the output's dtype / shape / layout (allocated when absent). Returns ``(a, b, sfa, sfb, alpha, out, M, N, K)``
    with the operands as uint8 views and the scale buffers flat."""
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
    sfa = _flat_sf(a_descale, M, K, "a_descale", a.device, cls, base)
    sfb = _flat_sf(b_descale, N, K, "b_descale", a.device, cls, base)
    if alpha is None:
        key = (
            a.device.index
            if a.device.index is not None
            else torch.cuda.current_device()
        )
        if key in _ONES:
            alpha = _ONES[key]
        elif torch.cuda.is_current_stream_capturing():
            # under graph capture the fill runs only on replay: a cached copy made here would read 0 eagerly
            alpha = torch.ones(1, dtype=torch.float32, device=a.device)
        else:
            alpha = _ONES[key] = torch.ones(1, dtype=torch.float32, device=a.device)
    else:
        if not isinstance(alpha, torch.Tensor) or alpha.dtype != torch.float32:
            raise unsupported(
                cls,
                f"alpha of type {type(alpha).__name__}"
                + (
                    f" / dtype {alpha.dtype}" if isinstance(alpha, torch.Tensor) else ""
                ),
                "a float32 tensor alpha",
            )
        if alpha.numel() != 1:
            raise unsupported(
                cls,
                f"an alpha of {alpha.numel()} elements",
                "one scalar alpha (a per-token alpha is not a class of the mode)",
            )
        if alpha.device != a.device:
            raise unsupported(cls, "alpha on another device", "alpha on A's device")
        alpha = alpha.reshape(1)
    if out_dtype not in out_dtypes:
        raise unsupported(cls, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    out = _prepare_out(out, M, N, out_dtype, a.device, cls)
    return a, b, sfa, sfb, alpha, out, M, N, K


def fp4_format_for(
    device: torch.device, use_nvfp4: bool, N: int, K: int, in_fmt: Optional[str] = None
) -> str:
    """The 4-bit class a device runs for a weight shape: NVFP4 by `schedule.fp4_class_for` (the sm_107 instruction's class on
    SM107 for the unsplit weight shapes), MXFP4 by `schedule.mx_class_for` (the architecture's); or ``in_fmt`` when given
    (tests: the sm_100 instructions' classes also run on SM107; the sm_107 instructions' classes exist nowhere else)."""
    family = part_of(device)[1]
    if in_fmt is None:
        return (
            fp4_class_for(N, K, family) if use_nvfp4 else mx_class_for("mxfp4", family)
        )
    allowed = NVFP4_FMTS if use_nvfp4 else MXFP4_FMTS
    if in_fmt not in allowed or (in_fmt.endswith("_k128") and family != "sm_107"):
        raise unsupported(
            CLASS_FP4,
            f"the class of in_fmt={in_fmt!r} on {family} with use_nvfp4={use_nvfp4}",
            f"{' / '.join(allowed[:1])} on SM100 / SM103 / SM107, {allowed[1]} on SM107",
        )
    return in_fmt


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
    """``C = cvt(mul.rn(A @ B, alpha))`` with the block scales applied inside the MMA instruction, by the 4-bit class of the
    device and the weight shape (NVFP4: ``gemm.nvfp4.kfold`` / ``kchunk[C]``, and on SM107 ``gemm.nvfp4.kfold128`` for the
    unsplit weight shapes; MXFP4: ``gemm.mxfp4.kfold`` / ``kchunk[C]``, on SM107 ``gemm.mxfp4.kfold128`` / ``kchunk128[C]``):
    ``C = chunks_for(N, K, fmt)`` is a function of the weight shape and the format only. :func:`flashinfer.mm_fp4`'s
    conventions: packed-fp4 ``a`` ``(M, K / 2)`` and ``b`` ``(K / 2, N)`` as the transpose of a row-major ``(N, K / 2)``
    weight, the 1-D swizzled scale buffers of :func:`flashinfer.nvfp4_quantize` / :func:`flashinfer.mxfp4_quantize`,
    ``alpha`` the caller's 1-element fp32 tensor (one ``mul.rn``, once, after the whole fold), ``block_size`` 16 (NVFP4) /
    32 (MXFP4), ``out_dtype`` bf16 / fp16 (fp32 exposes the fp32 chain input for tests). ``config``, ``chunks`` and
    ``workspace`` as in :func:`mm_bf16_batch_invariant`; ``in_fmt`` forces the class (tests). K is a multiple of 32; where
    it is not a multiple of the class's step (64, or 128 on SM107) the last step is partial: its padded values are TMA zero
    fill, its scales are read from the scale buffers' padding, so those bytes must be finite (the quantizers write zeros;
    a NaN or Inf there poisons the row)."""
    if block_size != (16 if use_nvfp4 else 32):
        raise unsupported(
            CLASS_FP4,
            f"block_size={block_size} with use_nvfp4={use_nvfp4}",
            "block_size 16 for NVFP4 and 32 for MXFP4 (the formats' block scales)",
        )
    a, b, sfa, sfb, alpha, out, M, N, K = prepare_fp4_operands(
        a, b, a_descale, b_descale, alpha, out, out_dtype, use_nvfp4
    )
    in_fmt = fp4_format_for(a.device, use_nvfp4, N, K, in_fmt)
    cls = f"gemm.{in_fmt}"
    if M == 0:
        return out  # no rows: nothing to launch
    C = chunks if chunks is not None else chunks_for(N, K, in_fmt)
    if C < 1:
        raise unsupported(cls, f"chunks = {C}", "a positive chunk count")
    if C > 1 and K % (chunk_unit(in_fmt) * C):
        raise unsupported(
            cls,
            f"chunks = {C} at K = {K}",
            f"each K chunk a whole number of {chunk_unit(in_fmt)}-element granules (MMA steps and scale-factor atom columns)",
        )
    if config is None:
        config = default_config(M, N, K, C, "k", *part_of(a.device), in_fmt)
    return _run_mx(
        a,
        b.t(),
        sfa,
        sfb,
        out,
        M,
        N,
        K,
        C,
        config,
        in_fmt,
        OUT_FMT_OF_DTYPE[out_dtype],
        workspace,
        cls,
        alpha=alpha,
    )


# ----------------------------------------------------------------------------- the block-scale fp8 class


#: The K-native scale staging: at most half the CTA tile's rows live (an empirical crossover).
NATIVE_K_MAX_ROWS = 64


def native_k_staging(M: int) -> bool:
    """Whether the block-scale class stages ``K``-layout scales natively -- four blocks per ring stage, a row's four
    consecutive scales one 16-byte ``cp.async`` from the caller's tensor as it is -- or through the MN-major copy and the
    per-block ring. The gather costs one 32-byte sector per live row of the CTA tile and stage (the rows clamped to the
    last one collapse onto its sector), the MN-major rows one sector per eight rows: with at most half the tile's rows
    live the gather hides under the stage's operand TMA and the staging saves the copy's launch; with every row live it
    stands on the chain's critical path. A placement: the bits never depend on it (``blocks/blockscale_fp8.py``)."""
    return M <= NATIVE_K_MAX_ROWS


def _blockscale_scale_view(
    s: torch.Tensor,
    name: str,
    rows: int,
    KB: int,
    major: str,
    device: torch.device,
) -> torch.Tensor:
    """The (rows, KB) view of a block-scale tensor: ``major == "K"`` takes it as ``(rows, KB)``, ``"MN"`` as ``(KB, rows)``
    (:func:`flashinfer.gemm_fp8_nt_groupwise`'s ``scale_major_mode``); fp32 on the operands' device, any strides (the
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
    """The block-scale class ``gemm.e4m3.kblock128`` of :func:`flashinfer.gemm_fp8_nt_groupwise` (``scale_granularity_mnk =
    (1, 128, 128)``): per element ``out = cvt(fold_kb fma.rn(mul.rn(s_a[m, kb], s_b[n // 128, kb]), P_kb, acc))`` with
    ``P_kb`` the 32-wide fp8 fold of K block ``kb`` from +0 and the block fold from +0 in block order. ``a`` (M, K) and
    ``b`` (N, K) e4m3 row-major; ``a_scale`` (M, K / 128) and ``b_scale`` (ceil(N / 128), K / 128) fp32 in the ``K``
    layout, or their transposes in the ``MN`` layout; ``out_dtype`` bf16 / fp16 (fp32 exposes the fold for tests);
    ``config`` forces a record of the class's space (the dispatcher's pick by default: every record computes the same
    bits), ``native_k`` the ``K`` layout's scale staging (:func:`native_k_staging` by default; a placement). One launch,
    no split, no workspace."""
    a, b, sfa, sfb, out, M, N, K = prepare_fp8_blockscale_operands(
        a, b, a_scale, b_scale, scale_major_mode, out, out_dtype
    )
    cls = CLASS_FP8BS
    if M == 0:
        return out  # no rows: nothing to launch
    from .blocks.compile_fp8bs import get_compiled_fp8bs
    from .blocks.operands import can_implement

    in_fmt = FP8_BLOCKSCALE_FMTS[0]
    if config is None:
        config = default_config(M, N, K, 1, "k", *part_of(a.device), in_fmt)
    if not can_implement(config, M, N, K, "k", 1, in_fmt):
        raise unsupported(
            cls,
            f"the configuration {config.describe()} at (M, N, K) = ({M}, {N}, {K})",
            "a configuration of the enumerated space for the shape (the dispatcher's, or the tuner's candidates)",
        )
    # the K layout's staging (a placement: the fold never changes): the K-native ring at a few rows, the tensors read as they
    # are from 16-byte-aligned bases (the four-block copies); the MN-major copy and the per-block ring otherwise
    if native_k is None:
        native_k = native_k_staging(M)
    native_k = (
        scale_major_mode == "K"
        and native_k
        and sfa.data_ptr() % 16 == 0
        and sfb.data_ptr() % 16 == 0
    )
    if scale_major_mode == "K" and not native_k:
        sfa = sfa.t().contiguous().t()
    _require_disjoint(
        cls,
        [(out, "the output")],
        [(a, "A"), (b, "B"), (sfa, "a_scale"), (sfb, "b_scale")],
    )
    # the kernel sees (m, k, l) / (n, k, l) / (m, n, l) operands with l = 1 and the scale views with their own strides
    a3 = a.as_strided((M, K, 1), (a.stride(0), 1, K))
    b3 = b.as_strided((N, K, 1), (b.stride(0), 1, K))
    c3 = out.as_strided((M, N, 1), (out.stride(0), 1, N))
    # the compile's architecture query and the launch follow A's device, not the current one
    with torch.cuda.device(a.device):
        kernel = get_compiled_fp8bs(
            config,
            persistent_clusters(a.device, config),
            OUT_FMT_OF_DTYPE[out_dtype],
            family=part_of(a.device)[1],
            native_k=native_k,
        )
        kernel(a3, b3, sfa, sfb, c3)
    return out
