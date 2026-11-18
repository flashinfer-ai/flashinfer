# SPDX-License-Identifier: Apache-2.0
"""DSA sparse-MLA decode with precomputed top-k indices on SM90.

Attends only the tokens named by a per-query top-k index list over an
``fp8_ds_mla`` 656-byte/token Page64 paged KV cache (512 fp8_e4m3 NoPE +
4 fp32 per-128-tile scales + 64 bf16 RoPE per token). This is the
DeepSeek-V3.2 / GLM-class sparse-attention decode contract; the indexer /
top-k selection stage is upstream and its output is an input here. SM120's
packed FP8/NVFP4 sparse-MLA family is a separate API.
"""

import functools
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import tvm_ffi

from ..api_logging import flashinfer_api
from ..jit.core import logger
from ..jit.mla import gen_sparse_mla_sm90_module
from ..utils import register_custom_op, register_fake_op

# Kill switch shared by FlashInfer's specialized kernels (read at call time):
# forces the composable torch "reference" backend.
_SPECIALIZED_KERNEL_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
# Deprecated alias (kept for one release): the per-kernel switch this op
# first shipped with.
_KILL_SWITCH_ENV = "FLASHINFER_DISABLE_SPARSE_MLA_SM90"

# Backends behind the one public API, selected per call (#5667 pattern):
#   "specialized": the agent-generated SM90 kernel
#                  (csrc/sparse_mla_sm90.cu).
#   "reference":   the composable torch path (gather + dequantise the
#                  fp8_ds_mla rows + fp32 attend) — exact-math, slow; it
#                  serves "auto" wherever the specialized kernel does not
#                  route (kill switch included).
#   "auto":        "specialized" on the dispatch allowlist
#                  (sparse_mla_sm90_dispatch.json) on a supported device,
#                  "reference" otherwise. An explicitly selected backend is
#                  never silently swapped and never falls back: "reference"
#                  never runs the fused kernel, explicit "specialized" that
#                  cannot serve a call raises with the refusal reason, and
#                  an unknown backend name raises.
BACKENDS = ("auto", "specialized", "reference")
_DISPATCH_FILE = "sparse_mla_sm90_dispatch.json"

# Qualified contract constants (the kernel's own guard relations): 8 q-heads
# per rank, 512+64 absorbed head dim, Page64 fp8_ds_mla 656-byte rows,
# top-k 2048. The kernel is compiled against these; other values are a
# different kernel, not a parameter.
_NUM_HEADS = 8
_CKV_DIM = 512
_KPE_DIM = 64
_QK_DIM = _CKV_DIM + _KPE_DIM
_PAGE_SIZE = 64
_BYTES_PER_TOKEN = 656
_TOPK = 2048
# Validated token-row envelope: the H200 qualification grid covered
# num_seqs x q_len products from 6 to 60 (B1-B10, q_len 6) plus
# single-row sanity shapes; the split heuristic is defined for any
# nq >= 1 but qualification stops at 64 rows.
_MAX_NQ = 64


def _disabled() -> bool:
    if os.environ.get(_SPECIALIZED_KERNEL_DISABLE_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    ):
        return True
    if os.environ.get(_KILL_SWITCH_ENV, "0") not in ("", "0", "false", "False"):
        logger.warning_once(
            "%s is deprecated and will be removed in a future release; set "
            "%s=1 instead (the kill switch shared by FlashInfer's "
            "specialized kernels)",
            _KILL_SWITCH_ENV,
            _SPECIALIZED_KERNEL_DISABLE_ENV,
        )
        return True
    return False


# Per-backend dispatch census and fallback reasons (#5667-style introspection).
_stats: Dict[str, Any] = {
    "calls": 0,
    "backend_dispatches": {"specialized": 0, "reference": 0},
    "fallback_reasons": {},
}


def _fallback(reason: str) -> None:
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _sparse_mla_sm90_stats() -> dict:
    """Diagnostics: per-backend dispatch counts, fallback reasons, backends."""
    return {
        "calls": _stats["calls"],
        "backend_dispatches": dict(_stats["backend_dispatches"]),
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "backends": BACKENDS,
        "dispatch_table_loaded": _dispatch_table() is not None,
    }


def _reset_sparse_mla_sm90_stats() -> None:
    _stats["calls"] = 0
    for k in _stats["backend_dispatches"]:
        _stats["backend_dispatches"][k] = 0
    _stats["fallback_reasons"].clear()


@functools.cache
def _dispatch_table() -> Optional[dict]:
    """Load the dispatch surface (package data); ``None`` means
    ``backend="auto"`` never routes to "specialized" (the compiled-in guards
    below stay authoritative for eligibility)."""
    try:
        from importlib import resources

        payload = json.loads(
            resources.files("flashinfer").joinpath(_DISPATCH_FILE).read_text()
        )
        if not isinstance(payload, dict):
            raise TypeError(f"expected a dict, got {type(payload).__name__}")
        return payload
    except Exception as exc:  # noqa: BLE001 - auto-routing is best-effort
        logger.warning_once(
            "sparse_mla_decode_sm90: dispatch allowlist %s unavailable "
            "(%s: %s); backend='auto' will use the composable reference path",
            _DISPATCH_FILE,
            type(exc).__name__,
            exc,
        )
        return None


def _auto_routes(query: torch.Tensor, nq: int) -> Tuple[bool, Optional[str]]:
    """Shape-keyed auto-routing decision for ``backend="auto"``."""
    table = _dispatch_table()
    if table is None:
        return False, "dispatch allowlist unavailable"
    try:
        op = table["ops"]["sparse_mla_decode"]
        routed = tuple(query.shape[-2:]) == (
            int(op["num_heads"]),
            int(op["qk_dim"]),
        ) and 1 <= nq <= int(op["envelope"]["max_token_rows"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"dispatch allowlist invalid ({type(exc).__name__}: {exc})"
    if not routed:
        return False, "shape off the auto-routing allowlist"
    if query.device.type != "cuda" or not _device_supported(query.device):
        return False, "device unsupported (needs SM90 + 93568B smem opt-in)"
    return True, None


def _backend_order(requested: str, auto_routed: bool) -> Tuple[str, ...]:
    """#5667-style backend order for one call. A caller who named a backend
    is never silently swapped and never falls back: explicit "reference"
    never sees the fused kernel, explicit "specialized" runs the fused
    kernel or raises. Only "auto" routes around a refusal."""
    if requested == "specialized":
        return ("specialized",)
    if requested == "reference":
        return ("reference",)
    if requested == "auto":
        return ("specialized", "reference") if auto_routed else ("reference",)
    return ()


def _composable_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """The composable torch path: gather the indexed fp8_ds_mla rows,
    dequantise (per-128-tile fp32 scales on the 512 NoPE dims, bf16 RoPE
    tail), attend in fp32, write bf16. Exact math, no fused kernel — the
    kill-switch / fallback semantics of this op, not a performance path."""
    B, QL, H, D = query.shape
    KPE = _KPE_DIM
    CKV = D - KPE
    TILE = 128
    ntiles = CKV // TILE
    num_pages, PAGE, BPT = kv_cache.shape
    flat = kv_cache.reshape(num_pages * PAGE, BPT)
    qf = query.to(torch.float32)
    acc = torch.zeros((B, QL, H, CKV), dtype=torch.float32, device=query.device)
    for b in range(B):
        for j in range(QL):
            # Capture-safe: no data-dependent host control flow (boolean
            # indexing lowers to nonzero(), which synchronizes and breaks
            # CUDA-graph capture of the kill-switch / fallback path).
            # Invalid (-1) slots gather row 0 and are masked to -inf in the
            # logits; a fully -1 row softmaxes to NaN, zeroed below.
            idx = indices[b, j].to(torch.int64)
            valid = idx >= 0
            rows = flat[idx.clamp_min(0)]
            nope = (
                rows[:, :CKV]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .to(torch.float32)
                .reshape(-1, ntiles, TILE)
            )
            scales = (
                rows[:, CKV : CKV + 4 * ntiles]
                .contiguous()
                .view(torch.float32)
                .reshape(-1, ntiles, 1)
            )
            ckv = (nope * scales).reshape(-1, CKV)
            kpe = (
                rows[:, CKV + 4 * ntiles :]
                .contiguous()
                .view(torch.bfloat16)
                .to(torch.float32)
            )
            qn = qf[b, j, :, :CKV]
            qp = qf[b, j, :, CKV:]
            sc = (qn @ ckv.T + qp @ kpe.T) * float(sm_scale)
            sc = sc.masked_fill(~valid, float("-inf"))
            pr = torch.nan_to_num(torch.softmax(sc, dim=-1))
            acc[b, j] = pr @ ckv
    out.copy_(acc.to(torch.bfloat16))
    return out


def _split_plan(nq: int):
    """Mirror of the kernel's launch heuristic (pure function of nq)."""
    tile = 48 if nq <= 12 else 64
    maxt = (_TOPK + tile - 1) // tile
    per = max(1, min(264 // nq if nq <= 264 else 1, maxt))
    tps = max(1, (maxt + per - 1) // per)
    nsplits = (maxt + tps - 1) // tps
    return tile, tps, nsplits


def _exact_workspace_bytes(nq: int) -> int:
    """The kernel's exact workspace need for one call of ``nq`` token rows
    (what the C++ binding enforces per call). Not monotonic in ``nq``: the
    split plan changes tile size and split count with the row count, so e.g.
    12 rows need more bytes than 64."""
    _, _, nsplits = _split_plan(nq)
    if nsplits <= 1:
        return 0
    return nq * nsplits * (_NUM_HEADS * _CKV_DIM) * 2 + nq * nsplits * _NUM_HEADS * 4


@flashinfer_api
def get_sparse_mla_sm90_workspace_size_bytes(
    batch_size: int, q_len_per_req: int
) -> int:
    """Return the byte workspace capacity for :func:`sparse_mla_decode_sm90`.

    Parameters
    ----------
    batch_size : int
        Decode batch (number of sequences).
    q_len_per_req : int
        Per-sequence query window (e.g. speculative-decode window).

    Returns
    -------
    int
        Bytes of FP16 partial output plus FP32 partial LSE covering the
        split-K plans the kernel chooses for every token-row count up to
        ``batch_size * q_len_per_req``. The per-call need is a pure function
        of the row count but is *not* monotonic in it (the plan's tile size
        and split count change with the row count), so this returns the
        running maximum over all row counts up to the requested one: a
        buffer sized at the largest row count a caller will use is valid
        for every smaller call — size it once, outside CUDA-graph capture.
    """
    nq = batch_size * q_len_per_req
    if nq < 1 or nq > _MAX_NQ:
        raise NotImplementedError(
            f"sparse_mla_decode_sm90 qualifies 1 <= batch*q_len <= {_MAX_NQ}, got {nq}"
        )
    return max(_exact_workspace_bytes(k) for k in range(1, nq + 1))


@functools.cache
def _get_module():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Warm up sparse_mla_decode_sm90 eagerly before CUDA Graph capture"
        )
    return gen_sparse_mla_sm90_module().build_and_load()


@functools.cache
def _get_run_fn():
    return _get_module().run


@functools.cache
def _ffi_device(index: int):
    return tvm_ffi.device("cuda", index)


@functools.cache
def _device_supported(device: torch.device) -> bool:
    # HW eligibility first (no module build), then the module's own probe.
    # A failed JIT build (e.g. FLASHINFER_DISABLE_JIT=1 with no cached
    # module) reports "unsupported": "auto" then routes to the composable
    # reference path and explicit "specialized" raises with the reason —
    # except under CUDA-graph capture, where the eager-warmup discipline
    # stays loud.
    if torch.cuda.get_device_capability(device) != (9, 0):
        return False
    try:
        module = _get_module()
    except Exception as exc:
        if torch.cuda.is_current_stream_capturing():
            raise
        logger.warning_once(
            "sparse_mla_decode_sm90: JIT module unavailable (%s: %s); the "
            "device probe reports unsupported and the composable reference "
            "path serves this process",
            type(exc).__name__,
            str(exc)[:300],
        )
        return False
    return bool(module.device_supported(device.index))


@flashinfer_api
def is_sparse_mla_sm90_supported(device: torch.device) -> bool:
    """Whether ``device`` qualifies for :func:`sparse_mla_decode_sm90`.

    True only on compute capability 9.0 devices with at least 93568 bytes of
    opt-in dynamic shared memory per block, and only while the
    ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE`` kill switch (and its deprecated
    alias ``FLASHINFER_DISABLE_SPARSE_MLA_SM90``) is unset. Calling this
    builds the JIT module, so call it (or the op) eagerly before any CUDA
    Graph capture.
    """
    if _disabled():
        return False
    return _device_supported(torch.device(device))


def _validate(
    query, kv_cache, indices, seq_lens, workspace, out, sm_scale, backend, fused=True
):
    if backend not in BACKENDS:
        raise NotImplementedError(
            f"sparse_mla_decode_sm90 supports backend in {BACKENDS}; an "
            "explicitly selected backend is never silently swapped"
        )
    tensors = (query, kv_cache, indices, seq_lens, workspace, out)
    if any(not t.is_cuda or t.device != query.device for t in tensors):
        raise ValueError("All tensors must be on the same CUDA device")
    if (
        query.ndim != 4
        or query.shape[-2:] != (_NUM_HEADS, _QK_DIM)
        or not query.is_contiguous()
    ):
        raise NotImplementedError(
            f"query must be contiguous [B, q_len, {_NUM_HEADS}, {_QK_DIM}] "
            "(8 local q-heads, absorbed 512+64 head dim); other head counts "
            "or head dims are outside the qualified contract"
        )
    b, sq = query.shape[:2]
    nq = b * sq
    if fused and (nq < 1 or nq > _MAX_NQ):
        raise NotImplementedError(
            f"qualified token-row envelope is 1 <= batch*q_len <= {_MAX_NQ}, got {nq}"
        )
    if query.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise NotImplementedError("query and out must be BF16")
    if (
        kv_cache.ndim != 3
        or kv_cache.shape[1:] != (_PAGE_SIZE, _BYTES_PER_TOKEN)
        or kv_cache.dtype != torch.uint8
        or not kv_cache.is_contiguous()
        or kv_cache.shape[0] < 1
    ):
        raise NotImplementedError(
            f"kv_cache must be contiguous uint8 [num_pages, {_PAGE_SIZE}, "
            f"{_BYTES_PER_TOKEN}] (fp8_ds_mla rows); other cache formats are "
            "outside the qualified contract"
        )
    if (
        indices.shape != (b, sq, _TOPK)
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
    ):
        raise NotImplementedError(
            f"indices must be contiguous int32 [B, q_len, {_TOPK}]; "
            "only top-k 2048 is qualified"
        )
    if (
        seq_lens.shape != (b,)
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous CUDA int32 [B]")
    required = _exact_workspace_bytes(nq) if fused else 0
    if fused and (
        workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.numel() < required
    ):
        raise ValueError(
            f"workspace must be a contiguous uint8 vector with at least "
            f"{required} bytes"
        )
    if out.shape != (b, sq, _NUM_HEADS, _CKV_DIM) or not out.is_contiguous():
        raise ValueError(f"out must be contiguous [B, q_len, {_NUM_HEADS}, {_CKV_DIM}]")
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("sm_scale must be positive finite")
    if fused and any(t.data_ptr() % 16 for t in (query, kv_cache, workspace, out)):
        raise ValueError("query/kv_cache/workspace/out require 16-byte alignment")

    # Writable buffers must not alias any other tensor.
    def extent(t):
        n = max(t.numel(), 1)
        return t.data_ptr(), t.data_ptr() + n * t.element_size()

    for target in (workspace, out):
        lo, hi = extent(target)
        for source in tensors:
            if source is target:
                continue
            other_lo, other_hi = extent(source)
            if max(lo, other_lo) < min(hi, other_hi):
                raise ValueError(
                    "Writable sparse-MLA buffers must not overlap any other tensor"
                )
    if fused and not _device_supported(query.device):
        raise NotImplementedError(
            "sparse_mla_decode_sm90 requires an SM90 device with at least "
            "93568 bytes of opt-in shared memory"
        )


# Structural Python-side validation runs once per argument signature
# (shapes/dtypes/device/backend, below); the C++ binding independently
# re-checks every structural requirement (dtype, dims, contiguity,
# alignment, workspace capacity, device support) on every call, so that
# cache only affects which layer raises, never what is enforced. Properties
# the signature cannot establish — storage pointers and scalar values —
# are re-validated on every call by _validate_per_call. This keeps the
# eager per-call host path short enough not to starve the GPU at
# decode-size grids (the kernel runs ~12 us at B1).
_validated_signatures: set = set()


def _validate_per_call(
    query, kv_cache, indices, seq_lens, workspace, out, sm_scale, fused
):
    """Per-call checks of properties the signature cache cannot establish:
    device placement, contiguity, alignment, writable-buffer overlap, and
    ``sm_scale`` — all functions of storage pointers or scalar values, which
    can change between calls that share a signature."""
    tensors = (query, kv_cache, indices, seq_lens, workspace, out)
    if any(not t.is_cuda or t.device != query.device for t in tensors):
        raise ValueError("All tensors must be on the same CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise NotImplementedError(
            "query/kv_cache/indices/seq_lens/workspace/out must be contiguous"
        )
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("sm_scale must be positive finite")
    if fused and any(t.data_ptr() % 16 for t in (query, kv_cache, workspace, out)):
        raise ValueError("query/kv_cache/workspace/out require 16-byte alignment")

    # Writable buffers must not alias any other tensor.
    def extent(t):
        n = max(t.numel(), 1)
        return t.data_ptr(), t.data_ptr() + n * t.element_size()

    for target in (workspace, out):
        lo, hi = extent(target)
        for source in tensors:
            if source is target:
                continue
            other_lo, other_hi = extent(source)
            if max(lo, other_lo) < min(hi, other_hi):
                raise ValueError(
                    "Writable sparse-MLA buffers must not overlap any other tensor"
                )


@register_custom_op(
    "flashinfer::sparse_mla_decode_sm90", mutates_args=("workspace", "out")
)
def _run(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
) -> None:
    # Re-query the caller's current stream on every call (CUDA-graph
    # requirement: never launch on a cached stream). Equivalent to
    # tvm_ffi.use_torch_stream(), with the FFI device object cached: this op
    # runs ~12 us at decode sizes, so per-call host work is kept minimal.
    stream = torch.cuda.current_stream(query.device)
    with tvm_ffi.use_raw_stream(_ffi_device(query.device.index), stream.cuda_stream):
        _get_run_fn()(query, kv_cache, indices, seq_lens, workspace, out, sm_scale)


@register_fake_op("flashinfer::sparse_mla_decode_sm90")
def _run_fake(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
) -> None:
    pass


@flashinfer_api
def sparse_mla_decode_sm90(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    out: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Sparse absorbed-MLA decode over precomputed top-k indices (SM90).

    Attends, per query token, exactly the KV-cache tokens named by its
    ``indices`` row (DSA: the indexer / top-k stage runs upstream). The
    kernel is non-causal; causality must already live in the index lists.

    Parameters
    ----------
    query : torch.Tensor
        BF16 contiguous ``[B, q_len, 8, 576]``: per head, 512 absorbed NoPE
        (latent) features followed by 64 RoPE features. 8 is the per-rank
        local head count (e.g. 64 heads at TP8); it is a compile-time kernel
        constant, not a parameter.
    kv_cache : torch.Tensor
        uint8 contiguous ``[num_pages, 64, 656]`` ``fp8_ds_mla`` paged cache.
        Each 656-byte token row is 512 fp8_e4m3 NoPE bytes, then 4 fp32
        per-128-element-tile scales, then 64 bf16 RoPE values.
    indices : torch.Tensor
        int32 contiguous ``[B, q_len, 2048]`` physical token slots into the
        flattened ``num_pages * 64`` token cache (page-table-translated
        storage offsets). ``-1`` marks an invalid slot and is accepted at any
        position; a fully ``-1`` row produces zero output. In-range validity
        is the caller's contract and is not checked on device: any
        non-negative entry must be below ``num_pages * 64``.
    seq_lens : torch.Tensor
        int32 contiguous ``[B]`` scheduling metadata (the engine's
        ``cache_seqlens`` clipped to top-k). Semantics live entirely in the
        index lists; this tensor is validated but not read by the kernel.
    workspace_buffer : torch.Tensor
        Caller-owned CUDA uint8 vector sized by
        :func:`get_sparse_mla_sm90_workspace_size_bytes` for the largest
        ``B * q_len`` the caller will use. Holds FP16 split partials and FP32
        partial LSE; do not share writable workspace across concurrent
        streams. Allocate it outside CUDA-graph capture.
    sm_scale : float
        Softmax scale, supplied by the model (DeepSeek-V3.2 / GLM-class
        checkpoints use ``qk_head_dim ** -0.5`` with the *checkpoint* head
        dim, e.g. ``256 ** -0.5`` — not the absorbed 576).
    out : torch.Tensor, optional
        BF16 contiguous ``[B, q_len, 8, 512]`` latent-space output (pre
        up-projection). Allocated when omitted; pass a persistent tensor and
        warm up eagerly before CUDA-graph capture.
    backend : str
        One of ``BACKENDS = ("auto", "specialized", "reference")``.
        ``"specialized"`` selects the SM90 fused kernel explicitly; a call it
        cannot serve (unsupported device, token-row envelope, kill switch,
        JIT failure) raises ``NotImplementedError`` with the refusal reason
        (recorded in the dispatch stats) — an explicitly selected backend
        never falls back. ``"reference"`` selects the composable torch path
        explicitly (exact math, slow) and never runs the fused kernel.
        ``"auto"`` routes to ``"specialized"`` only on the dispatch allowlist
        (``flashinfer/sparse_mla_sm90_dispatch.json``) on a supported
        device, and to ``"reference"`` otherwise — only ``"auto"`` routes
        around a refusal. Any other name raises: an explicitly selected
        backend is never silently swapped.

    Returns
    -------
    torch.Tensor
        The supplied or newly allocated output. Split reductions keep FP32
        LSE and FP16 partial values; the final write is BF16.

    Notes
    -----
    **CUDA graphs.** All launches are plain stream launches onto the current
    stream, re-queried on every call; the op performs no allocation, host
    synchronization, or autotuning at call time, so a warmed-up call is
    capturable. Required discipline: (1) call once eagerly first, on each
    device you will capture on — JIT module build and the once-per-device
    kernel shared-memory attribute setup must not happen under capture
    (module build raises if attempted while capturing); (2) the split plan
    is a pure function of ``B * q_len``, so a
    graph replays correctly for the captured shape; capture each shape
    separately; (3) keep ``query``/``kv_cache``/``indices``/``seq_lens``/
    ``workspace_buffer``/``out`` at stable addresses across replays (update
    cache contents and index values in place).

    **Kill switch.** Set ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (the
    switch shared by FlashInfer's specialized kernels, read at call time) to
    disable the fused kernel: ``"auto"`` then routes to the composable
    ``"reference"`` path and explicit ``backend="specialized"`` raises;
    :func:`is_sparse_mla_sm90_supported` then returns ``False``.
    ``FLASHINFER_DISABLE_SPARSE_MLA_SM90=1`` is a deprecated alias kept for
    one release.

    **Dispatch stats.** ``flashinfer.mla._sparse_mla_sm90.
    _sparse_mla_sm90_stats()`` reports per-backend dispatch counts
    (``specialized`` / ``reference``) and a ``fallback_reasons`` census.
    """
    _stats["calls"] += 1
    if backend not in BACKENDS:
        raise NotImplementedError(
            f"sparse_mla_decode_sm90 supports backend in {BACKENDS}; an "
            "explicitly selected backend is never silently swapped"
        )
    if out is None:
        out = torch.empty(
            (*query.shape[:-1], _CKV_DIM), device=query.device, dtype=query.dtype
        )
    scale = float(sm_scale)
    nq = int(query.shape[0]) * int(query.shape[1]) if query.ndim == 4 else 0

    # Backend resolution (#5667 order; GPU-free checks first).
    chosen = None
    if _disabled():
        if backend == "specialized":
            # An explicitly selected backend never falls back: refuse loudly.
            _fallback("specialized: kill switch")
            raise NotImplementedError(
                "sparse_mla_decode_sm90: backend='specialized' is disabled "
                f"({_SPECIALIZED_KERNEL_DISABLE_ENV}, or its deprecated alias "
                f"{_KILL_SWITCH_ENV}); an explicitly selected backend never "
                "falls back — use backend='auto' or 'reference'"
            )
        _fallback(f"{backend}: kill switch")
        logger.info_once(
            "sparse_mla_decode_sm90: disabled (%s, or its deprecated alias "
            "%s); the composable reference path serves every call",
            _SPECIALIZED_KERNEL_DISABLE_ENV,
            _KILL_SWITCH_ENV,
        )
        chosen = "reference"
    elif backend == "reference":
        chosen = "reference"
    elif backend == "specialized":
        if (
            query.ndim == 4
            and 1 <= nq <= _MAX_NQ
            and query.device.type == "cuda"
            and _device_supported(query.device)
        ):
            chosen = "specialized"
        else:
            # Explicit "specialized" that cannot serve this call RAISES with
            # the refusal reason — an explicitly selected backend never
            # falls back; only "auto" routes around a refusal.
            _fallback("specialized: device or token-row envelope")
            raise NotImplementedError(
                "sparse_mla_decode_sm90: backend='specialized' cannot serve "
                "this call (device or token-row envelope: needs SM90 with "
                f"the 93568B smem opt-in and 1 <= B*q_len <= {_MAX_NQ}); an "
                "explicitly selected backend never falls back — use "
                "backend='auto' or 'reference'"
            )
    else:  # "auto"
        routed, reason = _auto_routes(query, nq)
        if routed:
            chosen = "specialized"
        else:
            _fallback(f"auto: {reason}")
            logger.info_once(
                "sparse_mla_decode_sm90: backend='auto' does not route to "
                "'specialized' (%s); the composable reference path serves it",
                reason,
            )
            chosen = "reference"

    if chosen == "specialized":
        try:
            _get_run_fn()
        except Exception as exc:
            if torch.cuda.is_current_stream_capturing():
                raise  # the eager-warmup discipline stays loud under capture
            _fallback(f"specialized: JIT build failed ({type(exc).__name__})")
            if backend == "specialized":
                # An explicitly selected backend never falls back.
                raise NotImplementedError(
                    "sparse_mla_decode_sm90: backend='specialized' JIT build "
                    f"failed ({type(exc).__name__}: {str(exc)[:300]}); an "
                    "explicitly selected backend never falls back — use "
                    "backend='auto' or 'reference'"
                ) from exc
            logger.warning_once(
                "sparse_mla_decode_sm90: JIT build failed (%s: %s); the "
                "composable reference path serves this process",
                type(exc).__name__,
                str(exc)[:300],
            )
            chosen = "reference"

    signature = (
        query.shape,
        kv_cache.shape,
        indices.shape,
        seq_lens.shape,
        out.shape,
        workspace_buffer.shape,
        query.dtype,
        kv_cache.dtype,
        indices.dtype,
        seq_lens.dtype,
        workspace_buffer.dtype,
        out.dtype,
        query.device,
        backend,
        chosen,
    )
    if signature not in _validated_signatures:
        _validate(
            query,
            kv_cache,
            indices,
            seq_lens,
            workspace_buffer,
            out,
            scale,
            backend,
            fused=(chosen == "specialized"),
        )
        _validated_signatures.add(signature)
    _validate_per_call(
        query,
        kv_cache,
        indices,
        seq_lens,
        workspace_buffer,
        out,
        scale,
        fused=(chosen == "specialized"),
    )
    if chosen == "specialized":
        _run(query, kv_cache, indices, seq_lens, workspace_buffer, out, scale)
        _stats["backend_dispatches"]["specialized"] += 1
    else:
        _composable_reference(query, kv_cache, indices, out, scale)
        _stats["backend_dispatches"]["reference"] += 1
    return out
