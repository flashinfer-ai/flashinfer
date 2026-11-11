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
import math
import os

import torch
import tvm_ffi

from ..api_logging import flashinfer_api
from ..jit.mla import gen_sparse_mla_sm90_module
from ..utils import register_custom_op, register_fake_op

_KILL_SWITCH_ENV = "FLASHINFER_DISABLE_SPARSE_MLA_SM90"

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
    return os.environ.get(_KILL_SWITCH_ENV, "0") not in ("", "0", "false", "False")


def _split_plan(nq: int):
    """Mirror of the kernel's launch heuristic (pure function of nq)."""
    tile = 48 if nq <= 12 else 64
    maxt = (_TOPK + tile - 1) // tile
    per = max(1, min(264 // nq if nq <= 264 else 1, maxt))
    tps = max(1, (maxt + per - 1) // per)
    nsplits = (maxt + tps - 1) // tps
    return tile, tps, nsplits


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
        Bytes of FP16 partial output plus FP32 partial LSE for the split-K
        plan the kernel will choose for ``batch_size * q_len_per_req`` token
        rows. Zero when the plan has a single split. The capacity is a pure
        function of the token-row count, so a buffer sized at the largest
        row count a caller will use is valid for every smaller call — size
        it once, outside CUDA-graph capture.
    """
    nq = batch_size * q_len_per_req
    if nq < 1 or nq > _MAX_NQ:
        raise NotImplementedError(
            f"sparse_mla_decode_sm90 qualifies 1 <= batch*q_len <= {_MAX_NQ}, got {nq}"
        )
    _, _, nsplits = _split_plan(nq)
    if nsplits <= 1:
        return 0
    return nq * nsplits * (_NUM_HEADS * _CKV_DIM) * 2 + nq * nsplits * _NUM_HEADS * 4


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
    return torch.cuda.get_device_capability(device) == (9, 0) and bool(
        _get_module().device_supported(device.index)
    )


@flashinfer_api
def is_sparse_mla_sm90_supported(device: torch.device) -> bool:
    """Whether ``device`` qualifies for :func:`sparse_mla_decode_sm90`.

    True only on compute capability 9.0 devices with at least 93568 bytes of
    opt-in dynamic shared memory per block, and only while the
    ``FLASHINFER_DISABLE_SPARSE_MLA_SM90`` kill switch is unset. Calling this
    builds the JIT module, so call it (or the op) eagerly before any CUDA
    Graph capture.
    """
    if _disabled():
        return False
    return _device_supported(torch.device(device))


def _validate(query, kv_cache, indices, seq_lens, workspace, out, sm_scale, backend):
    if backend not in ("auto", "sm90"):
        raise NotImplementedError(
            "sparse_mla_decode_sm90 supports backend='auto' or 'sm90' only"
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
    if nq < 1 or nq > _MAX_NQ:
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
    required = get_sparse_mla_sm90_workspace_size_bytes(b, sq)
    if (
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
    if any(t.data_ptr() % 16 for t in (query, kv_cache, workspace, out)):
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
    if not _device_supported(query.device):
        raise NotImplementedError(
            "sparse_mla_decode_sm90 requires an SM90 device with at least "
            "93568 bytes of opt-in shared memory"
        )


# Python-side validation runs once per argument signature (below); the C++
# binding independently re-checks every structural requirement (dtype, dims,
# contiguity, alignment, workspace capacity, device support) on every call,
# so the cache only affects which layer raises, never what is enforced.
# This keeps the eager per-call host path short enough not to starve the
# GPU at decode-size grids (the kernel runs ~12 us at B1).
_validated_signatures: set = set()


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
    with tvm_ffi.use_raw_stream(
        _ffi_device(query.device.index), stream.cuda_stream
    ):
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
        ``"auto"`` or ``"sm90"``; both run the SM90 kernel. There is no
        hidden fallback: on any other device, or outside the qualified shape
        contract, this raises ``NotImplementedError`` instead of silently
        dispatching a different operator.

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
    capturable. Required discipline: (1) call once eagerly first — JIT
    module build and the one-time kernel shared-memory attribute setup must
    not happen under capture (module build raises if attempted while
    capturing); (2) the split plan is a pure function of ``B * q_len``, so a
    graph replays correctly for the captured shape; capture each shape
    separately; (3) keep ``query``/``kv_cache``/``indices``/``seq_lens``/
    ``workspace_buffer``/``out`` at stable addresses across replays (update
    cache contents and index values in place).

    **Kill switch.** Set ``FLASHINFER_DISABLE_SPARSE_MLA_SM90=1`` to disable
    the API cleanly (calls raise ``NotImplementedError`` and
    :func:`is_sparse_mla_sm90_supported` returns ``False``).
    """
    if _disabled():
        raise NotImplementedError(
            f"sparse_mla_decode_sm90 is disabled via {_KILL_SWITCH_ENV}. "
            "FlashInfer ships no other sparse-MLA path on SM90; dispatch to "
            "an external sparse-MLA implementation instead."
        )
    if out is None:
        out = torch.empty(
            (*query.shape[:-1], _CKV_DIM), device=query.device, dtype=query.dtype
        )
    scale = float(sm_scale)
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
    )
    if signature not in _validated_signatures:
        _validate(
            query, kv_cache, indices, seq_lens, workspace_buffer, out, scale, backend
        )
        _validated_signatures.add(signature)
    _run(query, kv_cache, indices, seq_lens, workspace_buffer, out, scale)
    return out
