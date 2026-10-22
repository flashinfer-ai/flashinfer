# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Semantic preparation and allocation-free replay for SM110 attention.

The host side of the Cake SM110 XQA package: argument validation, the route
(``cake_jit.select_record`` / ``select_decode_record``), the launch grid
(``cake_jit.launch_grid``), the workspace of the D128 decode route and the
positional argument list of the exported ``run`` entry, bound once at prepare
time (``cake_jit.argument_list``) so a replay interprets no plan.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import tvm_ffi

from . import cake_jit

#: The D512 tree schedules a caller may ask for: ``auto`` picks ``pair`` for an
#: E4M3 cache whose KV heads have an even number of 128-row Q tiles and ``tmem``
#: otherwise; ``tmem`` and ``pair`` select one schedule explicitly.
SELECTABLE_KERNELS = cake_jit.KERNELS
#: D512 tree routes retired with the Cake export: the ``tcgen05`` frozen tree and
#: the three register ``mma.sync`` families were superseded by the ``tmem`` and
#: ``pair`` schedules.  Asking for one raises ``ValueError`` naming the retirement.
RETIRED_KERNELS = ("tcgen05", "register_mma", "register_mma_split", "register_mma_auto")
#: The names a D128 decode call accepts: ``auto`` (the default) and ``tcgen05``,
#: the route's name before the export (its programs are tcgen05 programs; the
#: retirement above concerns the D512 tree routes only).  Any other name keeps
#: the previous rejection and message.
DECODE_KERNELS = ("auto", "tcgen05")
#: Byte alignment the exported programs need of each tensor's base address
#: (TMA descriptors and 16-byte vector accesses); checked at prepare, before any
#: launch, with the previous release's message.
_ALIGN_TMA = 16
_ALIGN_STATISTICS = 8
_ALIGN_WORD = 4

_HEAD_DIM_TREE = cake_jit.HEAD_DIM
_HEAD_DIM_DECODE = 128
_PAGE = 128


def _cuda_version() -> tuple[int, int]:
    text = torch.version.cuda or "0.0"
    parts = text.split(".")
    return (
        int(parts[0]),
        int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
    )


def require_sm110(device: torch.device) -> None:
    """Refuse a device other than the one exported target: exact compute
    capability 11.0 (``cake_jit.ARCH_BY_CAPABILITY``) with CUDA 13.0 or newer."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("device must be a cuda device")
    capability = tuple(torch.cuda.get_device_capability(device))
    if cake_jit.arch_for_capability(capability) is None or _cuda_version() < (13, 0):
        raise RuntimeError(
            "SM110 XQA requires exact compute capability 11.0 and CUDA 13.0 or newer"
        )


@functools.cache
def _require_sm110(device: torch.device) -> None:
    require_sm110(device)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _entry(record: str, device_index: int) -> Callable:
    """The ``run`` entry of a registry record for the device (built or loaded once per process)."""
    return cake_jit.entry(record, device_index)


def _tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    dtype: torch.dtype,
    device=None,
    shape=None,
    alignment: int = 1,
) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
        raise TypeError(f"{name} must be a {dtype} tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA storage")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must share device {device}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} has insufficient address alignment")


def _finite_scale(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class PreparedAttention:
    """Prepared program and owned tensor bindings; call ``run`` for replay.

    Shapes and addresses stay fixed. Tensor contents, including sequence
    lengths, mask and page indices, may be updated between calls. Device-side
    metadata values must remain within the contract described in README.md.
    Workspace is mutable: launches and re-preparation sharing it must be
    ordered. A new stream must wait for preparation and prior launches using
    standard PyTorch stream dependencies. ``route`` names the registry record
    the plan launches (``cake_jit.MODULES``), ``program`` its library (the
    exported program), ``grid`` its launch grid; ``inputs`` are the tensors the
    launch reads (the caller's, plus the uniform query prefix the backend
    synthesizes), retained for the life of the plan.
    """

    output: torch.Tensor
    route: str
    program: str
    workspace: tuple[torch.Tensor, ...]
    inputs: tuple[torch.Tensor, ...]
    grid: tuple[int, int, int]
    _execute: Callable[[], None] = field(repr=False)

    def run(self) -> torch.Tensor:
        """Submit on the caller's current stream without allocating tensors."""
        with torch.cuda.device(self.output.device):
            self._execute()
        return self.output


def prepare(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    q_cu_seq_lens: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    sm_scale: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    partition_tokens: Optional[int] = None,
    kernel: str = "auto",
) -> PreparedAttention:
    """Validate tensor metadata, load the native program and prepare replay.

    D512 accepts uniform or packed tree queries with contiguous FP16/FP8 KV
    or page128 KV. D128 accepts one FP16 decode query and contiguous FP16 KV.
    ``kernel`` selects the D512 schedule: ``"auto"`` (default: ``pair`` for
    E4M3 KV with an even Q-tile count per KV head, ``tmem`` otherwise),
    ``"tmem"`` (the tcgen05/TMEM-accumulator schedule: 128-row Q tile x
    256-column output half per CTA, Q and K/V fetched once per thread-block
    cluster by TMA multicast; every D512 cache mode, GQA ratios 2/4/8/16) or
    ``"pair"`` (the ``cta_group::2`` pair schedule: a (4, 1, 1) cluster issues
    one MMA stream for the two 128-row Q tiles of a KV head; E4M3 KV only,
    even Q-tile counts per head, GQA ratios 2/4/8/16).  The retired names
    ``RETIRED_KERNELS`` raise ``ValueError``.  D128 decode takes
    ``kernel="auto"`` or its previous name ``"tcgen05"`` (``DECODE_KERNELS``);
    the record is chosen from the partition count.
    Preparation performs compilation and may allocate output/scratch; call it
    before graph capture. D128 counters are zeroed once on the current stream;
    another launch stream must explicitly wait for that preparation stream.
    No device metadata is copied to the CPU.
    """
    _tensor(q, "q", dtype=torch.float16, alignment=_ALIGN_TMA)
    _require_sm110(q.device)
    if kernel not in SELECTABLE_KERNELS and kernel not in RETIRED_KERNELS:
        raise ValueError(f"kernel must be one of {SELECTABLE_KERNELS}")
    if q.ndim not in (3, 4) or q.shape[-1] not in (_HEAD_DIM_DECODE, _HEAD_DIM_TREE):
        raise ValueError("q must be rank 3 or 4 with head dimension 128 or 512")
    dim = q.shape[-1]
    decode = dim == _HEAD_DIM_DECODE
    if decode:
        if kernel not in DECODE_KERNELS:
            raise ValueError("D128 decode has only the tcgen05 kernel family")
    elif kernel in RETIRED_KERNELS:
        raise ValueError(
            f"kernel {kernel!r} is retired: the D512 tcgen05 and register_mma tree "
            "routes were superseded by the tmem and pair schedules; use "
            "kernel='auto', 'tmem' or 'pair'"
        )
    if kv.dtype not in (torch.float16, torch.float8_e4m3fn):
        raise TypeError("kv must contain FP16 or E4M3 values")
    _tensor(kv, "kv", dtype=kv.dtype, device=q.device, alignment=_ALIGN_TMA)
    if page_table is None:
        if page_size != 0 or kv.ndim != 5 or kv.shape[1] != 2 or kv.shape[-1] != dim:
            raise ValueError(
                "contiguous KV requires [B,2,Hkv,capacity,D] and page_size=0"
            )
        batch, _, heads, capacity, _ = kv.shape
        max_pages = 0
    else:
        if (
            page_size != _PAGE
            or dim != _HEAD_DIM_TREE
            or kv.ndim != 4
            or kv.shape[1] != _PAGE
            or kv.shape[-1] != dim
        ):
            raise ValueError(
                "paged KV requires [numPages,128,Hkv,512] and page_size=128"
            )
        _tensor(page_table, "page_table", dtype=torch.int32, device=q.device)
        if page_table.ndim != 3 or page_table.shape[1] != 2:
            raise ValueError("page_table must have shape [B,2,maxPages]")
        batch, _, max_pages = page_table.shape
        heads, capacity = kv.shape[2], max_pages * _PAGE
    if min(batch, heads, capacity) <= 0:
        raise ValueError("batch, KV heads and capacity must be positive")
    _tensor(
        sequence_lengths,
        "sequence_lengths",
        dtype=torch.int32,
        device=q.device,
        shape=(batch,),
    )

    if decode:
        if (
            page_table is not None
            or q_cu_seq_lens is not None
            or mask is not None
            or kv.dtype != torch.float16
        ):
            raise ValueError(
                "D128 supports one FP16 decode query with contiguous KV and no mask"
            )
        if q.ndim == 4 and q.shape[1] != 1:
            raise ValueError("D128 requires one query token")
        if q.shape[0] != batch or max_q_len not in (None, 1):
            raise ValueError("D128 query batch and maximum length do not match")
        q_heads, q_len = q.shape[-2], 1
    elif q_cu_seq_lens is None:
        if q.ndim != 4 or q.shape[0] != batch:
            raise ValueError("uniform D512 Q must be [B,Q,Hq,512]")
        q_len, q_heads = q.shape[1:3]
        if max_q_len is not None and max_q_len != q_len:
            raise ValueError("max_q_len differs from the uniform Q extent")
        _tensor(
            mask,
            "mask",
            dtype=torch.int32,
            device=q.device,
            shape=(batch, q_len, (q_len + 31) // 32),
        )
    else:
        if q.ndim != 3 or max_q_len is None or max_q_len <= 0:
            raise ValueError(
                "packed D512 Q requires [totalQ,Hq,512] and positive max_q_len"
            )
        q_len, q_heads = int(max_q_len), q.shape[1]
        _tensor(
            q_cu_seq_lens,
            "q_cu_seq_lens",
            dtype=torch.int32,
            device=q.device,
            shape=(batch + 1,),
        )
        _tensor(
            mask,
            "mask",
            dtype=torch.int32,
            device=q.device,
            shape=(q.shape[0], (q_len + 31) // 32),
        )
    ratios = cake_jit.DECODE_RATIOS if decode else cake_jit.RATIOS
    if (
        q_len <= 0
        or q_len > capacity
        or q_heads % heads
        or q_heads // heads not in ratios
    ):
        raise ValueError(
            f"query length must fit capacity and Hq/Hkv must be one of {ratios}"
        )
    ratio = q_heads // heads
    scales = (
        _finite_scale(
            1.0 / math.sqrt(dim) if sm_scale is None else sm_scale, "sm_scale"
        ),
        _finite_scale(k_scale, "k_scale"),
        _finite_scale(v_scale, "v_scale"),
    )
    if decode and scales[1:] != (1.0, 1.0):
        raise ValueError("D128 FP16 decode requires k_scale=v_scale=1")
    if out is None:
        out = torch.empty_like(q)
    # The decode program stores its output in 16-byte vectors; the tree programs
    # store FP16 elements.
    _tensor(
        out,
        "out",
        dtype=torch.float16,
        device=q.device,
        shape=q.shape,
        alignment=_ALIGN_TMA if decode else 1,
    )
    inputs = tuple(
        item
        for item in (q, kv, sequence_lengths, mask, page_table, q_cu_seq_lens)
        if item is not None
    )
    if any(
        out.untyped_storage().data_ptr() == item.untyped_storage().data_ptr()
        for item in inputs
    ):
        raise ValueError("out must not share storage with any input")
    if decode:
        return _prepare_decode(
            q,
            kv,
            sequence_lengths,
            out,
            workspace,
            partition_tokens,
            heads=heads,
            ratio=ratio,
            capacity=capacity,
            sm_scale=scales[0],
        )
    if workspace is not None:
        raise ValueError("the D512 tree route does not require external workspace")
    if partition_tokens is not None:
        raise ValueError("partition_tokens applies only to D128 decode")
    precision = "fp8" if kv.dtype == torch.float8_e4m3fn else "fp16"
    cache = "page128" if page_table is not None else "contig"
    # The record is the route: the schedule (``auto`` resolved from the cache
    # precision and the Q-tile parity), the precision, the layout and the ratio;
    # ``select_record`` refuses ``pair`` for FP16 KV or an odd Q-tile count.
    route = cake_jit.select_record(
        precision=precision, cache=cache, ratio=ratio, q_len=q_len, kernel=kernel
    )
    grid = cake_jit.launch_grid(route, num_kv_heads=heads, batch=batch, q_len=q_len)
    # The kernels read the query prefix (packed requests) through a non-null
    # pointer: uniform queries get the arithmetic prefix 0, Q, 2Q, ... once at
    # prepare; a packed call passes the caller's offsets.  Both are int32 values
    # the kernel reads as u32.
    if q_cu_seq_lens is None:
        prefix = torch.arange(
            0, (batch + 1) * q_len, q_len, device=q.device, dtype=torch.int32
        )
        inputs = (*inputs, prefix)
    else:
        prefix = q_cu_seq_lens
    # The generated binding takes the kernel's logical arguments: the TMA
    # sources (Q as its rows, an E4M3 cache as its bytes), typed buffers, the
    # by-value KV cache record member by member (the cache in its own dtype)
    # and the scalars; ``argument_list`` orders the record's own plan.
    kv_source = kv if precision == "fp16" else kv.view(torch.uint8)
    values = {
        "Q": q.view(-1, q_heads, _HEAD_DIM_TREE),
        "KV": kv_source,
        "KVV": kv_source,
        "q_seq_len": q_len,
        "num_kv_heads": heads,
        "head_group_size": ratio,
        "q_cu_seq_lens": prefix.view(torch.uint32),
        "attention_scale": scales[0],
        "output": out,
        "mask": mask.view(torch.uint32),
        "kv_cache_list.data": kv,
        "kv_cache_list.pool": kv,
        "kv_cache_list.sequence_lengths": sequence_lengths,
        "kv_cache_list.capacity": capacity,
        "kv_cache_list.max_pages": max_pages,
        "batch_size": batch,
        "k_cache_scale": scales[1],
        "v_cache_scale": scales[2],
    }
    if page_table is not None:
        values["kv_cache_list.page_list"] = page_table
    args = cake_jit.argument_list(route, values, grid=grid)
    run = _entry(route, _device_index(q.device))

    def execute_tree() -> None:
        with tvm_ffi.use_torch_stream():
            run(*args)

    return PreparedAttention(
        out, route, cake_jit.MODULES[route]["library"], (), inputs, grid, execute_tree
    )


def _prepare_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    partition_tokens: Optional[int],
    *,
    heads: int,
    ratio: int,
    capacity: int,
    sm_scale: float,
) -> PreparedAttention:
    # The partition length is the caller's or the registry default; the count
    # follows (``ValueError`` for a length that is not a positive multiple of
    # 64) and selects the record: a cluster program whose ``cluster_partitions``
    # equals the count when the registry ships one, else the program that merges
    # any count through the workspace.
    partitions, tokens = cake_jit.decode_partitions(capacity, partition_tokens)
    route = cake_jit.select_decode_record(ratio=ratio, partitions=partitions)
    batch, q_heads = q.shape[0], q.shape[-2]
    grid = cake_jit.launch_grid(
        route, num_kv_heads=heads, batch=batch, partitions=partitions
    )
    partial_shape, stats_shape = (
        (batch * q_heads, partitions, _HEAD_DIM_DECODE),
        (batch * q_heads, partitions, 2),
    )
    if workspace is None:
        workspace = (
            torch.empty(partial_shape, dtype=torch.float32, device=q.device),
            torch.empty(stats_shape, dtype=torch.float32, device=q.device),
            torch.empty((batch, heads), dtype=torch.int32, device=q.device),
        )
    if len(workspace) != 3:
        raise ValueError(
            "decode workspace must contain partial outputs, softmax statistics and counters"
        )
    partial, statistics, counters = workspace
    _tensor(
        partial,
        "partial",
        dtype=torch.float32,
        device=q.device,
        shape=partial_shape,
        alignment=_ALIGN_TMA,
    )
    _tensor(
        statistics,
        "statistics",
        dtype=torch.float32,
        device=q.device,
        shape=stats_shape,
        alignment=_ALIGN_STATISTICS,
    )
    _tensor(
        counters,
        "counters",
        dtype=torch.int32,
        device=q.device,
        shape=(batch, heads),
        alignment=_ALIGN_WORD,
    )
    addresses = [
        item.untyped_storage().data_ptr()
        for item in (q, kv, sequence_lengths, out, partial, statistics, counters)
    ]
    if len(addresses) != len(set(addresses)):
        raise ValueError(
            "decode workspace and input/output tensors must have disjoint storage"
        )
    # Preparation is outside replay/timing/capture. The last-CTA merge wraps each
    # counter back to zero; replay never resets it on the host.  The kernel takes
    # the int32 zeros as u32.
    counters.zero_()
    values = {
        "Q": q.view(batch, heads, ratio, _HEAD_DIM_DECODE),
        "KV": kv,
        "sequence_lengths": sequence_lengths,
        "output": out.view(batch, q_heads, _HEAD_DIM_DECODE),
        "partial": partial,
        "statistics": statistics,
        "counters": counters.view(torch.uint32),
        "heads": heads,
        "ratio": ratio,
        "capacity": capacity,
        "partitions": partitions,
        "partition_tokens": tokens,
        "attention_scale": sm_scale,
    }
    args = cake_jit.argument_list(route, values, grid=grid)
    run = _entry(route, _device_index(q.device))

    def execute_decode() -> None:
        with tvm_ffi.use_torch_stream():
            run(*args)

    return PreparedAttention(
        out,
        route,
        cake_jit.MODULES[route]["library"],
        tuple(workspace),
        (q, kv, sequence_lengths),
        grid,
        execute_decode,
    )


def attention(
    q: torch.Tensor, kv: torch.Tensor, sequence_lengths: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Prepare and execute attention once; reuse ``prepare(...).run()`` for replay."""
    return prepare(q, kv, sequence_lengths, **kwargs).run()
