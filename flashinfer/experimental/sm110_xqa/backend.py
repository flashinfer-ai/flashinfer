# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Semantic preparation and allocation-free replay for SM110 attention."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .jit import FROZEN, MODULES, ROUTES, load_sm110_xqa_module, require_sm110

TREE_KERNELS = ("tcgen05", "register_mma", "register_mma_split", "tmem", "pair")
SELECTABLE_KERNELS = TREE_KERNELS + ("register_mma_auto", "auto")
# D512 tree families: route suffix, Q-tile rows per CTA, grid.x CTAs per Q tile (the
# 512 output columns over each CTA's output tile) and Q tiles per thread-block cluster.
# The register families cover all 512 columns per CTA; tmem / pair split them over two
# 256-column halves; pair clusters the two Q tiles of a KV head.
_TREE_FAMILIES = {
    "register_mma": ("_mma", 32, 1, 1),
    "register_mma_split": ("_mma_split", 32, 1, 1),
    "tmem": ("_tmem", 128, 2, 1),
    "pair": ("_pair", 128, 4, 2),
}
# Argument kinds of a generated program's plan the host knows how to bind.
_PLAN_KINDS = {
    "parameter",
    "buffer",
    "tma_buffer",
    "pod_field",
    "nullable_raw_pointer",
    "grid",
}


@functools.cache
def _require_sm110(device: torch.device) -> None:
    require_sm110(device)


@functools.cache
def _entry(name: str) -> Callable:
    record = MODULES[name] if name in MODULES else FROZEN[name]
    return getattr(load_sm110_xqa_module(name), record["ffi_entry"])


def _ordered(name: str, bindings: dict[str, object]) -> list[object]:
    """``bindings`` in the generated program's argument-plan order.

    The plan is the program's own record of its ``run`` signature; every entry
    must be bound here and every kind must be one the host understands.
    """
    plan = MODULES[name]["arg_plan"]
    unknown = sorted({kind for kind, _ in plan} - _PLAN_KINDS)
    missing = [key for _, key in plan if key not in bindings]
    if unknown or missing:
        raise RuntimeError(
            f"{name}: cannot bind its argument plan "
            f"(unknown kinds {unknown}, unbound arguments {missing})"
        )
    return [bindings[key] for _, key in plan]


def _tensor(
    tensor: torch.Tensor, name: str, *, dtype: torch.dtype, device=None, shape=None
) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
        raise TypeError(f"{name} must be a {dtype} tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA storage")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must share device {device}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")


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
    standard PyTorch stream dependencies. ``program`` names the loaded
    translation units (``jit.MODULES`` or ``jit.FROZEN``); ``inputs`` are the
    caller tensors the launch reads, retained for the life of the plan.
    """

    output: torch.Tensor
    route: str
    program: str
    workspace: tuple[torch.Tensor, ...]
    inputs: tuple[torch.Tensor, ...]
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
    kernel: str = "tcgen05",
) -> PreparedAttention:
    """Validate tensor metadata, load the native program and prepare replay.

    D512 accepts uniform or packed tree queries with contiguous FP16/FP8 KV
    or page128 KV. D128 accepts one FP16 decode query and contiguous FP16 KV.
    ``kernel`` selects the D512 physical family: ``"tcgen05"`` (default, tensor
    memory), ``"register_mma"`` (warp-level ``mma.sync`` with a 32-row Q tile
    per CTA, split-K QK across idle warps and an in-kernel KV L2 prefetch),
    ``"register_mma_split"`` (the same Q tile, two eight-warp groups over the two
    KV halves with an in-CTA merge; FP16 page128 KV only) or
    ``"register_mma_auto"`` (``register_mma_split`` for FP16 page128 KV, the
    ``register_mma`` route for every other D512 cache mode), ``"tmem"`` (the
    tcgen05/TMEM-accumulator schedule: 128-row Q tile x 256-column output half per
    CTA, Q and K/V fetched once per thread-block cluster by TMA multicast; every
    D512 cache mode, GQA ratios 2/4/8/16), ``"pair"`` (the ``cta_group::2`` pair
    schedule: a (4, 1, 1) cluster issues one MMA stream for the two 128-row Q
    tiles of a KV head; E4M3 KV only, even Q-tile counts per head, GQA ratios
    2/4/8/16) or ``"auto"`` (``pair`` for E4M3 KV with an even Q-tile count per
    head, ``tmem`` otherwise).
    Preparation performs compilation and may allocate output/scratch; call it
    before graph capture. D128 counters are zeroed once on the current stream;
    another launch stream must explicitly wait for that preparation stream.
    No device metadata is copied to the CPU.
    """
    _tensor(q, "q", dtype=torch.float16)
    _require_sm110(q.device)
    if kernel not in SELECTABLE_KERNELS:
        raise ValueError(f"kernel must be one of {SELECTABLE_KERNELS}")
    if q.ndim not in (3, 4) or q.shape[-1] not in (128, 512):
        raise ValueError("q must be rank 3 or 4 with head dimension 128 or 512")
    dim = q.shape[-1]
    if kv.dtype not in (torch.float16, torch.float8_e4m3fn):
        raise TypeError("kv must contain FP16 or E4M3 values")
    _tensor(kv, "kv", dtype=kv.dtype, device=q.device)
    if page_table is None:
        if page_size != 0 or kv.ndim != 5 or kv.shape[1] != 2 or kv.shape[-1] != dim:
            raise ValueError(
                "contiguous KV requires [B,2,Hkv,capacity,D] and page_size=0"
            )
        batch, _, heads, capacity, _ = kv.shape
        max_pages = 0
    else:
        if (
            page_size != 128
            or dim != 512
            or kv.ndim != 4
            or kv.shape[1] != 128
            or kv.shape[-1] != dim
        ):
            raise ValueError(
                "paged KV requires [numPages,128,Hkv,512] and page_size=128"
            )
        _tensor(page_table, "page_table", dtype=torch.int32, device=q.device)
        if page_table.ndim != 3 or page_table.shape[1] != 2:
            raise ValueError("page_table must have shape [B,2,maxPages]")
        batch, _, max_pages = page_table.shape
        heads, capacity = kv.shape[2], max_pages * 128
    if min(batch, heads, capacity) <= 0:
        raise ValueError("batch, KV heads and capacity must be positive")
    _tensor(
        sequence_lengths,
        "sequence_lengths",
        dtype=torch.int32,
        device=q.device,
        shape=(batch,),
    )

    decode = dim == 128
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
        if kernel != "tcgen05":
            raise ValueError("D128 decode has only the tcgen05 kernel family")
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
    ratios = (4, 8, 16) if decode else (2, 4, 8, 16)
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
    _tensor(out, "out", dtype=torch.float16, device=q.device, shape=q.shape)
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
    layout = "paged" if page_table is not None else "contiguous"
    route = f"tree_{precision}_{layout}"
    fp16_paged = precision == "fp16" and layout == "paged"
    q_tiles = (q_len * ratio + 127) // 128
    if kernel == "auto":
        # The cta_group::2 pair schedule for E4M3 KV when every KV head has an even number of
        # 128-row Q tiles; the tcgen05/TMEM cluster-multicast schedule otherwise.
        kernel = "pair" if precision == "fp8" and q_tiles % 2 == 0 else "tmem"
    if kernel == "register_mma_auto":
        kernel = "register_mma_split" if fp16_paged else "register_mma"
    if kernel == "pair":
        if precision != "fp8":
            raise ValueError(
                "pair is frozen for E4M3 KV (contiguous or page128); use tmem or auto for FP16 KV"
            )
        if q_tiles % 2:
            raise ValueError(
                "pair needs an even number of 128-row Q tiles per KV head "
                "(ceil(q_len * ratio / 128)); use tmem or auto"
            )
    elif kernel == "register_mma_split" and not fp16_paged:
        raise ValueError(
            "register_mma_split is frozen for FP16 page128 KV only; use "
            "register_mma or register_mma_auto for other D512 cache modes"
        )
    if kernel == "tcgen05":
        frozen = FROZEN[route]
        tiles = (q_len * ratio + frozen["tile_rows"] - 1) // frozen["tile_rows"]
        grid = (512 // frozen["output_tile_columns"], heads * tiles, batch)
        args = (
            q,
            kv,
            sequence_lengths,
            mask,
            out,
            q_cu_seq_lens,
            page_table,
            q_len,
            heads,
            ratio,
            capacity,
            max_pages,
            *scales,
            *grid,
        )
        run_tree = _entry(route)

        def execute_frozen_tree() -> None:
            run_tree(*args)

        return PreparedAttention(out, route, route, (), inputs, execute_frozen_tree)

    suffix, tile_rows, ctas_per_tile, tiles_per_cluster = _TREE_FAMILIES[kernel]
    route += suffix
    # tmem / pair programs are instantiated per GQA ratio and cluster form (the pair
    # and (2, 2, 1) K/V-multicast forms need an even Q-tile count per KV head); the
    # register families take the ratio at runtime and have one form.
    if kernel.startswith("register_mma"):
        form = "any"
    else:
        form = "even" if q_tiles % 2 == 0 else "odd"
    program = ROUTES.get(f"{route}__{form}")
    if program is None:
        raise ValueError(f"no delivered program serves {route} ({form} cluster form)")
    dispatched = MODULES[program]["ratios"]
    if dispatched is not None and ratio not in dispatched:
        raise ValueError(f"{program} serves Hq/Hkv in {dispatched}, got {ratio}")
    tiles = (q_len * ratio + tile_rows - 1) // tile_rows
    # The generated binding takes the kernel's logical arguments: tensor-map sources
    # as the caller's tensors (E4M3 caches as their bytes), typed buffers, the
    # by-value KV cache record member by member, scalars, and the pointer arguments
    # the kernels never dereference (packed-query offsets when uniform, attention
    # sinks, semaphores, scratch) as raw device addresses, null when absent.
    kv_bytes = kv.view(torch.uint8)
    bindings = {
        "q_seq_len": q_len,
        "num_kv_heads": heads,
        "head_group_size": ratio,
        "q_cu_seq_lens": 0 if q_cu_seq_lens is None else q_cu_seq_lens.data_ptr(),
        "attention_scale": scales[0],
        "output": out,
        "q": q,
        "Q": q,
        "KV": kv if kv.dtype == torch.float16 else kv_bytes,
        "KVV": kv if kv.dtype == torch.float16 else kv_bytes,
        "mask": mask.view(torch.uint32),
        "attention_sinks": 0,
        "kv_cache_list.data": kv_bytes,
        "kv_cache_list.pool": kv_bytes,
        "kv_cache_list.sequence_lengths": sequence_lengths,
        "kv_cache_list.capacity": capacity,
        "kv_cache_list.max_pages": max_pages,
        "batch_size": batch,
        "k_cache_scale": scales[1],
        "v_cache_scale": scales[2],
        "semaphores": 0,
        "scratch": 0,
        "grid_x": ctas_per_tile * tiles_per_cluster,
        "grid_y": heads * (tiles // tiles_per_cluster),
        "grid_z": batch,
    }
    if page_table is not None:
        bindings["kv_cache_list.page_list"] = page_table
    ordered = _ordered(program, bindings)
    run = _entry(program)

    def execute_tree() -> None:
        run(*ordered)

    return PreparedAttention(out, route, program, (), inputs, execute_tree)


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
    route = "decode_fp16_contiguous"
    producer = FROZEN[route]
    tokens = (
        producer["partition_tokens"] if partition_tokens is None else partition_tokens
    )
    if type(tokens) is not int or not 0 < tokens <= (1 << 32) - 1 or tokens % 64:
        raise ValueError(
            "partition_tokens must fit uint32 and be a positive multiple of 64"
        )
    if not producer["fused_merge"]:
        raise RuntimeError("the delivered decode program must fuse its partition merge")
    partitions = (capacity + tokens - 1) // tokens
    if partitions == 1 and producer["merge_stats_cache"]:
        route = producer["single_partition_route"]
    batch, q_heads = q.shape[0], q.shape[-2]
    partial_shape, stats_shape = (
        (batch * q_heads, partitions, 128),
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
        partial, "partial", dtype=torch.float32, device=q.device, shape=partial_shape
    )
    _tensor(
        statistics,
        "statistics",
        dtype=torch.float32,
        device=q.device,
        shape=stats_shape,
    )
    _tensor(
        counters, "counters", dtype=torch.int32, device=q.device, shape=(batch, heads)
    )
    addresses = [
        item.untyped_storage().data_ptr()
        for item in (q, kv, sequence_lengths, out, partial, statistics, counters)
    ]
    if len(addresses) != len(set(addresses)):
        raise ValueError(
            "decode workspace and input/output tensors must have disjoint storage"
        )
    run_decode = _entry(route)
    # Preparation is outside replay/timing/capture. The last-CTA merge wraps each
    # counter back to zero; replay never resets it on the host.
    counters.zero_()
    args = (
        q.view(batch, q_heads, 128),
        kv,
        sequence_lengths,
        out.view(batch, q_heads, 128),
        partial,
        statistics,
        counters,
        heads,
        ratio,
        capacity,
        partitions,
        tokens,
        sm_scale,
        partitions,
        heads,
        batch,
    )

    def execute_decode() -> None:
        run_decode(*args)

    return PreparedAttention(
        out, route, route, tuple(workspace), (q, kv, sequence_lengths), execute_decode
    )


def attention(
    q: torch.Tensor, kv: torch.Tensor, sequence_lengths: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Prepare and execute attention once; reuse ``prepare(...).run()`` for replay."""
    return prepare(q, kv, sequence_lengths, **kwargs).run()
