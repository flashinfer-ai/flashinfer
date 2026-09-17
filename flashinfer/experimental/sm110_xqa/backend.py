# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Semantic preparation and allocation-free replay for SM110 attention."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ...utils import register_custom_op, register_fake_op
from .jit import (
    gen_sm110_xqa_module,
    get_manifest,
    load_sm110_xqa_module,
    require_sm110,
)


@functools.cache
def _tree_op(route: str) -> Callable:
    spec = gen_sm110_xqa_module(route)
    module = load_sm110_xqa_module(route)
    name = f"flashinfer::{spec.name}"

    @register_custom_op(name, mutates_args=("output",))
    def run(
        q: torch.Tensor,
        kv: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        output: torch.Tensor,
        q_offsets: Optional[torch.Tensor],
        pages: Optional[torch.Tensor],
        q_len: int,
        heads: int,
        ratio: int,
        capacity: int,
        max_pages: int,
        sm_scale: float,
        k_scale: float,
        v_scale: float,
        grid_x: int,
        grid_y: int,
        grid_z: int,
    ) -> None:
        module.run_tree(
            q,
            kv,
            lengths,
            mask,
            output,
            q_offsets,
            pages,
            q_len,
            heads,
            ratio,
            capacity,
            max_pages,
            sm_scale,
            k_scale,
            v_scale,
            grid_x,
            grid_y,
            grid_z,
        )

    @register_fake_op(name)
    def fake(
        q: torch.Tensor,
        kv: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        output: torch.Tensor,
        q_offsets: Optional[torch.Tensor],
        pages: Optional[torch.Tensor],
        q_len: int,
        heads: int,
        ratio: int,
        capacity: int,
        max_pages: int,
        sm_scale: float,
        k_scale: float,
        v_scale: float,
        grid_x: int,
        grid_y: int,
        grid_z: int,
    ) -> None:
        pass

    return run


@functools.cache
def _decode_ops(
    producer_route: str, fused_merge: bool
) -> tuple[Callable, Optional[Callable]]:
    merge_route = "decode_merge"
    producer_name = f"flashinfer::{gen_sm110_xqa_module(producer_route).name}"
    producer = load_sm110_xqa_module(producer_route)

    @register_custom_op(
        producer_name, mutates_args=("output", "partial", "statistics", "counters")
    )
    def run(
        q: torch.Tensor,
        kv: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        partial: torch.Tensor,
        statistics: torch.Tensor,
        counters: torch.Tensor,
        heads: int,
        ratio: int,
        capacity: int,
        partitions: int,
        partition_tokens: int,
        sm_scale: float,
        grid_x: int,
        grid_y: int,
        grid_z: int,
    ) -> None:
        producer.run_decode(
            q,
            kv,
            lengths,
            output,
            partial,
            statistics,
            counters,
            heads,
            ratio,
            capacity,
            partitions,
            partition_tokens,
            sm_scale,
            grid_x,
            grid_y,
            grid_z,
        )

    @register_fake_op(producer_name)
    def fake_run(
        q: torch.Tensor,
        kv: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        partial: torch.Tensor,
        statistics: torch.Tensor,
        counters: torch.Tensor,
        heads: int,
        ratio: int,
        capacity: int,
        partitions: int,
        partition_tokens: int,
        sm_scale: float,
        grid_x: int,
        grid_y: int,
        grid_z: int,
    ) -> None:
        pass

    if fused_merge:
        return run, None
    merge_name = f"flashinfer::{gen_sm110_xqa_module(merge_route).name}"
    merge = load_sm110_xqa_module(merge_route)

    @register_custom_op(merge_name, mutates_args=("output",))
    def run_merge(
        partial: torch.Tensor,
        statistics: torch.Tensor,
        output: torch.Tensor,
        partitions: int,
        grid_x: int,
        grid_y: int,
        grid_z: int,
    ) -> None:
        merge.run_decode_merge(
            partial, statistics, output, partitions, grid_x, grid_y, grid_z
        )

    @register_fake_op(merge_name)
    def fake_merge(
        partial: torch.Tensor,
        statistics: torch.Tensor,
        output: torch.Tensor,
        partitions: int,
        grid_x: int,
        grid_y: int,
        grid_z: int,
    ) -> None:
        pass

    return run, run_merge


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
    """Prepared module and owned tensor bindings; call ``run`` for replay.

    Shapes and addresses stay fixed. Tensor contents, including sequence
    lengths, mask and page indices, may be updated between calls. Device-side
    metadata values must remain within the contract described in README.md.
    Workspace is mutable: launches and re-preparation sharing it must be
    ordered. A new stream must wait for preparation and prior launches using
    standard PyTorch stream dependencies.
    """

    output: torch.Tensor
    route: str
    workspace: tuple[torch.Tensor, ...]
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
) -> PreparedAttention:
    """Validate tensor metadata, load the native module and prepare replay.

    D512 accepts uniform or packed tree queries with contiguous FP16/FP8 KV
    or page128 KV. D128 accepts one FP16 decode query and contiguous FP16 KV.
    Preparation performs compilation and may allocate output/scratch; call it
    before graph capture. D128 counters are zeroed once on the current stream;
    another launch stream must explicitly wait for that preparation stream.
    No device metadata is copied to the CPU.
    """
    _tensor(q, "q", dtype=torch.float16)
    require_sm110(q.device)
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
    inputs = (q, kv, sequence_lengths, mask, page_table, q_cu_seq_lens)
    if any(
        item is not None
        and out.untyped_storage().data_ptr() == item.untyped_storage().data_ptr()
        for item in inputs
    ):
        raise ValueError("out must not share storage with any input")
    manifest = get_manifest()
    if decode:
        route = "decode_fp16_contiguous"
        metadata = manifest["routes"][route]
        tokens = (
            metadata["partition_tokens"]
            if partition_tokens is None
            else partition_tokens
        )
        if type(tokens) is not int or not 0 < tokens <= (1 << 32) - 1 or tokens % 64:
            raise ValueError(
                "partition_tokens must fit uint32 and be a positive multiple of 64"
            )
        fused_merge = metadata["fused_merge"]
        partitions = (capacity + tokens - 1) // tokens
        if partitions == 1 and metadata["merge_stats_cache"]:
            route = metadata["single_partition_route"]
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
            partial,
            "partial",
            dtype=torch.float32,
            device=q.device,
            shape=partial_shape,
        )
        _tensor(
            statistics,
            "statistics",
            dtype=torch.float32,
            device=q.device,
            shape=stats_shape,
        )
        _tensor(
            counters,
            "counters",
            dtype=torch.int32,
            device=q.device,
            shape=(batch, heads),
        )
        addresses = [
            item.untyped_storage().data_ptr()
            for item in (q, kv, sequence_lengths, out, partial, statistics, counters)
        ]
        if len(addresses) != len(set(addresses)):
            raise ValueError(
                "decode workspace and input/output tensors must have disjoint storage"
            )
        run, merge = _decode_ops(route, fused_merge)
        # Preparation is outside replay/timing/capture. A last-CTA merge wraps
        # each counter back to zero; replay never resets it on the host.
        counters.zero_()
        q_view, out_view = q.view(batch, q_heads, 128), out.view(batch, q_heads, 128)

        def execute() -> None:
            run(
                q_view,
                kv,
                sequence_lengths,
                out_view,
                partial,
                statistics,
                counters,
                heads,
                ratio,
                capacity,
                partitions,
                tokens,
                scales[0],
                partitions,
                heads,
                batch,
            )
            if partitions > 1 and merge is not None:
                merge(partial, statistics, out_view, partitions, batch * q_heads, 1, 1)

        return PreparedAttention(out, route, tuple(workspace), execute)
    if workspace is not None:
        raise ValueError("the D512 tree route does not require external workspace")
    if partition_tokens is not None:
        raise ValueError("partition_tokens applies only to D128 decode")
    precision = "fp8" if kv.dtype == torch.float8_e4m3fn else "fp16"
    layout = "paged" if page_table is not None else "contiguous"
    route = f"tree_{precision}_{layout}"
    metadata = manifest["routes"][route]
    grid = (
        512 // metadata["output_tile_columns"],
        heads * ((q_len * ratio + metadata["tile_rows"] - 1) // metadata["tile_rows"]),
        batch,
    )
    run = _tree_op(route)

    def execute() -> None:
        run(
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

    return PreparedAttention(out, route, (), execute)


def attention(
    q: torch.Tensor, kv: torch.Tensor, sequence_lengths: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Prepare and execute attention once; reuse ``prepare(...).run()`` for replay."""
    return prepare(q, kv, sequence_lengths, **kwargs).run()
