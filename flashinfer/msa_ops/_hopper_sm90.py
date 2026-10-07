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

Compute-capability 9.0 (Hopper) Cake backend for MiniMax Sparse Attention.

The Cake-generated programs behind this module are registered in
``flashinfer.jit.hopper_msa``: ``ROUTES`` maps a logical route
(``<route key>:<stage>``) to the program that serves it and ``MODULES``
carries each program's physical argument order.  This module owns the public
semantics only: argument validation, the host-side plan that selects the
sparse-decode variant (head fold, split count, ring depth, L2 policy of the
page loads) and the proxy-score CTA order, the scratch contract of the split
decode, and the CUDA-graph rules.  Dispatch reads host-known scalars and
tensor metadata only; nothing here synchronizes the stream.

Routing on SM90 stays in ``flashinfer.msa_ops._sm90_dispatch``: it keeps the
surface's raise rules (paged KV only, K and V interleaved in one allocation,
fp8 e4m3 KV, bf16 q and output, head_dim 128, block 128, topk 16, no LSE,
no per-tensor K/V scales) and calls the functions below for the operation /
shape classes a Cake program serves.

Graph capture: every program launches on the current torch stream through
TVM-FFI and allocates nothing.  The split-decode scratch (partials, merge
state, monotonic per-item counters) is allocated once per
``(device, items, splits, head fold)`` and retained for the process, so a
captured decode keeps valid device pointers after later, larger captures;
capturing a decode whose scratch has not been warmed by an eager call raises.
"""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch
import tvm_ffi

from ..jit.hopper_msa import MODULES, load_hopper_msa_module, route_program
from ..utils import get_compute_capability, get_device_sm_count

_BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOPK = 16
_SUPPORTED_COMPUTE_CAPABILITIES = {(9, 0)}
_LOG2E = 1.4426950408889634
_GRID_AXES = {"grid_x": 0, "grid_y": 1, "grid_z": 2}

# Decode program geometry (fixed by the schedule family).
_KT = 64  # keys per WGMMA M half of a page tile
_HALF_BYTES = _KT * _HEAD_DIM  # one fp8 K (or V) half page
_PAGE_BYTES = 4 * _HALF_BYTES  # one ring stage: K half 0, K half 1, V half 0, V half 1
_IDENT_BYTES = 64 * _HEAD_DIM  # fp8 identity tile
_WARPGROUP = 128  # threads per CTA
_SMEM_PER_SM = 228 * 1024
_HINT_REUSE_THRESHOLD = 1.5
_HINT_STREAM = "evict_first"
_HINT_DEFAULT = "none"
# Proxy-score decode program coordinates.
_PROXY_HEADS = (1, 2, 4)
_PROXY_QLENS = (1, 2, 4)
# Both programs take a trace carrier they never write (tracing is compiled out).
_TRACE_WORDS = 12


def is_hopper_msa_device(device: torch.device | str) -> bool:
    """Return whether ``device`` is a compute-capability 9.0 MSA target."""

    normalized = torch.device(device)
    return (
        normalized.type == "cuda"
        and get_compute_capability(normalized) in _SUPPORTED_COMPUTE_CAPABILITIES
    )


# ---------------------------------------------------------------------------
# Device facts and program launch
# ---------------------------------------------------------------------------


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _num_sms(device_index: int) -> int:
    """Multiprocessor count of one device, resolved once."""

    device = torch.device("cuda", device_index)
    compute_capability = get_compute_capability(device)
    if compute_capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "the Hopper MSA backend requires compute capability 9.0; "
            f"got {compute_capability[0]}.{compute_capability[1]}"
        )
    return int(get_device_sm_count(device))


class _Program:
    """One loaded program: its FFI entry and physical argument order."""

    __slots__ = ("entry", "plan", "name")

    def __init__(
        self, name: str, entry: Any, plan: tuple[tuple[str, str], ...]
    ) -> None:
        self.name = name
        self.entry = entry
        self.plan = plan

    def launch(self, grid: tuple[int, int, int], **arguments: Any) -> None:
        """Launch on the current torch stream with the generated argument order."""

        values = []
        for kind, name in self.plan:
            if kind == "grid":
                values.append(int(grid[_GRID_AXES[name]]))
            else:
                values.append(arguments[name])
        with tvm_ffi.use_torch_stream():
            self.entry(*values)


@functools.cache
def _program(route: str) -> _Program:
    name = route_program(route)
    record = MODULES[name]
    module = load_hopper_msa_module(name)
    plan = tuple((str(kind), str(argument)) for kind, argument in record["arg_plan"])
    return _Program(name, getattr(module, record["ffi_entry"]), plan)


_trace_carriers: dict[int, torch.Tensor] = {}
_trace_carriers_lock = threading.Lock()


def _trace_carrier(device: torch.device) -> torch.Tensor:
    """The never-written trace buffer every program takes, allocated once per device."""

    index = _device_index(device)
    with _trace_carriers_lock:
        tensor = _trace_carriers.get(index)
        if tensor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Hopper MSA programs must run eagerly once per device before CUDA graph capture"
                )
            tensor = torch.zeros((_TRACE_WORDS,), dtype=torch.uint64, device=device)
            _trace_carriers[index] = tensor
    return tensor


# ---------------------------------------------------------------------------
# Sparse decode: plan, scratch, launch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HopperDecodePlan:
    """Physical decode variant: head fold, split count, ring depth, prologue issue, L2 policy."""

    nc: int
    chunks: int
    splits: int
    stages: int
    pre: int
    hint: str

    @property
    def route(self) -> str:
        return f"decode_v5:nc{self.nc}:s{self.splits}:st{self.stages}:{self.hint}:main"


def _decode_smem_bytes(nc: int, stages: int) -> int:
    off_ident = stages * _PAGE_BYTES
    off_q16 = off_ident + _IDENT_BYTES
    off_p16 = off_q16 + nc * 2 * _HEAD_DIM
    off_red = off_p16 + 2 * 2 * nc * 2 * _KT
    off_lred = off_red + nc * 4 * 4
    off_flag = off_lred + nc * 4 * 4
    off_meta = off_flag + 16
    total = off_meta + 2 * _TOPK * 4
    return -(-total // 128) * 128


def _max_splits(nc: int, stages: int) -> int:
    return min(_TOPK, (stages * _PAGE_BYTES) // (_WARPGROUP * nc * 2))


def plan_sparse_decode(
    *,
    total_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_sms: int,
    seqlen_q: int,
    max_pages: int,
) -> HopperDecodePlan:
    """Select the decode variant from a latency model of the per-CTA chain and the wave count.

    Exact port of the Cake source planner: the head fold follows the GQA group
    (8 heads per CTA up to group 8, else 16), the split count and ring depth
    minimize the modelled cost, and the page loads use the default L2 policy
    only when the tokens of one sequence re-select the same pages inside the
    launch (``seqlen_q * topk / max_pages`` at or above 1.5).
    """

    group = num_q_heads // num_kv_heads
    nc = 8 if group <= 8 else 16
    chunks = -(-group // nc)
    base = total_q * num_kv_heads * chunks
    c_page = 0.9 if nc == 8 else 1.3
    hbm_us_per_byte = 1.0e6 / 3.0e12
    best = None
    for stages in (2, 3):
        smem = _decode_smem_bytes(nc, stages)
        per_sm = max(1, min(2, _SMEM_PER_SM // (smem + 1024)))
        resident = num_sms * per_sm
        for splits in (1, 2, 4, 8, 16):
            if splits > _max_splits(nc, stages):
                continue
            pages = _TOPK // splits
            grid = base * splits
            active = min(grid, resident)
            waves = -(-grid // resident)
            page_stream = active * _PAGE_BYTES * hbm_us_per_byte
            first_land = 1.0 + page_stream
            later = max(c_page, page_stream)
            epi = (
                0.2
                if splits == 1
                else 0.8 + 0.25 + 0.056 * splits * (nc // 8) * (1.0 if nc == 8 else 1.6)
            )
            chain = 0.9 + first_land + (pages - 1) * later + c_page + epi
            launch = 0.004 * max(0, active - 128)
            cost = 0.7 + launch + chain + (waves - 1) * (pages * later + 1.5)
            if nc == 8 and grid <= num_sms and pages == 1:
                cost -= 0.3
            if stages == 3:
                cost += 0.1
            if best is None or cost < best[0]:
                best = (cost, splits, stages)
    if best is None:
        raise ValueError("no admissible decode variant")
    _, splits, stages = best
    reuse = (seqlen_q * _TOPK) / max_pages if max_pages else 0.0
    hint = _HINT_DEFAULT if reuse >= _HINT_REUSE_THRESHOLD else _HINT_STREAM
    return HopperDecodePlan(
        nc=nc, chunks=chunks, splits=splits, stages=stages, pre=1, hint=hint
    )


_decode_scratch: dict[tuple[int, int, int, int], tuple[torch.Tensor, ...]] = {}
_decode_scratch_lock = threading.Lock()


def _scratch(
    device: torch.device, *, items: int, splits: int, nc: int
) -> tuple[torch.Tensor, ...]:
    """Split partials, merge statistics and the monotonic per-item counters of the decode program.

    Retained for the process per ``(device, items, splits, nc)``: the counters
    advance across launches (the merge protocol never resets them) and a
    captured graph keeps pointers into these buffers.
    """

    key = (_device_index(device), int(items), int(splits), int(nc))
    with _decode_scratch_lock:
        buffers = _decode_scratch.get(key)
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "msa_sparse_decode_attention must be invoked eagerly with this batch, head and "
                    "page-table geometry before CUDA graph capture (its split scratch is allocated once)"
                )
            buffers = (
                torch.empty(
                    (items * splits * _WARPGROUP * (nc // 2),),
                    dtype=torch.uint32,
                    device=device,
                ),
                torch.empty(
                    (items * splits * 2 * nc,), dtype=torch.float32, device=device
                ),
                torch.zeros((items,), dtype=torch.uint32, device=device),
                torch.zeros((items,), dtype=torch.uint32, device=device),
            )
            _decode_scratch[key] = buffers
    return buffers


def hopper_msa_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    softmax_scale: Optional[float],
    out: torch.Tensor,
) -> torch.Tensor:
    """Sparse decode attention on compute capability 9.0.  Writes and returns ``out``.

    ``k`` and ``v`` are the ``(num_pages, num_kv_heads, 128, 128)`` fp8 e4m3
    halves of the interleaved cache (the layout contract is checked by the
    caller); the program reads them through their own strides.
    """

    total_q, num_q_heads, head_dim = (int(x) for x in q.shape)
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}, got {head_dim}")
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("SM90 sparse decode needs contiguous bf16 q")
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        raise ValueError("SM90 sparse decode needs an fp8 e4m3 KV cache")
    if (
        k.ndim != 4
        or k.shape[2] != _BLOCK_SIZE
        or k.shape[3] != _HEAD_DIM
        or k.shape != v.shape
    ):
        raise ValueError(
            f"paged k/v must be (num_pages, num_kv_heads, {_BLOCK_SIZE}, {_HEAD_DIM})"
        )
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("k and v must be dense along head_dim")
    num_kv_heads = int(k.shape[1])
    if seqlen_q <= 0 or total_q % seqlen_q:
        raise ValueError(
            f"q rows ({total_q}) must be batch_size * seqlen_q ({seqlen_q})"
        )
    batch = total_q // seqlen_q
    if q2k_indices.dtype != torch.int32 or not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous int32")
    if tuple(q2k_indices.shape) != (num_kv_heads, total_q, _TOPK):
        raise NotImplementedError(
            f"SM90 sparse decode serves q2k_indices of shape (num_kv_heads, total_q, {_TOPK}); "
            f"got {tuple(q2k_indices.shape)}"
        )
    if (
        page_table.dtype != torch.int32
        or not page_table.is_contiguous()
        or page_table.ndim != 2
    ):
        raise ValueError("page_table must be contiguous int32 (batch_size, max_pages)")
    if int(page_table.shape[0]) != batch:
        raise ValueError(
            f"page_table has {page_table.shape[0]} rows for batch_size {batch}"
        )
    if (
        seqused_k.dtype != torch.int32
        or not seqused_k.is_contiguous()
        or int(seqused_k.numel()) != batch
    ):
        raise ValueError(f"seqused_k must be contiguous int32 with {batch} entries")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("out must be a contiguous bf16 tensor shaped like q")
    device = q.device
    max_pages = int(page_table.shape[1])
    plan = plan_sparse_decode(
        total_q=total_q,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_sms=_num_sms(_device_index(device)),
        seqlen_q=seqlen_q,
        max_pages=max_pages,
    )
    items = total_q * num_kv_heads * plan.chunks
    part_o, part_ml, counters, done = _scratch(
        device, items=items, splits=plan.splits, nc=plan.nc
    )
    scale = (
        1.0 / math.sqrt(_HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    )
    _program(plan.route).launch(
        (num_kv_heads * plan.chunks * plan.splits, total_q, 1),
        Q32=q.view(torch.uint32),
        K=k.view(torch.uint8),
        V=v.view(torch.uint8),
        O=out,
        q2k_indices=q2k_indices,
        page_table=page_table,
        seqused_k=seqused_k,
        part_o=part_o,
        part_ml=part_ml,
        counters=counters,
        done=done,
        total_q=total_q,
        seqlen_q=int(seqlen_q),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        max_pages=max_pages,
        num_chunks=plan.chunks,
        softmax_scale_log2=scale * _LOG2E,
        zero_u32=0,
        trace=_trace_carrier(device),
    )
    return out


# ---------------------------------------------------------------------------
# Proxy score, decode regime
# ---------------------------------------------------------------------------


def proxy_decode_route_available(
    *, q_dtype: torch.dtype, num_q_heads: int, max_seqlen_q: int
) -> bool:
    """Whether a Cake proxy-score program serves these host-known coordinates.

    The decode-regime programs exist for fp8 e4m3 and bf16 q / index cache,
    ``Hq`` in {1, 2, 4} and ``max_seqlen_q`` in {1, 2, 4}; ``_sm90_dispatch``
    keeps its other decode schedules for the remaining admitted coordinates.
    """

    return (
        q_dtype in (torch.float8_e4m3fn, torch.bfloat16)
        and int(num_q_heads) in _PROXY_HEADS
        and int(max_seqlen_q) in _PROXY_QLENS
    )


def proxy_route(*, q_dtype: torch.dtype, num_q_heads: int, max_seqlen_q: int) -> str:
    dtype = "fp8" if q_dtype == torch.float8_e4m3fn else "bf16"
    return f"proxy_decode:{dtype}:hq{int(num_q_heads)}:sq{int(max_seqlen_q)}:main"


def plan_proxy_score(*, batch: int, max_k_tiles: int, num_pages: int) -> bool:
    """CTA order: walk the batch for a fixed tile unless the (batch x tiles) rectangle is ragged."""

    return bool(num_pages * 20 >= batch * max_k_tiles * 17)


def hopper_msa_proxy_score_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    per_head: torch.Tensor,
    max_seqlen_q: int,
    batch_size: int,
    q_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSA proxy score, decode regime, on compute capability 9.0.  Writes ``per_head``."""

    total_q, num_q_heads, head_dim = (int(x) for x in q.shape)
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}, got {head_dim}")
    if (
        k.ndim != 4
        or k.shape[1] != 1
        or k.shape[2] != _BLOCK_SIZE
        or k.shape[3] != _HEAD_DIM
    ):
        raise ValueError(
            f"the MSA index cache must be (num_pages, 1, {_BLOCK_SIZE}, {_HEAD_DIM})"
        )
    if q.dtype != k.dtype or not q.is_contiguous() or not k.is_contiguous():
        raise ValueError(
            "q and the index cache must be contiguous tensors of one dtype"
        )
    if not proxy_decode_route_available(
        q_dtype=q.dtype, num_q_heads=num_q_heads, max_seqlen_q=max_seqlen_q
    ):
        raise NotImplementedError(
            f"no Cake SM90 proxy-score program for dtype {q.dtype}, Hq {num_q_heads}, max_seqlen_q {max_seqlen_q}"
        )
    sq = int(max_seqlen_q)
    batch = int(batch_size)
    if total_q != batch * sq:
        raise ValueError(
            f"decode proxy needs total_q == batch_size * max_seqlen_q ({batch} * {sq}), got {total_q}"
        )
    if (
        page_table.dtype != torch.int32
        or page_table.ndim != 2
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            "page_table must be int32 (batch_size, max_pages) with unit column stride"
        )
    if int(page_table.shape[0]) != batch:
        raise ValueError(
            f"page_table has {page_table.shape[0]} rows for batch_size {batch}"
        )
    if (
        seqused_k.dtype != torch.int32
        or not seqused_k.is_contiguous()
        or int(seqused_k.numel()) != batch
    ):
        raise ValueError(f"seqused_k must be contiguous int32 with {batch} entries")
    num_heads, max_k_tiles, out_q = (int(x) for x in per_head.shape)
    if (
        (num_heads, out_q) != (num_q_heads, total_q)
        or per_head.dtype != torch.float32
        or not per_head.is_contiguous()
    ):
        raise ValueError(
            f"per_head must be contiguous float32 ({num_q_heads}, max_k_tiles, {total_q})"
        )
    if int(page_table.shape[1]) < max_k_tiles:
        raise ValueError("page_table is narrower than max_k_tiles")
    if q_offset is not None and (
        q_offset.dtype != torch.int32
        or not q_offset.is_contiguous()
        or int(q_offset.numel()) != batch
    ):
        raise ValueError(f"q_offset must be contiguous int32 with {batch} entries")
    num_pages = int(k.shape[0])
    if q.dtype == torch.float8_e4m3fn:
        q2 = q.view(torch.uint8).view(total_q * num_q_heads, _HEAD_DIM)
        k2 = k.view(torch.uint8).view(num_pages * _BLOCK_SIZE, _HEAD_DIM)
    else:
        q2 = q.view(total_q * num_q_heads, _HEAD_DIM)
        k2 = k.view(num_pages * _BLOCK_SIZE, _HEAD_DIM)
    batch_fast = plan_proxy_score(
        batch=batch, max_k_tiles=max_k_tiles, num_pages=num_pages
    )
    _program(
        proxy_route(q_dtype=q.dtype, num_q_heads=num_q_heads, max_seqlen_q=sq)
    ).launch(
        (batch, max_k_tiles, 1) if batch_fast else (max_k_tiles, batch, 1),
        Q=q2,
        K=k2,
        out=per_head,
        page_table=page_table,
        seqused_k=seqused_k,
        q_offset=q_offset if q_offset is not None else seqused_k,
        has_qoff=1 if q_offset is not None else 0,
        pt_stride=int(page_table.stride(0)),
        max_k_tiles=max_k_tiles,
        total_q=total_q,
        batch_fast=1 if batch_fast else 0,
        trace=_trace_carrier(q.device),
    )
    return per_head


__all__ = [
    "HopperDecodePlan",
    "hopper_msa_proxy_score_decode",
    "hopper_msa_sparse_decode_attention",
    "is_hopper_msa_device",
    "plan_proxy_score",
    "plan_sparse_decode",
    "proxy_decode_route_available",
    "proxy_route",
]
