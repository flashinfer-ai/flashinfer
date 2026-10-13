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

Cake backend: NVFP4 paged-KV MiniMax Sparse Attention decode (SM100/SM103/SM107).

The sm_107a (Rubin) programs are registered unconditionally but built only
when the CUDA toolkit of this checkout can emit ``compute_107a``
(``cake_jit.toolchain_supports``); on an older toolkit a 10.7 device has no
generated program and the routes decline it with that reason.

One persistent launch of the generated program serves a whole decode step over
the planar NVFP4 page pool described in
``docs/design_docs/nvfp4_msa_paged_kv_layout.md``.  Each work item is one
(query token, KV head) pair with its sixteen selected 128-token pages; the
kernel streams the packed E2M1 pages and E4M3 block scales with TMA,
dequantizes them to BF16 in shared memory, keeps the K tiles resident in tensor
memory and runs both MMAs in the swapped (S^T / O^T) orientation.  When the
batch is too small to fill the machine the page pairs of every item are split
across 2, 4 or 8 CTAs that launch as one cluster and merge their FP32 partials
through distributed shared memory (``split_factor``); no workspace is involved
and a prepared runner launches with no allocation, so it can be captured into
a CUDA Graph.  When the architecture also registers the short-item program,
batches whose requests span at most ``max_pages`` (four) selected pages run it
instead: one eight-CTA cluster of register-MMA CTAs per work item, two CTAs
per selected page, partials merged through distributed shared memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from ...utils import get_compute_capability, get_device_sm_count
from .cake_jit import (
    ARG_PLANS,
    PROGRAMS,
    load_program,
    programs_for,
    select_program,
    short_program,
    tail_programs,
    toolchain_supports,
)

HEAD_DIM = 128
PAGE_SIZE = 128  # tokens per page == sparse block size
TOPK = 16
SCALE_VEC = 16  # head-dim values per E4M3 block scale
DATA_DIM = HEAD_DIM // 2  # packed E2M1 bytes per token
SCALE_DIM = HEAD_DIM // SCALE_VEC  # E4M3 bytes per token
MAX_SEQLEN_Q = 32
MAX_GROUP_SIZE = 16  # query heads per KV head served by one softmax warp
SPLIT_FACTORS = (1, 2, 4, 8)
TMA_ALIGN = 16
SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
LN2 = math.log(2.0)
STRIDE_LIMIT = 2**31  # the short program's page / head byte strides are int32

# Argument-plan names the persistent program's ABI still lists but never
# reads (the global-partial merge path they belonged to is not built).  They
# are bound to tensors the launch already holds; the names disappear from
# ``ARG_PLANS`` together with the kernel ABI.
_PERSISTENT_ABI_CARRIERS = {
    "partial_O": "msa_lse",
    "partial_M": "msa_lse",
    "partial_D": "msa_lse",
    "split_completion": "task_kind",
    "kv_indptr": "task_kv_head",
    "task_request": "task_kv_head",
}
_SHORT_ABI_CARRIERS = {"kv_indptr": "task_kv_head", "task_request": "task_kv_head"}


# ---------------------------------------------------------------------------
# Device geometry and the split-KV rule
# ---------------------------------------------------------------------------


def arch_for(device: torch.device) -> Optional[str]:
    return SUPPORTED_COMPUTE_CAPABILITIES.get(get_compute_capability(device))


def generated_program_available(device: torch.device) -> bool:
    """True when this checkout registers a generated program for ``device``
    and its CUDA toolkit can compile that program (``cake_jit.toolchain_supports``)."""
    arch = arch_for(device)
    return arch is not None and toolchain_supports(arch) and bool(programs_for(arch))


def ctas_per_sm(arch: str) -> int:
    """Resident CTAs per SM of the persistent program (a compile-time property)."""
    values = {
        int(record["ctas_per_sm"])
        for record in programs_for(arch).values()
        if record["route"] == "persistent"
    }
    if len(values) != 1:
        raise NotImplementedError(
            f"The generated NVFP4 MSA decode program for {arch} is not registered in this checkout"
        )
    return values.pop()


def persistent_cta_capacity(device: torch.device) -> int:
    """Number of CTAs the persistent grid may hold on ``device``."""
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "NVFP4 MSA decode requires compute capability 10.0, 10.3 or 10.7"
        )
    return get_device_sm_count(device) * ctas_per_sm(arch)


def split_factor(total_work_items: int, cta_capacity: int, max_pages: int) -> int:
    """Split-KV factor for ``total_work_items`` (query token, KV head) items.

    One CTA per item whenever the items alone occupy at least half of the
    resident CTAs; otherwise the largest power of two (at most eight) that
    still fits every (item, split) CTA in one wave.  Items with fewer than
    four page pairs (``max_pages`` < 7) never split: publishing and merging
    partials costs more than the short serial chain it removes, and every
    split must own at least one page pair.
    """
    items = max(1, int(total_work_items))
    max_pairs = (min(max(int(max_pages), 0), TOPK) + 1) // 2
    if max_pairs < 4 or items * 2 > int(cta_capacity):
        return 1
    splits = 1
    while (
        splits < 8
        and splits * 2 <= max_pairs
        and items * splits * 2 <= int(cta_capacity)
    ):
        splits *= 2
    return splits


def persistent_grid(total_work_items: int, splits: int, cta_capacity: int) -> int:
    return max(1, min(int(total_work_items) * int(splits), int(cta_capacity)))


def tail_plan(
    arch: str, total_work_items: int, cta_capacity: int, max_pages: int
) -> Optional[tuple[int, int]]:
    """``(splits, grid)`` of a last-round split, or ``None`` for the plain persistent launch.

    Persistent rounds quantise a batch of N items on G resident CTAs to
    ceil(N / G) item-times.  A registered tail program (``tail_programs``) runs
    every full round unsplit and only the M = N - floor(N / G) G remainder
    items as S-way cluster units merged through distributed shared memory, on
    the largest S-aligned grid the part co-schedules as S-CTA clusters (the
    program's ``cluster_capacity`` for this SM count); the split round must fit
    that grid (M S <= G).  S minimises floor(N / G) + 1 / S below the plain
    makespan.  Items with fewer than four page pairs never split, a batch that
    already fits one round never splits, and a part without a capacity entry
    keeps the plain launch.
    """
    items = max(1, int(total_work_items))
    capacity = int(cta_capacity)
    max_pairs = (min(max(int(max_pages), 0), TOPK) + 1) // 2
    if max_pairs < 4 or items <= capacity:
        return None
    sms = str(capacity // ctas_per_sm(arch))
    best, best_cost = None, float(math.ceil(items / capacity))
    for _name, record in tail_programs(arch):
        s = int(record["splits"])
        if s > max_pairs:
            continue
        cluster_ctas = record.get("cluster_capacity", {}).get(sms)
        if cluster_ctas is None:
            continue
        grid = min((capacity // s) * s, int(cluster_ctas))
        if grid < s:
            continue
        full = items // grid
        rem = items - full * grid
        if rem == 0 or rem * s > grid:
            continue
        cost = full + 1.0 / s
        if cost < best_cost - 1e-9:
            best, best_cost = (s, grid), cost
    return best


# ---------------------------------------------------------------------------
# Short-item cluster program (work items of at most ``max_pages`` pages)
# ---------------------------------------------------------------------------


def short_program_record(arch: str) -> Optional[dict]:
    """Registry record of the short-item program for ``arch`` (``None`` when absent)."""
    name = short_program(arch)
    return None if name is None else PROGRAMS[name]


def short_route_applies(arch: str, max_pages: int) -> bool:
    """True when ``arch`` registers the short-item program and every request spans at most its page budget."""
    record = short_program_record(arch)
    return record is not None and int(max_pages) <= int(record["max_pages"])


def short_grid(
    total_work_items: int, num_sms: int, record: dict
) -> tuple[int, int, int]:
    """Cluster-launch grid: one ``cluster``-CTA cluster per item, at most ``max_clusters`` clusters."""
    cluster = int(record["cluster"])
    clusters = max(
        1,
        min(
            int(total_work_items),
            int(record["max_clusters"]),
            max(1, int(num_sms) // cluster),
        ),
    )
    return (clusters * cluster, 1, 1)


def _flat_page_operand(
    name: str, view: torch.Tensor, inner: int
) -> tuple[torch.Tensor, int, int]:
    """Flat byte alias of a validated ``[pages, heads, 128, inner]`` page view plus its byte strides.

    The binding checks contiguity per tensor while the pool views are strided
    windows of one planar pool, so each view is passed as the contiguous byte
    span it covers together with its page and head strides (dense rows are the
    layout contract; ``validate_msa_nvfp4_decode_inputs`` has checked them).
    """
    b = view.view(torch.uint8) if view.dtype != torch.uint8 else view
    pages, heads = int(b.shape[0]), int(b.shape[1])
    page_stride, head_stride = int(b.stride(0)), int(b.stride(1))
    if page_stride >= STRIDE_LIMIT or head_stride >= STRIDE_LIMIT:
        raise ValueError(
            f"{name} page / head strides ({page_stride}, {head_stride}) exceed the short "
            f"program's 32-bit stride parameters"
        )
    span = (pages - 1) * page_stride + (heads - 1) * head_stride + PAGE_SIZE * inner
    return b.as_strided((span,), (1,)), page_stride, head_stride


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class MSANvfp4DecodeRunner:
    """Launch the prepared NVFP4 MSA decode.

    Calling the runner or ``launch()`` writes the caller-owned ``out`` (and the
    natural-log softmax normalizer ``lse``) with no CUDA allocation and no
    host synchronization and returns ``out``.  Every tensor is read on device
    at each launch, so the same runner (or a CUDA Graph capturing it) stays
    valid when the caller writes new queries, selections, page ids or KV
    lengths into the bound buffers.  Prepare a new runner when shapes, dtypes
    or tensor bindings change.
    """

    program: str
    arch: str
    route: str  # ``persistent`` split-KV program or ``short`` cluster program
    splits: int
    tail: bool  # last-round split: cluster units on the remainder items only
    grid: tuple[int, int, int]
    arguments: tuple
    out: torch.Tensor
    lse: torch.Tensor
    entry: Callable[..., Any]

    def launch(self) -> torch.Tensor:
        # Tensor maps are encoded by the host binding and passed by value.
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.grid[0])


def _bind(
    program: str,
    arch: str,
    route: str,
    values: dict[str, Any],
    grid: tuple[int, int, int],
    carriers: dict[str, str],
) -> tuple[Callable[..., Any], tuple]:
    """Order ``values`` by the route's generated argument plan and load the program."""
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in ARG_PLANS[route]:
        if kind == "grid":
            arguments.append(grid_values[name])
        elif name in values:
            arguments.append(values[name])
        elif name in carriers:
            arguments.append(values[carriers[name]])
        else:
            raise ValueError(
                f"the {route} program's argument plan names {name!r}, which this backend does not bind"
            )
    module = load_program(program, arch)
    return module.run, tuple(arguments)


# ---------------------------------------------------------------------------
# Validation and preparation
# ---------------------------------------------------------------------------


def _check_page_view(
    name: str, view: torch.Tensor, inner: int, num_pages: int, num_kv_heads: int
):
    if view.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError(
            f"{name} must be a uint8 or float8_e4m3fn view of the page pool"
        )
    if view.ndim != 4 or tuple(view.shape) != (
        num_pages,
        num_kv_heads,
        PAGE_SIZE,
        inner,
    ):
        raise ValueError(
            f"{name} must be [num_pages, num_kv_heads, {PAGE_SIZE}, {inner}], got {tuple(view.shape)}"
        )
    strides = tuple(int(s) for s in view.stride())
    if strides[3] != 1 or strides[2] != inner:
        raise ValueError(
            f"{name} must hold each head's {PAGE_SIZE} x {inner} bytes contiguously "
            f"(strides {strides})"
        )
    if strides[1] % TMA_ALIGN or strides[0] % TMA_ALIGN:
        raise ValueError(
            f"{name} head and page strides must be multiples of {TMA_ALIGN} bytes"
        )
    if strides[1] < PAGE_SIZE * inner or strides[0] < num_kv_heads * strides[1]:
        raise ValueError(f"{name} head and page strides overlap")
    if view.data_ptr() % TMA_ALIGN:
        raise ValueError(f"{name} must start at a {TMA_ALIGN}-byte aligned address")


def validate_msa_nvfp4_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    seqlen_q: int,
    k_global_scale: float,
    v_global_scale: float,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
) -> tuple[int, int, int, int, int]:
    """Check shapes, dtypes and strides against the kernel's contract.

    Returns ``(batch, num_q_heads, num_kv_heads, num_pages, max_pages)``.
    Device placement is checked by :func:`prepare_msa_nvfp4_sparse_decode`.
    """
    seqlen_q = int(seqlen_q)
    if not 1 <= seqlen_q <= MAX_SEQLEN_Q:
        raise ValueError(f"seqlen_q must be in [1, {MAX_SEQLEN_Q}], got {seqlen_q}")
    if q.dtype != torch.bfloat16 or q.ndim != 3 or int(q.shape[2]) != HEAD_DIM:
        raise ValueError(
            f"q must be bfloat16 [batch * seqlen_q, num_q_heads, {HEAD_DIM}]"
        )
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    total_q, num_q_heads = int(q.shape[0]), int(q.shape[1])
    if total_q == 0 or total_q % seqlen_q:
        raise ValueError("q rows must be batch * seqlen_q with a positive batch")
    batch = total_q // seqlen_q
    if k.ndim != 4:
        raise ValueError(
            f"k must be a 4-D [num_pages, num_kv_heads, {PAGE_SIZE}, {DATA_DIM}] view"
        )
    num_pages, num_kv_heads = int(k.shape[0]), int(k.shape[1])
    if num_kv_heads == 0 or num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    group = num_q_heads // num_kv_heads
    if not 1 <= group <= MAX_GROUP_SIZE:
        raise ValueError(f"the GQA group must be in [1, {MAX_GROUP_SIZE}], got {group}")
    for name, view in (("k", k), ("v", v)):
        _check_page_view(name, view, DATA_DIM, num_pages, num_kv_heads)
    for name, view in (("k_scale", k_scale), ("v_scale", v_scale)):
        _check_page_view(name, view, SCALE_DIM, num_pages, num_kv_heads)
    if (
        q2k_indices.dtype != torch.int32
        or q2k_indices.ndim != 3
        or tuple(q2k_indices.shape[:2]) != (num_kv_heads, total_q)
        or not q2k_indices.is_contiguous()
    ):
        raise ValueError(
            f"q2k_indices must be contiguous int32 [num_kv_heads, batch * seqlen_q, {TOPK}]"
        )
    if int(q2k_indices.shape[2]) != TOPK:
        raise ValueError(
            f"this decode program serves exactly topk={TOPK} selected pages"
        )
    if (
        page_table.dtype != torch.int32
        or page_table.ndim != 2
        or int(page_table.shape[0]) != batch
        or not page_table.is_contiguous()
    ):
        raise ValueError("page_table must be contiguous int32 [batch, max_pages]")
    max_pages = int(page_table.shape[1])
    if max_pages == 0:
        raise ValueError("page_table must hold at least one page per request")
    if (
        seqused_k.dtype != torch.int32
        or seqused_k.ndim != 1
        or int(seqused_k.shape[0]) != batch
        or not seqused_k.is_contiguous()
    ):
        raise ValueError("seqused_k must be contiguous int32 [batch]")
    if not (float(k_global_scale) > 0.0 and float(v_global_scale) > 0.0):
        raise ValueError("k_global_scale and v_global_scale must be positive")
    if out is not None and (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != tuple(q.shape)
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous bfloat16 with q's shape")
    if lse is not None and (
        lse.dtype != torch.float32
        or tuple(lse.shape) != (total_q, num_q_heads)
        or not lse.is_contiguous()
    ):
        raise ValueError(
            "lse must be contiguous float32 [batch * seqlen_q, num_q_heads]"
        )
    return batch, num_q_heads, num_kv_heads, num_pages, max_pages


def prepare_msa_nvfp4_sparse_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    k_global_scale: float,
    v_global_scale: float,
    seqlen_q: int = 1,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> MSANvfp4DecodeRunner:
    """Validate and bind one NVFP4 paged-KV sparse decode step.

    The only allocations are the optional ``out`` / ``lse`` outputs when the
    caller does not pass them; the returned runner launches with none.  When
    the device registers the short-item program and every request spans at
    most its page budget (``short_route_applies``), the runner binds that
    program (``route == "short"``, one eight-CTA cluster per work item).
    Otherwise the split factor is decided from the batch geometry and the
    device's resident-CTA capacity (``split_factor``); a split item runs as
    one cluster of ``splits`` CTAs that merge through distributed shared
    memory.  A batch of more items than resident CTAs takes the registered
    last-round-split program instead when that shortens the makespan
    (``tail_plan``; ``runner.tail``).
    """
    if backend != "cake":
        raise ValueError("NVFP4 MSA decode supports backend='cake'")
    batch, num_q_heads, num_kv_heads, num_pages, max_pages = (
        validate_msa_nvfp4_decode_inputs(
            q,
            k,
            v,
            q2k_indices,
            k_scale,
            v_scale,
            page_table,
            seqused_k,
            seqlen_q=seqlen_q,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
            out=out,
            lse=lse,
        )
    )
    device = q.device
    tensors = [q, k, v, k_scale, v_scale, q2k_indices, page_table, seqused_k]
    tensors += [t for t in (out, lse) if t is not None]
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    arch = arch_for(device)
    if arch is None:
        capability = get_compute_capability(device)
        raise ValueError(
            "NVFP4 MSA decode requires compute capability 10.0, 10.3 or 10.7 "
            f"(got {capability[0]}.{capability[1]})"
        )
    total_q = batch * int(seqlen_q)
    total_work_items = total_q * num_kv_heads
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    # The global K multiplier is separable from every per-block scale: fold it
    # into the softmax scale; the global V multiplier is applied in the epilogue.
    softmax_scale_log2 = scale * float(k_global_scale) / LN2
    if out is None:
        out = q.new_empty((total_q, num_q_heads, HEAD_DIM))
    if lse is None:
        lse = q.new_empty((total_q, num_q_heads), dtype=torch.float32)
    scalars = dict(
        total_q=total_q,
        seqlen_q=int(seqlen_q),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        softmax_scale_log2=float(softmax_scale_log2),
        output_scale=float(v_global_scale),
        msa_max_pages=max_pages,
    )
    # Paged decode derives lengths from seqused_k and pages from page_table;
    # the generated names ``task_kind`` / ``task_kv_head`` are the selection
    # and the per-request KV lengths.
    metadata = dict(
        kv_indices=page_table, task_kind=q2k_indices, task_kv_head=seqused_k
    )

    if short_route_applies(arch, max_pages):
        name = short_program(arch)
        assert name is not None
        record = PROGRAMS[name]
        k_flat, k_ps, k_hs = _flat_page_operand("k", k, DATA_DIM)
        ks_flat, ks_ps, ks_hs = _flat_page_operand("k_scale", k_scale, SCALE_DIM)
        v_flat, v_ps, v_hs = _flat_page_operand("v", v, DATA_DIM)
        vs_flat, vs_ps, vs_hs = _flat_page_operand("v_scale", v_scale, SCALE_DIM)
        values = dict(
            Q=q,
            K=k_flat,
            K_scale=ks_flat,
            V=v_flat,
            V_scale=vs_flat,
            O=out,
            msa_lse=lse,
            **metadata,
            **scalars,
            k_page_stride=k_ps,
            k_head_stride=k_hs,
            ks_page_stride=ks_ps,
            ks_head_stride=ks_hs,
            v_page_stride=v_ps,
            v_head_stride=v_hs,
            vs_page_stride=vs_ps,
            vs_head_stride=vs_hs,
        )
        grid = short_grid(total_work_items, get_device_sm_count(device), record)
        entry, arguments = _bind(name, arch, "short", values, grid, _SHORT_ABI_CARRIERS)
        return MSANvfp4DecodeRunner(
            name, arch, "short", 1, False, grid, arguments, out, lse, entry
        )

    capacity = persistent_cta_capacity(device)
    splits = split_factor(total_work_items, capacity, max_pages)
    tail = (
        tail_plan(arch, total_work_items, capacity, max_pages) if splits == 1 else None
    )
    if tail is not None:
        # Last-round split: every full persistent round runs whole items and
        # only the remainder items of the last round run as ``splits``-CTA
        # cluster units, on the largest cluster-aligned grid the part
        # co-schedules.
        splits, grid = tail[0], (tail[1], 1, 1)
    else:
        grid = (persistent_grid(total_work_items, splits, capacity), 1, 1)
    name = select_program(arch, splits=splits, tail=tail is not None)
    values = dict(
        Q=q,
        K=k.view(torch.uint8),
        K_scale=k_scale.view(torch.uint8),
        V=v.view(torch.uint8),
        V_scale=v_scale.view(torch.uint8),
        O=out,
        msa_lse=lse,
        **metadata,
        **scalars,
    )
    entry, arguments = _bind(
        name, arch, "persistent", values, grid, _PERSISTENT_ABI_CARRIERS
    )
    return MSANvfp4DecodeRunner(
        name,
        arch,
        "persistent",
        int(splits),
        tail is not None,
        grid,
        arguments,
        out,
        lse,
        entry,
    )
