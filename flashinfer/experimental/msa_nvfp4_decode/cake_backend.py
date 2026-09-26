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

Cake backend: NVFP4 paged-KV MiniMax Sparse Attention decode (SM100/SM103).

One persistent launch of the generated program serves a whole decode step over
the planar NVFP4 page pool described in
``docs/design_docs/nvfp4_msa_paged_kv_layout.md``.  Each work item is one
(query token, KV head) pair with its sixteen selected 128-token pages; the
kernel streams the packed E2M1 pages and E4M3 block scales with TMA,
dequantizes them to BF16 in shared memory, keeps the K tiles resident in tensor
memory and runs both MMAs in the swapped (S^T / O^T) orientation.  When the
batch is too small to fill the machine the page pairs of every item are split
across 2, 4 or 8 CTAs whose FP32 partials are merged by the last CTA to finish
(``split_factor``); the split scratch lives in a caller-owned workspace so a
prepared runner launches with no allocation and can be captured into a CUDA
Graph.  When the architecture also registers the short-item program, batches
whose requests span at most ``max_pages`` (four) selected pages run it
instead: one eight-CTA cluster of register-MMA CTAs per work item, two CTAs per
selected page, the partials merged through distributed shared memory, no
workspace.  See ``README.md`` in this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    load_cake_msa_nvfp4_decode_module,
    route_of,
    select_module,
    select_short_module,
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
STATS_PER_SLOT = PAGE_SIZE  # partial max/sum entries per (item, split) slot
PARTIAL_O_PER_SLOT = STATS_PER_SLOT * HEAD_DIM
WORKSPACE_ALIGN = 256
TMA_ALIGN = 16
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
LN2 = math.log(2.0)

# Semantic argument names of the single stage, in the order the generated
# binding consumes them (mirrors the export's argument plan); ``grid`` is
# expanded to ``grid_x/y/z``.
MAIN_KWARGS = (
    "Q",
    "K",
    "K_scale",
    "V",
    "V_scale",
    "O",
    "msa_lse",
    "partial_O",
    "partial_M",
    "partial_D",
    "split_completion",
    "kv_indices",
    "kv_indptr",
    "task_kind",
    "task_request",
    "task_kv_head",
    "total_q",
    "seqlen_q",
    "num_q_heads",
    "num_kv_heads",
    "softmax_scale_log2",
    "output_scale",
    "msa_max_pages",
    "grid",
)

# Semantic argument names of the short-item cluster program, in the order the
# generated binding consumes them.  The four page views are passed as flat
# byte aliases of their storage spans plus their page and head byte strides
# (the program forms 64-bit offsets in place; no TMA descriptors).
SHORT_KWARGS = (
    "Q",
    "K",
    "K_scale",
    "V",
    "V_scale",
    "O",
    "msa_lse",
    "kv_indices",
    "kv_indptr",
    "task_kind",
    "task_request",
    "task_kv_head",
    "total_q",
    "seqlen_q",
    "num_q_heads",
    "num_kv_heads",
    "softmax_scale_log2",
    "output_scale",
    "msa_max_pages",
    "k_page_stride",
    "k_head_stride",
    "ks_page_stride",
    "ks_head_stride",
    "v_page_stride",
    "v_head_stride",
    "vs_page_stride",
    "vs_head_stride",
    "grid",
)
STRIDE_LIMIT = 2**31  # the short program's page / head byte strides are int32


# ---------------------------------------------------------------------------
# Device geometry and the split-KV rule
# ---------------------------------------------------------------------------


def arch_for(device: torch.device) -> Optional[str]:
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def generated_program_available(device: torch.device) -> bool:
    """True when this checkout registers a generated program for ``device``."""
    arch = arch_for(device)
    return arch is not None and any(r["arch"] == arch for r in MODULES.values())


def ctas_per_sm(arch: str) -> int:
    """Resident CTAs per SM of the generated program (a compile-time property)."""
    values = {
        int(r["ctas_per_sm"])
        for r in MODULES.values()
        if r["arch"] == arch and route_of(r) == "swap_tsk"
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
        raise ValueError("NVFP4 MSA decode requires compute capability 10.0 or 10.3")
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return sms * ctas_per_sm(arch)


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


# ---------------------------------------------------------------------------
# Short-item cluster program (work items of at most ``max_pages`` pages)
# ---------------------------------------------------------------------------


def short_program_record(arch: str) -> Optional[dict]:
    """Registry record of the short-item program for ``arch`` (``None`` when absent)."""
    name = select_short_module(arch)
    return None if name is None else MODULES[name]


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
# Workspace (split-KV scratch)
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return (nbytes + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN


def workspace_layout(total_work_items: int, splits: int) -> dict:
    """Byte offsets of the split-KV scratch for ``total_work_items`` items.

    A split factor of one needs no scratch (``total`` is zero).
    """
    items = int(total_work_items)
    splits = int(splits)
    if splits not in SPLIT_FACTORS:
        raise ValueError(f"split factor must be one of {SPLIT_FACTORS}, got {splits}")
    if splits == 1:
        return {"total": 0}
    slots = items * splits
    layout: dict[str, Any] = {}
    offset = 0
    for name, nbytes in (
        ("partial_o", slots * PARTIAL_O_PER_SLOT * 4),
        ("partial_m", slots * STATS_PER_SLOT * 4),
        ("partial_d", slots * STATS_PER_SLOT * 4),
        ("split_completion", items * 4),
    ):
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def msa_nvfp4_decode_workspace_size(
    batch: int, num_kv_heads: int, device: torch.device, *, seqlen_q: int = 1
) -> int:
    """Bytes of caller-owned scratch that cover any split factor for this batch.

    Sized for the largest factor the device could select for ``batch *
    seqlen_q * num_kv_heads`` work items, so one buffer serves every KV length.
    """
    items = int(batch) * int(seqlen_q) * int(num_kv_heads)
    capacity = persistent_cta_capacity(device)
    worst = max(split_factor(items, capacity, max_pages) for max_pages in (TOPK, 1))
    return workspace_layout(items, worst)["total"]


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape):
    offset, nbytes = layout[name]
    numel = 1
    for extent in shape:
        numel *= int(extent)
    if numel * torch.empty((), dtype=dtype).element_size() != nbytes:
        raise AssertionError(f"workspace region {name} does not match its shape")
    return flat[offset : offset + nbytes].view(dtype).view(*shape)


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

    module_name: str
    splits: int
    main_kwargs: dict
    out: torch.Tensor
    lse: torch.Tensor
    entry: Callable[..., Any]
    arguments: tuple
    route: str = (
        "swap_tsk"  # ``swap_tsk`` persistent program or ``short`` cluster program
    )

    def launch(self) -> torch.Tensor:
        # Tensor maps are encoded by the host binding and passed by value.
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.main_kwargs["grid"][0])


def bind_decode_payload(
    arch: str,
    splits: int,
    main_kwargs: dict,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    module_name: Optional[str] = None,
    route: str = "swap_tsk",
) -> MSANvfp4DecodeRunner:
    """Bind the prepared buffers to the generated physical argument order.

    Without ``module_name`` the persistent program of ``arch`` and ``splits``
    is bound; the short-item program passes its record name and
    ``route="short"`` (``splits`` is then one).
    """
    if module_name is None:
        module_name = select_module(arch, splits)
    physical = MODULES[module_name]["main"]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), main_kwargs["grid"], strict=True))
    arguments = tuple(
        grid[name] if kind == "grid" else main_kwargs[name]
        for kind, name in physical["arg_plan"]
    )
    module = load_cake_msa_nvfp4_decode_module(module_name, "main")
    entry = getattr(module, physical["ffi_entry"])
    return MSANvfp4DecodeRunner(
        module_name, int(splits), main_kwargs, out, lse, entry, arguments, route
    )


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
    workspace_buffer: Optional[torch.Tensor] = None,
    seqlen_q: int = 1,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> MSANvfp4DecodeRunner:
    """Validate and bind one NVFP4 paged-KV sparse decode step.

    Every allocation happens here (only the optional ``out`` / ``lse``); the
    returned runner launches with none.  When the device registers the
    short-item program and every request spans at most its page budget
    (``short_route_applies``), the runner binds that program (``route ==
    "short"``, one eight-CTA cluster per work item, no workspace).  Otherwise
    the split factor is decided from the batch geometry and the device's
    resident-CTA capacity; when it is greater than one the FP32 partials and
    completion counters are carved out of ``workspace_buffer`` and the
    counters are zeroed once (the kernel resets them after every merge).
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
    tensors += [t for t in (out, lse, workspace_buffer) if t is not None]
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    arch = arch_for(device)
    if arch is None:
        capability = torch.cuda.get_device_capability(device)
        raise ValueError(
            "NVFP4 MSA decode requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    total_q = batch * int(seqlen_q)
    total_work_items = total_q * num_kv_heads
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    # The global K multiplier is separable from every per-block scale: fold it
    # into the softmax scale; the global V multiplier is applied in the epilogue.
    softmax_scale_log2 = scale * float(k_global_scale) / LN2
    if out is None:
        out = torch.empty(
            (total_q, num_q_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    if lse is None:
        lse = torch.empty((total_q, num_q_heads), dtype=torch.float32, device=device)

    if short_route_applies(arch, max_pages):
        short_name = select_short_module(arch)
        assert short_name is not None
        record = MODULES[short_name]
        k_flat, k_ps, k_hs = _flat_page_operand("k", k, DATA_DIM)
        ks_flat, ks_ps, ks_hs = _flat_page_operand("k_scale", k_scale, SCALE_DIM)
        v_flat, v_ps, v_hs = _flat_page_operand("v", v, DATA_DIM)
        vs_flat, vs_ps, vs_hs = _flat_page_operand("v_scale", v_scale, SCALE_DIM)
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        short_kwargs = dict(
            Q=q,
            K=k_flat,
            K_scale=ks_flat,
            V=v_flat,
            V_scale=vs_flat,
            O=out,
            msa_lse=lse,
            kv_indices=page_table,
            kv_indptr=seqused_k,
            task_kind=q2k_indices,
            task_request=seqused_k,
            task_kv_head=seqused_k,
            total_q=total_q,
            seqlen_q=int(seqlen_q),
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            softmax_scale_log2=float(softmax_scale_log2),
            output_scale=float(v_global_scale),
            msa_max_pages=max_pages,
            k_page_stride=k_ps,
            k_head_stride=k_hs,
            ks_page_stride=ks_ps,
            ks_head_stride=ks_hs,
            v_page_stride=v_ps,
            v_head_stride=v_hs,
            vs_page_stride=vs_ps,
            vs_head_stride=vs_hs,
            grid=short_grid(total_work_items, num_sms, record),
        )
        assert tuple(short_kwargs) == SHORT_KWARGS
        return bind_decode_payload(
            arch, 1, short_kwargs, out, lse, module_name=short_name, route="short"
        )

    capacity = persistent_cta_capacity(device)
    splits = split_factor(total_work_items, capacity, max_pages)
    grid = (persistent_grid(total_work_items, splits, capacity), 1, 1)

    if splits > 1:
        layout = workspace_layout(total_work_items, splits)
        if workspace_buffer is None:
            raise ValueError(
                f"this batch splits each item across {splits} CTAs and needs a workspace_buffer of "
                f"{layout['total']} bytes (msa_nvfp4_decode_workspace_size)"
            )
        flat = workspace_buffer.view(-1).view(torch.uint8)
        if flat.numel() < layout["total"]:
            raise ValueError(
                f"workspace_buffer needs {layout['total']} bytes for {total_work_items} items "
                f"split {splits} ways, got {flat.numel()}"
            )
        slots = total_work_items * splits
        partial_o = _carve(
            flat, layout, "partial_o", torch.float32, (slots * STATS_PER_SLOT, HEAD_DIM)
        )
        partial_m = _carve(
            flat, layout, "partial_m", torch.float32, (slots * STATS_PER_SLOT,)
        )
        partial_d = _carve(
            flat, layout, "partial_d", torch.float32, (slots * STATS_PER_SLOT,)
        )
        split_completion = _carve(
            flat, layout, "split_completion", torch.int32, (total_work_items,)
        )
        # Completion counters start at zero once; the merge CTA resets them.
        split_completion.zero_()
    else:
        # Unsplit launches never touch the partial buffers: bind valid dummies.
        partial_o = partial_m = partial_d = lse.view(-1)
        split_completion = q2k_indices.view(-1)

    main_kwargs = dict(
        Q=q,
        K=k.view(torch.uint8),
        K_scale=k_scale.view(torch.uint8),
        V=v.view(torch.uint8),
        V_scale=v_scale.view(torch.uint8),
        O=out,
        msa_lse=lse,
        partial_O=partial_o,
        partial_M=partial_m,
        partial_D=partial_d,
        split_completion=split_completion,
        kv_indices=page_table,
        # Paged decode derives lengths from seqused_k and pages from page_table;
        # the indptr / request-offset carriers are ABI-only and never launch
        # metadata arithmetic on the hot path.
        kv_indptr=seqused_k,
        task_kind=q2k_indices,
        task_request=seqused_k,
        task_kv_head=seqused_k,
        total_q=total_q,
        seqlen_q=int(seqlen_q),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        softmax_scale_log2=float(softmax_scale_log2),
        output_scale=float(v_global_scale),
        msa_max_pages=max_pages,
        grid=grid,
    )
    assert tuple(main_kwargs) == MAIN_KWARGS
    return bind_decode_payload(arch, splits, main_kwargs, out, lse)
