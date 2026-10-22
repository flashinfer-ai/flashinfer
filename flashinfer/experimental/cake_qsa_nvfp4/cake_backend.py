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

Host side of the Cake QSA sparse paged GQA decode over an NVFP4 KV cache
(SM80 through SM90): the argument checks of ``qsa_sparse_paged_attention``,
the split rule, the form rule, the workspace rule and the launch sequence.

A call attends ``rows`` decode rows (one query token each, ``num_qo_heads``
heads of 256 dims) to the ``selection_width`` tokens each row selects from a
paged E2M1 cache with E4M3 block scales, and writes ``bf16(attention) *
sigmoid(output_gate)``.  The selection is cut into tiles of 64 tokens and the
tiles into ``splits`` splits; one decode CTA serves one (row, KV head, split).

Split rule (:func:`num_splits`): ``tiles = ceil(selection_width / 64)``;
``splits = max(1, min(tiles, 2 * sm_count // (rows * num_kv_heads)))`` (two
resident CTAs per SM, the decode's launch bound); then the measured table
:data:`SPLIT_OVERRIDES` of the device's SM count replaces the formula where it
names ``(sm_count, rows)``; then ``splits <= min(tiles, MAX_SPLITS)``.  The
rule reads the SM count alone, cached per device.

Form rule (:func:`select_form`): with one split the plain decode writes the
output itself.  Above one split, on compute capability 9.0 the call takes the
*fused* form -- the cooperative build of the decode, whose CTAs merge the
splits in the same launch through a counter per (row, KV head) -- when its
grid ``rows * num_kv_heads * splits`` does not exceed the co-residency the
decode's launch bound guarantees on the device (:func:`fused_capacity`: two
CTAs per SM times the SM count, 264 on an H100; a cooperative launch above the
device's co-residency is refused by the driver); else,
and always on compute capability 8.x, the *split* form: the plain decode, then
the separate merge.

Workspace rule: the partials (``partial_out`` FP32 ``[splits, rows,
num_qo_heads, 256]``, ``partial_lse`` FP32 ``[splits, rows, num_qo_heads]``)
and the fused form's counters (uint32 ``[rows * num_kv_heads]``) come from the
caller's ``workspace_buffer`` when one is given (uint8 CUDA, at least
:func:`qsa_workspace_bytes` bytes; carved, never copied; the counters are
zeroed once, the first time a buffer is seen for a ``rows * num_kv_heads``,
and reset themselves after every launch), so a call with a workspace allocates
nothing and is CUDA-graph capturable with a replay-stable layout.  Without a
workspace the partials and counters are allocated per call, as the reference
implementation does; that path is outside the zero-allocation guarantee.

Nothing here reads tensor contents: every dispatch is on host scalars and
tensor metadata.  The kernels run on the current torch stream of the device.
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch

from . import cake_jit

HEAD_DIM = 256
TILE_TOKENS = cake_jit.TILE_TOKENS
MAX_GROUP_SIZE = 16
MAX_SPLITS = cake_jit.MAX_SPLITS
# Resident decode CTAs per SM the split rule and the fused form's capacity count on (the kernel's launch bound).
CTAS_PER_SM = cake_jit.DECODE_CTAS_PER_SM
# Every workspace slot starts on a 256-byte boundary.
ALIGN = 256
_LOG2E = math.log2(math.e)
# The architecture whose decode may take the fused form (the H100 measured it faster at every multi-split row of
# the owner's shape list; the A100 measured it slower at rows 1-4).
_FUSED_ARCH = "sm_90a"

# The measured split table: ``(sm_count, rows) -> splits`` where the fastest split count of the acceptance fixture
# (2 KV heads, a selection of 2051 tokens = 33 tiles) differs from the formula.  Keyed by the SM count, never by a
# product name.  H100 (132 SMs): rows 4 take 16 splits and rows 8 take 8; A100 (108 SMs): rows 64 take 4 and rows
# 128 take 2 (both above the formula's capacity; a plain launch runs in more than one wave).
SPLIT_OVERRIDES: dict[tuple[int, int], int] = {
    (132, 4): 16,
    (132, 8): 8,
    (108, 64): 4,
    (108, 128): 2,
}


def num_tiles(selection_width: int) -> int:
    """``ceil(selection_width / 64)``: the tiles of one row's selection."""
    return -(-int(selection_width) // TILE_TOKENS)


def num_splits(
    num_rows: int, num_kv_heads: int, selection_width: int, sm_count: int
) -> int:
    """The split count of a call on a device of ``sm_count`` SMs (module docstring, "Split rule")."""
    if num_rows < 1 or num_kv_heads < 1 or sm_count < 1:
        raise ValueError(
            f"num_rows={num_rows}, num_kv_heads={num_kv_heads}, sm_count={sm_count} must all be at least 1"
        )
    tiles = num_tiles(selection_width)
    formula = max(
        1, min(tiles, (CTAS_PER_SM * int(sm_count)) // (int(num_rows) * int(num_kv_heads)))
    )
    chosen = SPLIT_OVERRIDES.get((int(sm_count), int(num_rows)), formula)
    return max(1, min(tiles, MAX_SPLITS, chosen))


def is_supported(device: torch.device | int | None = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) is an exported target."""
    if not torch.cuda.is_available():
        return False
    index = _device_index(device)
    return (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is not None
    )


@functools.cache
def fused_capacity(device_index: int) -> int:
    """The largest decode grid (``rows * num_kv_heads * splits`` CTAs) the fused form serves on the device: the
    co-residency the decode's launch bound guarantees, ``CTAS_PER_SM`` CTAs per SM times the SM count (264 on an
    H100); 0 on an architecture whose calls always take the split form."""
    if cake_jit.device_arch(device_index) != _FUSED_ARCH:
        return 0
    return CTAS_PER_SM * cake_jit.sm_count(device_index)


def select_form(num_rows: int, num_kv_heads: int, splits: int, device_index: int) -> str:
    """``"fused"`` or ``"split"`` (module docstring, "Form rule")."""
    if int(splits) > 1 and int(num_rows) * int(num_kv_heads) * int(splits) <= fused_capacity(
        device_index
    ):
        return "fused"
    return "split"


def _round_up(n: int, m: int) -> int:
    return -(-int(n) // m) * m


def _layout(
    num_rows: int, num_qo_heads: int, num_kv_heads: int, splits: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """``(offset, bytes)`` of the counters, the partial outputs and the partial LSEs, in that order, each on a
    256-byte boundary; the counters come first so that their slot is the same for every split count."""
    counters = (0, 4 * int(num_rows) * int(num_kv_heads))
    out_offset = _round_up(counters[0] + counters[1], ALIGN)
    partial_out = (out_offset, 4 * int(splits) * int(num_rows) * int(num_qo_heads) * HEAD_DIM)
    lse_offset = _round_up(partial_out[0] + partial_out[1], ALIGN)
    partial_lse = (lse_offset, 4 * int(splits) * int(num_rows) * int(num_qo_heads))
    return counters, partial_out, partial_lse


def qsa_workspace_bytes(
    num_rows: int, num_qo_heads: int, num_kv_heads: int, splits: int
) -> int:
    """Bytes a ``workspace_buffer`` needs for a call of these sizes (``splits`` from :func:`num_splits`; a buffer
    sized for the largest split count a caller launches, ``MAX_SPLITS`` at most, serves every smaller one)."""
    _counters, _partial_out, partial_lse = _layout(num_rows, num_qo_heads, num_kv_heads, splits)
    return _round_up(partial_lse[0] + partial_lse[1], ALIGN)


# ``(data_ptr, rows * num_kv_heads)`` of every workspace whose counters were zeroed once (module docstring).
_zeroed_counters: set[tuple[int, int]] = set()


def _carve(
    workspace: torch.Tensor,
    num_rows: int,
    num_qo_heads: int,
    num_kv_heads: int,
    splits: int,
    *,
    counters: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The partial outputs, the partial LSEs and (when ``counters``) the zeroed group counters as views of the
    caller's ``workspace`` (``qsa_workspace_bytes`` bytes at least); nothing is allocated or copied."""
    _check(
        workspace.device.type == "cuda"
        and workspace.dtype == torch.uint8
        and workspace.is_contiguous(),
        "workspace_buffer must be a contiguous uint8 CUDA tensor",
    )
    raw = workspace.reshape(-1)
    need = qsa_workspace_bytes(num_rows, num_qo_heads, num_kv_heads, splits)
    _check(
        raw.numel() >= need,
        f"workspace_buffer needs at least {need} bytes for this call (qsa_workspace_bytes), got {raw.numel()}",
    )
    _check(raw.data_ptr() % ALIGN == 0, f"workspace_buffer must be {ALIGN}-byte aligned")
    slot_counters, slot_out, slot_lse = _layout(num_rows, num_qo_heads, num_kv_heads, splits)
    partial_out = (
        raw[slot_out[0] : slot_out[0] + slot_out[1]]
        .view(torch.float32)
        .view(splits, num_rows, num_qo_heads, HEAD_DIM)
    )
    partial_lse = (
        raw[slot_lse[0] : slot_lse[0] + slot_lse[1]]
        .view(torch.float32)
        .view(splits, num_rows, num_qo_heads)
    )
    if not counters:
        return partial_out, partial_lse, None
    group_counters = (
        raw[slot_counters[0] : slot_counters[0] + slot_counters[1]]
        .view(torch.uint32)
        .view(num_rows * num_kv_heads)
    )
    key = (raw.data_ptr(), num_rows * num_kv_heads)
    if key not in _zeroed_counters:
        group_counters.zero_()
        _zeroed_counters.add(key)
    return partial_out, partial_lse, group_counters


def reset_workspace(workspace: torch.Tensor) -> None:
    """Forget that ``workspace`` was seen: the next call carving it zeroes its counters again (for a buffer whose
    bytes another use overwrote)."""
    ptr = workspace.reshape(-1).data_ptr()
    for key in [key for key in _zeroed_counters if key[0] == ptr]:
        _zeroed_counters.discard(key)


def _device_index(device: torch.device | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_aligned(tensor: torch.Tensor, name: str, num_bytes: int) -> None:
    # The kernel moves rows in num_bytes vectors: base and every outer stride must be aligned.
    element_size = tensor.element_size()
    _check(
        tensor.stride(-1) == 1
        and tensor.data_ptr() % num_bytes == 0
        and all(s * element_size % num_bytes == 0 for s in tensor.stride()[:-1]),
        f"{name} needs a contiguous last dim and {num_bytes}-byte aligned base and strides",
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: Optional[torch.Tensor] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    *,
    output_gate: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    workspace_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gated sparse paged GQA attention over an NVFP4 KV cache (the contract of
    ``flashinfer.sparse.qsa_sparse_paged_attention``), through the Cake kernels.

    The arguments are the public entry's; ``workspace_buffer`` is the caller-owned scratch of the module docstring
    ("Workspace rule"), ``None`` for per-call allocation.
    """
    del (
        use_prefill_config
    )  # kept for the signature; the split rule does not depend on it
    capability = tuple(torch.cuda.get_device_capability(q.device))
    if cake_jit.arch_for_capability(capability) is None:
        raise RuntimeError(
            f"QSA NVFP4 attention supports SM80 through SM90, got SM{capability[0]}{capability[1]}"
        )
    _check(q.ndim == 3 and k_cache.ndim == 4, "q must be 3D and k_cache 4D")
    num_rows, num_qo_heads, head_dim = q.shape
    num_blocks, page_size, num_kv_heads, packed_dim = k_cache.shape
    _check(q.dtype == torch.bfloat16, "q must be bfloat16")
    _check(head_dim == HEAD_DIM, f"head_dim must be {HEAD_DIM}")
    _check(packed_dim * 2 == head_dim, "k_cache must hold head_dim / 2 packed bytes")
    _check(
        k_cache.dtype == torch.uint8
        and v_cache.shape == k_cache.shape
        and v_cache.dtype == k_cache.dtype
        and v_cache.stride() == k_cache.stride(),
        "k_cache and v_cache must be uint8 views with the same shape and strides",
    )
    sf_shape = (num_blocks, page_size, num_kv_heads, head_dim // 16)
    _check(
        k_sf.dtype == torch.float8_e4m3fn
        and tuple(k_sf.shape) == sf_shape
        and v_sf.shape == k_sf.shape
        and v_sf.dtype == k_sf.dtype
        and v_sf.stride() == k_sf.stride(),
        f"k_sf and v_sf must be float8_e4m3fn views of shape {sf_shape} with the same strides",
    )
    _check(num_blocks > 0 and page_size > 0, "k_cache must not be empty")
    _check(
        num_qo_heads % num_kv_heads == 0
        and num_qo_heads // num_kv_heads <= MAX_GROUP_SIZE,
        f"num_qo_heads must be a multiple of num_kv_heads, at most {MAX_GROUP_SIZE}x",
    )
    _check(
        logical_indices.ndim == 2
        and logical_indices.shape[0] == num_rows
        and logical_indices.shape[1] >= 2
        and logical_indices.dtype == torch.int32
        and logical_indices.stride(1) == 1,
        "logical_indices must be int32 [rows, selection_width + 1] with a contiguous last dim",
    )
    _check(
        block_table.ndim == 2
        and block_table.numel() > 0
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1,
        "block_table must be a non-empty int32 2D tensor with a contiguous last dim",
    )
    _check(
        token_to_req.shape == (num_rows,)
        and token_to_req.dtype == torch.int32
        and token_to_req.is_contiguous(),
        "token_to_req must be contiguous int32 [rows]",
    )
    _check(
        k_scale is not None and v_scale is not None, "k_scale and v_scale are required"
    )
    _check(
        output_gate.shape == q.shape
        and output_gate.dtype == torch.bfloat16
        and output_gate.is_contiguous(),
        "output_gate must be contiguous bfloat16 with the shape of q",
    )
    if out is None:
        out = torch.empty_like(q)
    _check(
        out.shape == q.shape and out.dtype == q.dtype,
        "out must match q in shape and dtype",
    )
    tensors = (q, k_cache, v_cache, k_sf, v_sf, logical_indices, block_table)
    _check(
        all(x.device == q.device for x in (*tensors, token_to_req, output_gate, out)),
        "all tensors must be on the same device",
    )
    _check_aligned(q, "q", 8)
    for name, tensor in (
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("k_sf", k_sf),
        ("v_sf", v_sf),
        ("output_gate", output_gate),
        ("out", out),
    ):
        _check_aligned(tensor, name, 16)
    if num_rows == 0:
        return out

    device_index = _device_index(q.device)
    selection_width = logical_indices.shape[1] - 1
    splits = num_splits(
        num_rows, num_kv_heads, selection_width, cake_jit.sm_count(device_index)
    )
    form = select_form(num_rows, num_kv_heads, splits, device_index)
    fused = form == "fused"
    if splits == 1:
        partial_out = partial_lse = merge_counters = None
    elif workspace_buffer is not None:
        partial_out, partial_lse, merge_counters = _carve(
            workspace_buffer, num_rows, num_qo_heads, num_kv_heads, splits, counters=fused
        )
    else:
        partial_out = q.new_empty(
            (splits, num_rows, num_qo_heads, head_dim), dtype=torch.float32
        )
        partial_lse = q.new_empty((splits, num_rows, num_qo_heads), dtype=torch.float32)
        merge_counters = (
            torch.zeros(num_rows * num_kv_heads, dtype=torch.uint32, device=q.device)
            if fused
            else None
        )
    values = {
        "q": q,
        "k_data": k_cache,
        "v_data": v_cache,
        "k_sf": k_sf.view(torch.uint8),
        "v_sf": v_sf.view(torch.uint8),
        "indices": logical_indices,
        "block_table": block_table,
        "token_to_req": token_to_req,
        "gate": output_gate,
        "out": out,
        "partial_out": partial_out,
        "partial_lse": partial_lse,
        "merge_counters": merge_counters,
        "q_stride_row": q.stride(0),
        "q_stride_head": q.stride(1),
        "data_stride_block": k_cache.stride(0),
        "data_stride_token": k_cache.stride(1),
        "data_stride_head": k_cache.stride(2),
        "sf_stride_block": k_sf.stride(0),
        "sf_stride_token": k_sf.stride(1),
        "sf_stride_head": k_sf.stride(2),
        "indices_stride_row": logical_indices.stride(0),
        "table_stride_req": block_table.stride(0),
        "out_stride_row": out.stride(0),
        "out_stride_head": out.stride(1),
        "score_scale": float(k_scale) * _LOG2E / math.sqrt(head_dim),
        "output_scale": float(v_scale),
        "num_rows": num_rows,
        "num_requests": block_table.shape[0],
        "num_blocks": num_blocks,
        "page_div": page_size,
        "table_width": block_table.shape[1],
        "topk": selection_width,
        "group_size": num_qo_heads // num_kv_heads,
        "num_qo_heads": num_qo_heads,
        "num_tiles": num_tiles(selection_width),
        "num_splits": splits,
    }
    for name in cake_jit.select_records(form=form, splits=splits):
        grid = cake_jit.launch_grid(
            name,
            rows=num_rows,
            kv_heads=num_kv_heads,
            qo_heads=num_qo_heads,
            splits=splits,
        )
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)
    return out


__all__ = [
    "ALIGN",
    "CTAS_PER_SM",
    "HEAD_DIM",
    "MAX_GROUP_SIZE",
    "MAX_SPLITS",
    "SPLIT_OVERRIDES",
    "TILE_TOKENS",
    "fused_capacity",
    "is_supported",
    "num_splits",
    "num_tiles",
    "qsa_sparse_paged_attention",
    "qsa_workspace_bytes",
    "reset_workspace",
    "select_form",
]
