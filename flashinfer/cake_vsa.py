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
"""

from __future__ import annotations

import functools
import math
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api
from .jit.cake_vsa import arch_for_capability, load_cake_vsa_module
from .trace.templates.attention import (
    block_sparse_attention_run_trace,
    cake_vsa_plan_trace,
)

_MAX_COMPACT_BLOCKS = 64
_MAX_DIRECT_TOPK = 32
_MAX_LONGSEQ_BLOCKS = 192
_BLK64_ORDINARY_MAX_AVERAGE_SELECTED_BLOCKS = 24
_BLK64_WS_PROFILE = "blk64_persistent_ws_m64n256"
_BLK64_BALANCED_PROFILE = "blk64_balanced"
# The balanced block-64 profile reorders the persistent tile stream on device;
# it needs at least this many full waves (MB * heads >= waves * SM count).
_BLK64_BALANCED_MIN_WAVES = 2
# Per-arch upper bound on the selected blocks per row for the balanced profile.
# On sm_100a the queue-ordered tile stream loses to the static stripe once rows
# select more than 64 blocks (k = 96..192 read 0.93..0.99 of the weight-stationary
# profile); sm_103a gains across the measured band (k = 28..192: 1.06..1.33).
_BLK64_BALANCED_MAX_SELECTED_BLOCKS = {"sm_100a": 64}
_FP16_DIRECT_Q_TILE = 256


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(f"Cake VSA requires a CUDA device, got {device}")
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _arch_for_device_index(index: int) -> str:
    return arch_for_capability(torch.cuda.get_device_capability(index))


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _arch_for_device(device: torch.device) -> str:
    return _arch_for_device_index(_device_index(device))


def _module(profile: str, device: torch.device):
    return load_cake_vsa_module(profile, _arch_for_device(device))


def _dense_mask(
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    block_mask: Optional[torch.Tensor],
    *,
    mb: int,
    nb: int,
    num_qo_heads: int,
    num_kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    if block_mask is not None:
        source = block_mask.to(device=device, dtype=torch.bool).contiguous()
        if tuple(source.shape) == (num_kv_heads, mb, nb):
            source = source.repeat_interleave(
                num_qo_heads // num_kv_heads, dim=0
            ).contiguous()
        if tuple(source.shape) != (num_qo_heads, mb, nb):
            raise ValueError(
                "block_mask must have shape [num_qo_heads, MB, NB] or "
                "[num_kv_heads, MB, NB]"
            )
        return source
    if indptr is None or indices is None:
        raise ValueError("Cake VSA requires block_mask or BSR indptr/indices")
    row_offsets = indptr.to(device="cpu", dtype=torch.int64).tolist()
    columns = indices.to(device="cpu", dtype=torch.int64).tolist()
    if len(row_offsets) != mb + 1:
        raise ValueError("indptr must have MB + 1 entries")
    shared = torch.zeros((mb, nb), dtype=torch.bool, device=device)
    for row in range(mb):
        selected = columns[row_offsets[row] : row_offsets[row + 1]]
        if not selected:
            raise ValueError("every Cake VSA block row must select at least one block")
        if min(selected) < 0 or max(selected) >= nb:
            raise ValueError("BSR column index is out of range")
        shared[row, torch.tensor(selected, device=device)] = True
    return shared.unsqueeze(0).expand(num_qo_heads, -1, -1).contiguous()


def _shared_bsr(
    dense: torch.Tensor,
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    *,
    trust_bsr: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.equal(dense, dense[:1].expand_as(dense)):
        raise ValueError(
            "Cake blk128 routes require one shared BSR pattern across heads"
        )
    shared = dense[0]
    counts = shared.sum(dim=1, dtype=torch.int32)
    ptr = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int32, device=dense.device),
            counts.cumsum(0, dtype=torch.int32),
        ]
    )
    if trust_bsr and indptr is not None and indices is not None:
        raw_ptr = indptr.to(device=dense.device, dtype=torch.int32).contiguous()
        raw_cols = indices.to(device=dense.device, dtype=torch.int32).contiguous()
        if raw_cols.numel() == int(ptr[-1].item()) and torch.equal(raw_ptr, ptr):
            return raw_ptr, raw_cols

        # The ultrasparse kernel consumes six columns per row with a fixed
        # stride rather than consulting indptr. Canonicalize duplicate or
        # otherwise non-packed BSR rows before making that metadata launchable.
    cols = shared.nonzero(as_tuple=False)[:, 1].to(torch.int32).contiguous()
    return ptr, cols


def _blk64_dense_from_q2k(
    q2k_indices: torch.Tensor, q2k_num: torch.Tensor, *, nb: int
) -> torch.Tensor:
    """Dense ``[heads, MB, NB]`` block mask of validated direct block-64 selections."""
    heads, mb, topk = q2k_indices.shape
    active = (
        torch.arange(topk, device=q2k_indices.device) < q2k_num.unsqueeze(-1)
    ).view(heads * mb, topk)
    rows = torch.arange(heads * mb, device=q2k_indices.device).unsqueeze(-1)
    dense = torch.zeros((heads * mb, nb), dtype=torch.bool, device=q2k_indices.device)
    dense[
        rows.expand(heads * mb, topk)[active],
        q2k_indices.view(heads * mb, topk)[active].to(torch.int64),
    ] = True
    return dense.view(heads, mb, nb)


def _fp16_direct_metadata(
    dense: torch.Tensor,
    row_counts: torch.Tensor,
    *,
    M: int,
    N: int,
    R: int,
    mb: int,
    num_qo_heads: int,
    num_kv_heads: int,
    device: torch.device,
) -> dict[str, Any]:
    """Direct-route metadata of the FP16 GQA kernel, built once at plan time.

    The kernel walks per-token ``q2k_indices`` of one KV-head group each; the
    block mask is identical within a group (validated by the caller), so the
    first head of every group provides the group's selections.
    """
    group_size = num_qo_heads // num_kv_heads
    masks = dense[::group_size]
    counts = row_counts[::group_size]
    topk = int(counts.max().item())
    if topk > _MAX_DIRECT_TOPK or not bool(torch.all(counts == topk).item()):
        raise ValueError("Cake FP16 direct route requires fixed top-k <= 32")
    per_block = masks.nonzero(as_tuple=False)[:, 2].view(num_kv_heads, mb, topk)
    q2k_indices = (
        per_block.repeat_interleave(R, dim=1)
        .to(device=device, dtype=torch.int32)
        .contiguous()
    )
    return {
        "q2k_indices": q2k_indices,
        "cu_seqlens_q": torch.tensor([0, M], dtype=torch.int32, device=device),
        "cu_seqlens_k": torch.tensor([0, N], dtype=torch.int32, device=device),
        "q_offsets": torch.zeros((1,), dtype=torch.int32, device=device),
        "kv_lens": torch.tensor([N], dtype=torch.int32, device=device),
        "page_table": torch.zeros((1,), dtype=torch.int32, device=device),
        "scale_dummy": torch.empty((1, 1, 128, 8), dtype=torch.uint8, device=device),
        "topk": topk,
    }


@flashinfer_api(trace=cake_vsa_plan_trace)
def plan_cake_vsa(
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    block_mask: Optional[torch.Tensor],
    kv_block_lens: Optional[torch.Tensor],
    q2k_indices: Optional[torch.Tensor],
    q2k_num: Optional[torch.Tensor],
    *,
    M: int,
    N: int,
    R: int,
    C: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_data_type: torch.dtype,
    sm_scale: Optional[float],
    device: torch.device,
) -> dict[str, Any]:
    """Create stable metadata and workspaces for the source-level backend.

    Parameters
    ----------
    indptr : Optional[torch.Tensor]
        CSR-style row pointers for a block pattern shared by all heads, with
        shape ``(M // R + 1,)``. Required with ``indices`` when neither
        ``block_mask`` nor ``q2k_indices`` is supplied.
    indices : Optional[torch.Tensor]
        Column indices corresponding to ``indptr``. Duplicate or non-packed
        rows are canonicalized before a fixed-stride source kernel can use
        them.
    block_mask : Optional[torch.Tensor]
        Boolean block mask with shape ``(num_qo_heads, M // R, N // C)`` or
        ``(num_kv_heads, M // R, N // C)``.
    kv_block_lens : Optional[torch.Tensor]
        Valid-token count for each KV block, with shape ``(N // C,)``. This is
        supported only for 64-token blocks.
    q2k_indices : Optional[torch.Tensor]
        Direct block-64 selections as contiguous int32 metadata with shape
        ``(num_qo_heads, M // R, topk)``.
    q2k_num : Optional[torch.Tensor]
        Number of active ``q2k_indices`` entries per row, as contiguous int32
        metadata with shape ``(num_qo_heads, M // R)``.
    M : int
        Query sequence length.
    N : int
        Key/value sequence length.
    R : int
        Query block size. Cake supports 64 and 128.
    C : int
        Key/value block size, which must equal ``R``.
    num_qo_heads : int
        Number of query/output heads.
    num_kv_heads : int
        Number of key/value heads.
    head_dim : int
        Per-head dimension. Cake supports 64, 96, and 128.
    q_data_type : torch.dtype
        Planned Q/K/V dtype, either ``torch.float16`` or ``torch.bfloat16``.
    sm_scale : Optional[float]
        Softmax scale. ``None`` selects ``1 / sqrt(head_dim)``.
    device : torch.device
        SM100 or SM103 CUDA device that will execute the plan.

    Returns
    -------
    dict[str, Any]
        Validated metadata and reusable workspaces consumed by
        :func:`run_cake_vsa`. Every device-side reduction and every
        route-specific metadata tensor (including the FP16 direct route's
        per-token selections) is built here, so :func:`run_cake_vsa` only
        validates tensor metadata and launches.
    """

    _arch_for_device(device)
    if R != C or R not in (64, 128):
        raise ValueError("Cake VSA supports square 64- or 128-token blocks")
    if kv_block_lens is not None and R != 64:
        raise ValueError("kv_block_lens is supported only by Cake blk64")
    if q2k_indices is not None and R != 64:
        raise ValueError("q2k_indices is supported only by Cake blk64")
    if q2k_num is not None and q2k_indices is None:
        raise ValueError("q2k_num requires q2k_indices")
    if M % R or N % C:
        raise ValueError("M and N must be divisible by the Cake VSA block size")
    if head_dim not in (64, 96, 128):
        raise ValueError("Cake VSA supports head dimensions 64, 96, and 128")
    if q_data_type not in (torch.float16, torch.bfloat16):
        raise ValueError("Cake VSA supports float16 and bfloat16 inputs")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if R == 64 and (
        head_dim != 128 or q_data_type != torch.bfloat16 or num_qo_heads != num_kv_heads
    ):
        raise ValueError("Cake blk64 supports native-head BF16 D128 only")
    if head_dim in (64, 96) and (
        q_data_type != torch.bfloat16 or num_qo_heads != num_kv_heads
    ):
        raise ValueError("Cake D64/D96 routes support native-head BF16 only")
    if (
        q_data_type == torch.bfloat16
        and num_qo_heads != num_kv_heads
        and (num_qo_heads != 8 or num_qo_heads // num_kv_heads not in (2, 4, 8))
    ):
        raise ValueError("Cake BF16 GQA routes require Hq=8 and group size 2, 4, or 8")
    mb, nb = M // R, N // C
    dense = None
    if q2k_indices is not None:
        if block_mask is not None or indptr is not None or indices is not None:
            raise ValueError(
                "q2k_indices is mutually exclusive with block_mask and BSR metadata"
            )
        if (
            q2k_indices.dtype != torch.int32
            or q2k_indices.device != device
            or not q2k_indices.is_contiguous()
            or q2k_indices.ndim != 3
            or tuple(q2k_indices.shape[:2]) != (num_qo_heads, mb)
        ):
            raise ValueError(
                "q2k_indices must be contiguous int32 [num_qo_heads, MB, topk] "
                "on the wrapper device"
            )
        max_selected_blocks = int(q2k_indices.shape[2])
        if max_selected_blocks <= 0 or max_selected_blocks > nb:
            raise ValueError("q2k_indices topk must be in [1, NB]")
        uniform_selected_blocks = q2k_num is None
        if q2k_num is None:
            q2k_num = torch.full(
                (num_qo_heads, mb),
                max_selected_blocks,
                dtype=torch.int32,
                device=device,
            )
        elif (
            q2k_num.dtype != torch.int32
            or q2k_num.device != device
            or not q2k_num.is_contiguous()
            or tuple(q2k_num.shape) != (num_qo_heads, mb)
        ):
            raise ValueError(
                "q2k_num must be contiguous int32 [num_qo_heads, MB] on the "
                "wrapper device"
            )
        row_counts = q2k_num
        if bool(torch.any((q2k_num < 1) | (q2k_num > max_selected_blocks)).item()):
            raise ValueError("q2k_num entries must be in [1, topk]")
        slots = torch.arange(max_selected_blocks, device=device)
        active_indices = q2k_indices[slots < q2k_num.unsqueeze(-1)]
        if bool(torch.any((active_indices < 0) | (active_indices >= nb)).item()):
            raise ValueError("active q2k_indices entries must be in [0, NB)")
    else:
        dense = _dense_mask(
            indptr,
            indices,
            block_mask,
            mb=mb,
            nb=nb,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            device=device,
        )
        row_counts = dense.sum(dim=-1, dtype=torch.int32)
        min_selected_blocks = int(row_counts.min().item())
        max_selected_blocks = int(row_counts.max().item())
        uniform_selected_blocks = bool(
            torch.all(row_counts == max_selected_blocks).item()
        )
        if min_selected_blocks <= 0:
            raise ValueError("every Cake VSA block row must select at least one block")
        if num_qo_heads > num_kv_heads:
            group_size = num_qo_heads // num_kv_heads
            grouped = dense.view(num_kv_heads, group_size, mb, nb)
            if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
                raise ValueError(
                    "Cake GQA masks must be identical within each KV-head group"
                )
        if R == 64:
            q2k_num = row_counts.contiguous()
            q2k_indices = (
                torch.argsort(
                    dense.to(torch.int8),
                    dim=-1,
                    descending=True,
                    stable=True,
                )
                .to(torch.int32)
                .contiguous()
            )

    if head_dim in (64, 96) and max_selected_blocks > _MAX_COMPACT_BLOCKS:
        raise ValueError(
            "Cake D64/D96 routes support at most 64 selected blocks per row"
        )

    planned_kv_block_lens = None
    blk64_profile = None
    blk64_selected_blocks_total = None
    if R == 64:
        # Planning already validates the device-resident row counts. Cache the
        # measured schedule crossover here so repeated run() calls do not add a
        # reduction or host synchronization. Use the actual mixed row counts,
        # not the padded q2k_indices capacity or a min/max surrogate.
        blk64_selected_blocks_total = int(row_counts.sum().item())
        if kv_block_lens is None:
            planned_kv_block_lens = torch.full(
                (nb,), C, dtype=torch.int32, device=device
            )
        else:
            if tuple(kv_block_lens.shape) != (nb,):
                raise ValueError("kv_block_lens must have shape [NB]")
            planned_kv_block_lens = kv_block_lens.to(
                device=device, dtype=torch.int32
            ).contiguous()
            if bool(
                torch.any(
                    (planned_kv_block_lens < 1) | (planned_kv_block_lens > C)
                ).item()
            ):
                raise ValueError("kv_block_lens entries must be in [1, C]")
        slots = torch.arange(q2k_indices.shape[2], device=device)
        active_indices = q2k_indices[slots < q2k_num.unsqueeze(-1)]
        full_kv_groups = bool(torch.all(row_counts.remainder(4) == 0).item())
        if full_kv_groups:
            full_kv_groups = bool(
                torch.all(
                    planned_kv_block_lens[active_indices.to(torch.int64)] == C
                ).item()
            )
        blk64_profile = (
            _BLK64_WS_PROFILE
            if full_kv_groups
            and blk64_selected_blocks_total
            > _BLK64_ORDINARY_MAX_AVERAGE_SELECTED_BLOCKS * row_counts.numel()
            else "blk64_persistent"
        )
    shared_indptr = shared_indices = None
    balanced_max_blocks = _BLK64_BALANCED_MAX_SELECTED_BLOCKS.get(
        _arch_for_device(device)
    )
    if (
        blk64_profile == _BLK64_WS_PROFILE
        and mb * num_qo_heads
        >= _BLK64_BALANCED_MIN_WAVES * _sm_count(_device_index(device))
        and (
            balanced_max_blocks is None
            or int(row_counts.max().item()) <= balanced_max_blocks
        )
    ):
        # Inside the weight-stationary band (full groups, more than 24 selected
        # blocks per row on average, at most the per-arch bound), a selection
        # shared by every head over at least two persistent waves runs the
        # balanced profile: the same M64N256
        # datapath with an on-device ticket queue that hands the tiles out in
        # descending size order.  Host metadata only, so the choice and the
        # launch below are CUDA-graph safe.
        blk64_dense = (
            dense
            if dense is not None
            else _blk64_dense_from_q2k(q2k_indices, q2k_num, nb=nb)
        )
        if torch.equal(blk64_dense, blk64_dense[:1].expand_as(blk64_dense)):
            blk64_profile = _BLK64_BALANCED_PROFILE
            shared_indptr, shared_indices = _shared_bsr(
                blk64_dense, indptr, indices, trust_bsr=block_mask is None
            )
    if R != 64 and dense is not None and torch.equal(dense, dense[:1].expand_as(dense)):
        shared_indptr, shared_indices = _shared_bsr(
            dense,
            indptr,
            indices,
            trust_bsr=block_mask is None,
        )
    fp16_direct = None
    if (
        R != 64
        and head_dim == 128
        and q_data_type == torch.float16
        and num_qo_heads != num_kv_heads
    ):
        fp16_direct = _fp16_direct_metadata(
            dense,
            row_counts,
            M=M,
            N=N,
            R=R,
            mb=mb,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            device=device,
        )
    workspace: dict[str, Any] = {}
    if blk64_profile == _BLK64_BALANCED_PROFILE:
        # Plan-owned launch workspaces of the balanced profile: the ticket-queue
        # counters start at zero and are reset on device by every launch; the
        # order / trace sinks are one-element placeholders (debug_order = 0).
        workspace["blk64_queue_counters"] = torch.zeros(
            (4,), dtype=torch.uint32, device=device
        )
        workspace["blk64_order_debug"] = torch.zeros(
            (1,), dtype=torch.int32, device=device
        )
        workspace["blk64_trace_out"] = torch.zeros(
            (1,), dtype=torch.uint32, device=device
        )
    return {
        "M": M,
        "N": N,
        "R": R,
        "C": C,
        "mb": mb,
        "nb": nb,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": q_data_type,
        "sm_scale": sm_scale,
        "block_mask": dense,
        "row_counts": row_counts,
        "max_selected_blocks": max_selected_blocks,
        "uniform_selected_blocks": uniform_selected_blocks,
        "indptr": shared_indptr,
        "indices": shared_indices,
        "q2k_indices": q2k_indices,
        "q2k_num": q2k_num,
        "kv_block_lens": planned_kv_block_lens,
        "blk64_profile": blk64_profile,
        "blk64_selected_blocks_total": blk64_selected_blocks_total,
        "fp16_direct": fp16_direct,
        "workspace": workspace,
    }


def _workspace_tensor(
    plan: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    workspace = plan["workspace"]
    tensor = workspace.get(name)
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != device
    ):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        workspace[name] = tensor
    return tensor


def _outputs(
    plan: dict[str, Any],
    q: torch.Tensor,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("out must match the query shape, dtype, and device")
    stats_shape = (int(q.shape[0]), int(q.shape[1])) if return_lse else (1,)
    if lse is None:
        if return_lse:
            lse = torch.empty(stats_shape, dtype=torch.float32, device=q.device)
        else:
            lse = _workspace_tensor(
                plan, "stats_scratch", stats_shape, torch.float32, q.device
            )
    elif (
        not return_lse
        or tuple(lse.shape) != stats_shape
        or lse.dtype != torch.float32
        or lse.device != q.device
    ):
        raise ValueError("lse must be float32 [M, num_qo_heads] on the query device")
    return out, lse


def _check_inputs(
    plan: dict[str, Any], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    expected_q = (plan["M"], plan["num_qo_heads"], plan["head_dim"])
    expected_kv = (plan["N"], plan["num_kv_heads"], plan["head_dim"])
    for name, tensor, shape in (
        ("q", q, expected_q),
        ("k", k, expected_kv),
        ("v", v, expected_kv),
    ):
        if (
            tensor.device.type != "cuda"
            or tensor.device != q.device
            or tensor.dtype != plan["dtype"]
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} does not match the Cake VSA plan")


def _softmax_scale_log2(plan: dict[str, Any]) -> float:
    return float(plan["sm_scale"] or 1.0 / math.sqrt(plan["head_dim"])) / math.log(2.0)


def _run_standard(
    profile: str,
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    stats: torch.Tensor,
    *,
    return_lse: bool,
    selected_blocks: Optional[int] = None,
) -> None:
    import tvm_ffi

    module = _module(profile, q.device)
    args: list[Any] = [
        q,
        k,
        v,
        out,
        stats,
        stats,
        (
            plan["indices"]
            if profile == "ultrasparse_bsr"
            else plan["block_mask"].view(torch.uint8)
        ),
        plan["mb"],
        plan["nb"],
    ]
    if selected_blocks is not None:
        args.append(selected_blocks)
    if profile == "ultrasparse_bsr":
        if selected_blocks != 6:
            raise ValueError(
                "Cake ultrasparse route requires exactly six selected blocks"
            )
        total_tiles = plan["mb"] * plan["num_qo_heads"]
        args.append(total_tiles)
    args.extend(
        [
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            _softmax_scale_log2(plan),
            1.0,
            int(return_lse),
            0,
        ]
    )
    if profile == "gqa_mask":
        group_size = plan["num_qo_heads"] // plan["num_kv_heads"]
        tokens_per_tile = 2 * (64 // group_size)
        grid_x = (plan["M"] + tokens_per_tile - 1) // tokens_per_tile
        grid_y = plan["num_kv_heads"]
    elif profile == "ultrasparse_bsr":
        grid_x, grid_y = min(total_tiles, _sm_count(_device_index(q.device))), 1
    elif profile in {"head64_native", "head96_native"}:
        grid_x, grid_y = plan["mb"] * 2, plan["num_qo_heads"]
    else:
        grid_x, grid_y = plan["mb"], plan["num_qo_heads"]
    args.extend([grid_x, grid_y, 1])
    with tvm_ffi.use_torch_stream():
        module.run(*args)


def _run_blk64(
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    stats: torch.Tensor,
    return_lse: bool,
) -> None:
    import tvm_ffi

    profile = plan["blk64_profile"]
    module = _module(profile, q.device)
    total_tiles = plan["mb"] * plan["num_qo_heads"]
    persistent_ctas = min(total_tiles, _sm_count(_device_index(q.device)))
    if profile == _BLK64_BALANCED_PROFILE:
        workspace = plan["workspace"]
        queue_counters = workspace.get("blk64_queue_counters")
        order_debug = workspace.get("blk64_order_debug")
        trace_out = workspace.get("blk64_trace_out")
        if (
            plan["indptr"] is None
            or plan["indices"] is None
            or queue_counters is None
            or order_debug is None
            or trace_out is None
        ):
            raise RuntimeError("the Cake balanced blk64 route was not planned")
        with tvm_ffi.use_torch_stream():
            module.run(
                q,
                k,
                v,
                out,
                stats,
                plan["indptr"],
                plan["indices"],
                queue_counters,
                order_debug,
                trace_out,
                plan["M"],
                plan["mb"],
                total_tiles,
                plan["num_qo_heads"],
                _softmax_scale_log2(plan),
                int(return_lse),
                0,
                persistent_ctas,
                1,
                1,
            )
        return
    tiles_per_cta = (total_tiles + persistent_ctas - 1) // persistent_ctas
    with tvm_ffi.use_torch_stream():
        module.run(
            q,
            k,
            v,
            out,
            stats,
            plan["q2k_indices"],
            plan["q2k_num"],
            plan["kv_block_lens"],
            plan["q2k_indices"].shape[-1],
            plan["M"],
            plan["mb"],
            total_tiles,
            tiles_per_cta,
            plan["num_qo_heads"],
            _softmax_scale_log2(plan),
            int(return_lse),
            persistent_ctas,
            1,
            1,
        )


def _run_fp16(
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    stats: torch.Tensor,
    return_lse: bool,
) -> None:
    import tvm_ffi

    module = _module("fp16_direct", q.device)
    metadata = plan["fp16_direct"]
    if metadata is None:
        raise RuntimeError("the Cake FP16 direct route was not planned")
    scale_dummy = metadata["scale_dummy"]
    with tvm_ffi.use_torch_stream():
        module.run(
            q,
            k,
            scale_dummy,
            v,
            scale_dummy,
            out,
            stats,
            stats,
            metadata["q2k_indices"],
            metadata["cu_seqlens_q"],
            metadata["cu_seqlens_k"],
            metadata["q_offsets"],
            metadata["kv_lens"],
            metadata["page_table"],
            plan["M"],
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            metadata["topk"],
            1,
            0,
            0,
            0,
            0,
            _softmax_scale_log2(plan),
            1.0,
            1.0,
            1.0,
            int(return_lse),
            0,
            (plan["M"] + _FP16_DIRECT_Q_TILE - 1) // _FP16_DIRECT_Q_TILE,
            plan["num_qo_heads"],
            1,
        )


@flashinfer_api(trace=block_sparse_attention_run_trace)
def run_cake_vsa(
    plan: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
    backend: str,
):
    """Run one explicit source-level route; no external fallback is available.

    Parameters
    ----------
    plan : dict[str, Any]
        Metadata returned by :func:`plan_cake_vsa`.
    q : torch.Tensor
        Contiguous query tensor with shape
        ``(M, num_qo_heads, head_dim)``.
    k : torch.Tensor
        Contiguous key tensor with shape
        ``(N, num_kv_heads, head_dim)``.
    v : torch.Tensor
        Contiguous value tensor with the same shape and dtype as ``k``.
    out : Optional[torch.Tensor]
        Optional output buffer matching ``q``.
    lse : Optional[torch.Tensor]
        Optional float32 log-sum-exp buffer with shape
        ``(M, num_qo_heads)``. It is accepted only when ``return_lse`` is true.
    return_lse : bool
        Return log-sum-exp values with the output. D64/D96, BF16 GQA,
        ultrasparse, and long-sequence routes do not support this option.
    backend : str
        Must be ``"cake"``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Attention output, or ``(output, lse)`` when ``return_lse`` is true.
    """

    if backend != "cake":
        raise ValueError("run_cake_vsa requires backend='cake'")
    _check_inputs(plan, q, k, v)
    output, stats = _outputs(plan, q, out, lse, return_lse)
    if plan["head_dim"] in (64, 96):
        if return_lse:
            raise ValueError("Cake D64/D96 routes do not support return_lse")
        _run_standard(
            f"head{plan['head_dim']}_native",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=False,
        )
        return output

    if plan["R"] == 64:
        _run_blk64(plan, q, k, v, output, stats, return_lse)
    elif q.dtype == torch.float16:
        if plan["num_qo_heads"] == plan["num_kv_heads"]:
            _run_standard(
                "blk128_fp16_compact",
                plan,
                q,
                k,
                v,
                output,
                stats,
                return_lse=return_lse,
            )
        else:
            _run_fp16(plan, q, k, v, output, stats, return_lse)
    elif plan["num_qo_heads"] != plan["num_kv_heads"]:
        if return_lse:
            raise ValueError("Cake BF16 GQA routes do not support return_lse")
        _run_standard("gqa_mask", plan, q, k, v, output, stats, return_lse=False)
    elif (
        plan["mb"] >= 625
        and plan["num_qo_heads"] == plan["num_kv_heads"] == 8
        and plan["indices"] is not None
    ):
        if return_lse:
            raise ValueError("Cake ultrasparse routes do not support return_lse")
        selected = plan["max_selected_blocks"]
        if selected != 6 or not plan["uniform_selected_blocks"]:
            raise ValueError(
                "Cake ultrasparse route requires exactly six selected blocks"
            )
        _run_standard(
            "ultrasparse_bsr",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=False,
            selected_blocks=selected,
        )
    elif plan["N"] >= 16384 and plan["num_qo_heads"] == 8:
        if return_lse:
            raise ValueError("Cake long-sequence routes do not support return_lse")
        selected = plan["max_selected_blocks"]
        if selected > _MAX_LONGSEQ_BLOCKS or not plan["uniform_selected_blocks"]:
            raise ValueError("Cake long-sequence route requires fixed top-k <= 192")
        _run_standard(
            "longseq",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=False,
            selected_blocks=selected,
        )
    else:
        if plan["max_selected_blocks"] > _MAX_COMPACT_BLOCKS:
            raise ValueError("Cake compact route supports at most 64 selected blocks")
        _run_standard(
            "blk128_compact",
            plan,
            q,
            k,
            v,
            output,
            stats,
            return_lse=return_lse,
        )
    return (output, stats) if return_lse else output


__all__ = ["plan_cake_vsa", "run_cake_vsa"]
