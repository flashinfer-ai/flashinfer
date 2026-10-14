"""Host helpers for the request-ordered Cake FMHA paged-decode component."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .jit.cake_fmha_request_ordered import (
    get_cake_fmha_request_ordered_manifest,
    load_cake_fmha_request_ordered_module,
)
from .utils import get_compute_capability


@dataclass(frozen=True)
class CakeFmhaRequestOrderedDecodePlan:
    """Immutable host-selected launch plan for graph-safe request ordering."""

    module_name: str
    batch_size: int
    q_len: int
    workspace_parts: int
    grid: tuple[int, int, int]
    total_tiles: int
    write_lse: bool


def _fallback_cake_fmha_request_ordered_plan(
    *, batch_size: int, q_len: int, write_lse: bool
) -> CakeFmhaRequestOrderedDecodePlan:
    """The single-split persistent program for ``q_len``/``write_lse``: valid for every KV length."""

    if batch_size <= 0 or q_len not in (1, 6):
        raise ValueError(
            "request-ordered Cake FMHA requires a positive batch and q_len 1 or 6"
        )
    matches = [
        module["name"]
        for module in get_cake_fmha_request_ordered_manifest()["modules"]
        if module["kind"] == "persistent"
        and module["num_split"] == 1
        and module["q_len"] == q_len
        and module["write_lse"] is write_lse
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"generated request-order fallback for q_len {q_len} "
            f"(write_lse={write_lse}) is not unique"
        )
    return CakeFmhaRequestOrderedDecodePlan(
        module_name=matches[0],
        batch_size=batch_size,
        q_len=q_len,
        workspace_parts=1,
        grid=(q_len, 1, batch_size),
        total_tiles=batch_size * q_len,
        write_lse=write_lse,
    )


def _exact_route_lengths(route: dict[str, Any]) -> tuple[int, ...]:
    period = tuple(int(value) for value in route["kv_lens"]["period"])
    count = int(route["kv_lens"]["count"])
    return (period * -(-count // len(period)))[:count]


def _plan_from_exact_route(
    route: dict[str, Any], *, batch_size: int
) -> CakeFmhaRequestOrderedDecodePlan:
    grid_x, grid_y, grid_z = (int(value) for value in route["grid"])
    return CakeFmhaRequestOrderedDecodePlan(
        module_name=str(route["module"]),
        batch_size=batch_size,
        q_len=int(route["q_len"]),
        workspace_parts=int(route["workspace_parts"]),
        grid=(grid_x, grid_y, grid_z),
        total_tiles=int(route["total_tiles"]),
        write_lse=bool(route["write_lse"]),
    )


def plan_cake_fmha_request_ordered_paged_decode(
    kv_lens: Sequence[int],
    q_len: int,
    *,
    request_order_case: Literal["identity", "length_desc"] = "length_desc",
    real_batch_size: int | None = None,
    write_lse: bool = False,
) -> CakeFmhaRequestOrderedDecodePlan:
    """Select an exported schedule from host-visible immutable metadata.

    The returned plan contains no device data.  Call this before CUDA Graph
    capture, then update the contents of the device ``request_order`` tensor
    in place between replays.  An exact exported route (two-wave and low-Q1
    cluster schedules, split LSE rows) is selected only for its exported
    request lengths; everything else uses the single-split persistent program.
    """

    lengths = tuple(int(value) for value in kv_lens)
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("kv_lens must contain one positive length per request")
    if q_len not in (1, 6):
        raise ValueError("request-ordered Cake FMHA requires q_len 1 or 6")
    batch_size = len(lengths)
    logical_batch = batch_size if real_batch_size is None else int(real_batch_size)
    if not 0 < logical_batch <= batch_size:
        raise ValueError("real_batch_size must be in [1, len(kv_lens)]")

    for route in get_cake_fmha_request_ordered_manifest()["exact_routes"]:
        if (
            int(route["q_len"]) == q_len
            and int(route["kv_lens"]["count"]) == batch_size
            and int(route["real_batch_size"]) == logical_batch
            and route["request_order_case"] == request_order_case
            and bool(route["write_lse"]) is bool(write_lse)
            and _exact_route_lengths(route) == lengths
        ):
            return _plan_from_exact_route(route, batch_size=batch_size)
    return _fallback_cake_fmha_request_ordered_plan(
        batch_size=batch_size,
        q_len=q_len,
        write_lse=write_lse,
    )


def _is_authenticated_request_ordered_plan(
    plan: CakeFmhaRequestOrderedDecodePlan,
) -> bool:
    """Return whether ``plan`` names an exported mutable-order route."""

    fallback = _fallback_cake_fmha_request_ordered_plan(
        batch_size=plan.batch_size,
        q_len=plan.q_len,
        write_lse=plan.write_lse,
    )
    if plan == fallback:
        return True
    for route in get_cake_fmha_request_ordered_manifest()["exact_routes"]:
        if (
            int(route["kv_lens"]["count"]) == plan.batch_size
            and route["module"] == plan.module_name
            and int(route["q_len"]) == plan.q_len
            and int(route["workspace_parts"]) == plan.workspace_parts
            and tuple(int(value) for value in route["grid"]) == plan.grid
            and int(route["total_tiles"]) == plan.total_tiles
            and bool(route["write_lse"]) is plan.write_lse
        ):
            return True
    return False


def _run_cake_fmha_request_ordered_paged_decode(
    *,
    backend: Literal["cake"],
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    workspace_buffer: torch.Tensor,
    completion_buffer: torch.Tensor | None,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    request_order: torch.Tensor,
    max_seq_len: int,
    bmm1_scale_log2: torch.Tensor,
    bmm2_scale: torch.Tensor,
    uses_shared_paged_kv_idx: bool,
    plan: CakeFmhaRequestOrderedDecodePlan,
) -> None:
    """Launch the explicitly selected Cake backend without auxiliary kernels.

    ``backend`` is the typed handoff from the public dispatcher; route admission
    has already rejected any other backend before entering this function.
    """

    if not query.is_cuda:
        raise ValueError("request-ordered Cake FMHA tensors must be on CUDA")
    if get_compute_capability(query.device) != (10, 3):
        raise RuntimeError("request-ordered Cake FMHA requires compute capability 10.3")
    sm_count = torch.cuda.get_device_properties(query.device).multi_processor_count
    if sm_count != 152:
        raise RuntimeError(
            f"request-ordered Cake FMHA requires a 152-SM device, got {sm_count}"
        )
    batch_size, q_len = plan.batch_size, plan.q_len
    if query.shape != (batch_size * q_len, 8, 256):
        raise ValueError(
            "request-ordered Cake FMHA requires Q shape [batch*q_len, 8, 256]"
        )
    if query.dtype != torch.bfloat16 or not query.is_contiguous():
        raise TypeError("request-ordered Cake FMHA requires contiguous BF16 Q")
    if (
        out.shape != query.shape
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise TypeError(
            "request-ordered Cake FMHA requires contiguous BF16 O matching Q"
        )
    if (
        key_cache.ndim != 4
        or value_cache.ndim != 4
        or tuple(key_cache.shape[1:]) != (1, 64, 256)
        or value_cache.shape != key_cache.shape
        or key_cache.dtype != torch.float8_e4m3fn
        or value_cache.dtype != torch.float8_e4m3fn
        or key_cache.stride(-1) != 1
        or value_cache.stride(-1) != 1
    ):
        raise TypeError(
            "request-ordered Cake FMHA requires FP8 E4M3 HND K/V with a "
            "contiguous head dimension and shape [pages, 1, 64, 256]"
        )
    if max_seq_len <= 0:
        raise ValueError("request-ordered Cake FMHA requires max_seq_len > 0")
    tensors = (
        key_cache,
        value_cache,
        out,
        workspace_buffer,
        block_tables,
        seq_lens,
        request_order,
        bmm1_scale_log2,
        bmm2_scale,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all request-ordered Cake FMHA tensors must share Q's device")
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.shape != (batch_size,)
        or not seq_lens.is_contiguous()
    ):
        raise TypeError("seq_lens must be contiguous CUDA int32 [batch]")
    if (
        request_order.dtype != torch.int32
        or request_order.shape != (batch_size,)
        or not request_order.is_contiguous()
    ):
        raise TypeError("request_order must be contiguous CUDA int32 [batch]")
    if block_tables.dtype != torch.int32 or not block_tables.is_contiguous():
        raise TypeError("block_tables must be contiguous CUDA int32")
    required_pages = 2 * (((max(1, (int(max_seq_len) + 127) // 128) + 1) // 2) * 2)
    if uses_shared_paged_kv_idx:
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            raise ValueError("shared block_tables must have shape [batch, pages]")
        pages = int(block_tables.shape[1])
        page_table_stride = pages
        page_table_v_offset = 0
        page_table_arg = block_tables
    else:
        if block_tables.ndim != 3 or block_tables.shape[:2] != (batch_size, 2):
            raise ValueError("separate block_tables must have shape [batch, 2, pages]")
        pages = int(block_tables.shape[2])
        page_table_stride = 2 * pages
        page_table_v_offset = pages
        page_table_arg = block_tables.view(batch_size, 2 * pages)
    if pages < required_pages:
        raise ValueError(
            f"block_tables needs at least {required_pages} page slots for max_seq_len"
        )
    if (
        workspace_buffer.dtype != torch.uint8
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.data_ptr() % 128
    ):
        raise TypeError(
            "workspace_buffer must be contiguous, 128-byte-aligned CUDA uint8"
        )
    for name, scale in (
        ("bmm1_scale_log2", bmm1_scale_log2),
        ("bmm2_scale", bmm2_scale),
    ):
        if (
            scale.dtype != torch.float32
            or scale.numel() < 1
            or not scale.is_contiguous()
        ):
            raise TypeError(f"{name} must be nonempty contiguous CUDA float32")
    if plan.write_lse:
        if (
            lse is None
            or lse.device != query.device
            or lse.dtype != torch.float32
            or lse.shape != query.shape[:-1]
            or not lse.is_contiguous()
        ):
            raise TypeError("the selected plan requires contiguous FP32 LSE")
        lse_arg = lse
    else:
        if lse is not None:
            raise ValueError("the selected plan does not write LSE")
        if workspace_buffer.numel() < 388:
            raise ValueError(
                "workspace_buffer is too small for the TMA and dummy LSE slots"
            )
        lse_arg = workspace_buffer[384:388].view(torch.float32)

    if workspace_buffer.numel() < 384:
        raise ValueError(
            "workspace_buffer needs at least 384 bytes for TMA descriptors"
        )
    if not _is_authenticated_request_ordered_plan(plan):
        raise ValueError("request_order_plan does not match an exported route")
    tma_workspace = workspace_buffer[:384]
    if plan.workspace_parts == 1:
        partial_o = out
        partial_lse = lse_arg
        completion = seq_lens.view(torch.uint32)
    else:
        completion_elems = batch_size * q_len
        if (
            completion_buffer is None
            or completion_buffer.device != query.device
            or completion_buffer.dtype not in (torch.int32, torch.uint32)
            or not completion_buffer.is_contiguous()
            or completion_buffer.numel() < completion_elems
        ):
            raise ValueError(
                "split request-order plans require a zero-initialized contiguous "
                "int32/uint32 multi_ctas_kv_counter_buffer with batch*q_len elements"
            )
        completion = completion_buffer[:completion_elems].view(torch.uint32)
        rows = batch_size * q_len * 8 * plan.workspace_parts
        partial_o_bytes = rows * 256 * 2
        partial_lse_bytes = rows * 4
        cursor = 32 * 1024 * 1024
        required = cursor + partial_o_bytes + partial_lse_bytes
        if workspace_buffer.numel() < required:
            raise ValueError(
                f"workspace_buffer needs at least {required} bytes for this plan"
            )
        partial_o = workspace_buffer[cursor : cursor + partial_o_bytes].view(
            torch.bfloat16
        )
        cursor += partial_o_bytes
        partial_lse = workspace_buffer[cursor : cursor + partial_lse_bytes].view(
            torch.float32
        )

    import tvm_ffi

    module = load_cake_fmha_request_ordered_module(plan.module_name)
    with tvm_ffi.use_torch_stream():
        module.run(
            query,
            key_cache.view(torch.uint8),
            value_cache.view(torch.uint8),
            partial_o,
            partial_lse,
            completion,
            out,
            lse_arg,
            page_table_arg,
            seq_lens,
            request_order,
            page_table_stride,
            page_table_v_offset,
            0.0,
            0.0,
            bmm1_scale_log2,
            bmm2_scale,
            1,
            8,
            1,
            batch_size,
            plan.total_tiles,
            tma_workspace,
            *plan.grid,
        )


__all__ = [
    "CakeFmhaRequestOrderedDecodePlan",
    "plan_cake_fmha_request_ordered_paged_decode",
]
