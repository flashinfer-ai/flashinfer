"""Source-level CAKE backend for DeepSeek V4 sparse MLA on SM103."""

from __future__ import annotations

import threading
from typing import Literal, Union

import torch


_HEAD_DIM = 512
_TILE_KV = 128
_scale_cache: dict[tuple[int, float], torch.Tensor] = {}
_scale_cache_lock = threading.Lock()


def _module(kind: Literal["pointer", "pointer_uumn", "grid_constant"]):
    from ..jit.cake_dsv4 import get_cake_dsv4_module

    return get_cake_dsv4_module(kind)


def _stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _device_scale(
    value: Union[float, torch.Tensor], *, device: torch.device, name: str
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise ValueError(f"{name} must be a one-element FP32 tensor")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}, got {value.device}")
        return value.contiguous()

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, float(value))
    with _scale_cache_lock:
        result = _scale_cache.get(key)
        if result is None:
            result = torch.tensor([key[1]], dtype=torch.float32, device=device)
            _scale_cache[key] = result
    return result


def _workspace_views(
    workspace: torch.Tensor,
    *,
    partial_o_elems: int,
    partial_lse_elems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workspace.device.type != "cuda" or not workspace.is_contiguous():
        raise ValueError("workspace_buffer must be a contiguous CUDA tensor")
    raw = workspace.view(torch.uint8).reshape(-1)
    partial_o_bytes = partial_o_elems * torch.bfloat16.itemsize
    lse_offset = (partial_o_bytes + 15) & ~15
    required = lse_offset + partial_lse_elems * torch.float32.itemsize
    if raw.numel() < required:
        raise ValueError(
            f"workspace_buffer requires at least {required} bytes for this CAKE route, "
            f"got {raw.numel()}"
        )
    partial_o = raw[:partial_o_bytes].view(torch.bfloat16)
    partial_lse = raw[lse_offset:required].view(torch.float32)
    return partial_o, partial_lse


def _direct_lse(workspace: torch.Tensor, elems: int) -> torch.Tensor:
    return _workspace_views(
        workspace,
        partial_o_elems=0,
        partial_lse_elems=elems,
    )[1]


def _reduce(
    module,
    variant: str,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
    num_query_tokens: int,
    num_heads: int,
    num_splits: int,
    stream: int,
) -> None:
    getattr(module, f"run_{variant}")(
        partial_o,
        partial_lse,
        out,
        num_heads,
        num_splits,
        num_query_tokens,
        num_heads,
        1,
        stream,
    )


def _run_bf16_split(
    *,
    module,
    variant: str,
    reducer: str,
    query: torch.Tensor,
    swa: torch.Tensor,
    compressed: torch.Tensor,
    workspace: torch.Tensor,
    indices: torch.Tensor,
    active_lens: torch.Tensor,
    sinks: torch.Tensor,
    bmm1_scale: torch.Tensor,
    bmm2_scale: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    with_head_tiles: bool,
    stream: int,
) -> None:
    num_query_tokens = query.shape[0]
    sparse_topk = indices.shape[1]
    num_splits = (sparse_topk + _TILE_KV - 1) // _TILE_KV
    partial_lse_elems = num_query_tokens * num_heads * num_splits
    if num_splits == 1:
        partial_o = out.reshape(-1)
        partial_lse = _direct_lse(workspace, partial_lse_elems)
    else:
        partial_o, partial_lse = _workspace_views(
            workspace,
            partial_o_elems=partial_lse_elems * _HEAD_DIM,
            partial_lse_elems=partial_lse_elems,
        )

    args = [
        query,
        swa,
        compressed,
        partial_o,
        partial_lse,
        indices,
        active_lens,
        sinks,
        bmm1_scale,
        bmm2_scale,
        num_heads,
    ]
    num_head_tiles = (num_heads + 63) // 64
    if with_head_tiles:
        args.append(num_head_tiles)
    args.extend(
        [
            sparse_topk,
            num_splits,
            int(sinks.numel() == num_heads),
        ]
    )
    grid_x = num_query_tokens * num_splits * 4
    if with_head_tiles:
        grid_x *= num_head_tiles
    getattr(module, f"run_{variant}")(*args, grid_x, 1, 1, stream)
    if num_splits > 1:
        _reduce(
            module,
            reducer,
            partial_o,
            partial_lse,
            out,
            num_query_tokens,
            num_heads,
            num_splits,
            stream,
        )


def _route(
    *,
    dtype: torch.dtype,
    num_heads: int,
    max_q_len: int,
    ragged: bool,
    sparse_topk: int,
) -> str:
    if dtype == torch.float8_e4m3fn:
        if num_heads == 128:
            return "fp8_h128"
        if num_heads not in (8, 16, 32, 64):
            raise ValueError(f"unsupported CAKE FP8 DSv4 head count: {num_heads}")
        return "fp8_lowhead_prefill" if max_q_len >= 257 else "fp8_lowhead_decode"
    if dtype != torch.bfloat16:
        raise ValueError(f"unsupported CAKE DSv4 dtype: {dtype}")
    if num_heads in (8, 16, 32):
        return "bf16_h8_h32"
    if num_heads == 64:
        if not ragged:
            return "bf16_h64_fixed_q"
        if max_q_len >= 257:
            return "bf16_h64_prefill"
        return "bf16_swa128_single_cta" if sparse_topk == 128 else "bf16_h64_compressed"
    if num_heads == 128:
        if max_q_len >= 257:
            return "bf16_h128_prefill"
        if sparse_topk == 128:
            return "bf16_h128_swa128"
        if sparse_topk == 1152:
            return "bf16_h128_topk4x"
        return "bf16_h128_topk128x"
    raise ValueError(f"unsupported CAKE BF16 DSv4 head count: {num_heads}")


def run_cake_dsv4(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: torch.Tensor | None,
    max_q_len: int,
    cum_seq_lens_q: torch.Tensor | None,
    backend: Literal["cake"],
) -> torch.Tensor:
    if backend != "cake":
        raise ValueError(f"expected backend='cake', got {backend!r}")
    num_query_tokens, num_heads, head_dim = query.shape
    if head_dim != _HEAD_DIM:
        raise ValueError(f"CAKE DSv4 requires head dim {_HEAD_DIM}, got {head_dim}")
    query = query.contiguous()
    indices = sparse_indices.reshape(num_query_tokens, -1).contiguous()
    active_lens = sparse_topk_lens.reshape(-1).contiguous()
    swa = swa_kv_cache.reshape(-1, _HEAD_DIM).contiguous()
    compressed = compressed_kv_cache.reshape(-1, _HEAD_DIM).contiguous()
    out_rows = out.reshape(num_query_tokens, num_heads, _HEAD_DIM)
    scale1 = _device_scale(bmm1_scale, device=query.device, name="bmm1_scale")
    scale2 = _device_scale(bmm2_scale, device=query.device, name="bmm2_scale")
    has_sinks = sinks is not None
    sink_tensor = sinks if sinks is not None else scale1
    route = _route(
        dtype=query.dtype,
        num_heads=num_heads,
        max_q_len=max_q_len,
        ragged=cum_seq_lens_q is not None,
        sparse_topk=indices.shape[1],
    )
    stream = _stream_ptr(query.device)
    pointer = _module("pointer_uumn" if route == "bf16_h64_prefill" else "pointer")

    if route == "bf16_h8_h32":
        _run_bf16_split(
            module=pointer,
            variant=route,
            reducer="bf16_h8_h32_reduce",
            query=query,
            swa=swa,
            compressed=compressed,
            workspace=workspace_buffer,
            indices=indices,
            active_lens=active_lens,
            sinks=sink_tensor,
            bmm1_scale=scale1,
            bmm2_scale=scale2,
            out=out_rows,
            num_heads=num_heads,
            with_head_tiles=False,
            stream=stream,
        )
        return out

    if route in ("bf16_h64_compressed", "bf16_h64_fixed_q"):
        _run_bf16_split(
            module=pointer,
            variant=route,
            reducer=f"{route}_reduce",
            query=query,
            swa=swa,
            compressed=compressed,
            workspace=workspace_buffer,
            indices=indices,
            active_lens=active_lens,
            sinks=sink_tensor,
            bmm1_scale=scale1,
            bmm2_scale=scale2,
            out=out_rows,
            num_heads=num_heads,
            with_head_tiles=False,
            stream=stream,
        )
        return out

    if route == "bf16_h128_topk128x":
        _run_bf16_split(
            module=pointer,
            variant=route,
            reducer="bf16_h128_topk128x_reduce",
            query=query,
            swa=swa,
            compressed=compressed,
            workspace=workspace_buffer,
            indices=indices,
            active_lens=active_lens,
            sinks=sink_tensor,
            bmm1_scale=scale1,
            bmm2_scale=scale2,
            out=out_rows,
            num_heads=num_heads,
            with_head_tiles=True,
            stream=stream,
        )
        return out

    if route in ("bf16_swa128_single_cta", "bf16_h128_swa128"):
        scalars: tuple[int, ...]
        if route == "bf16_swa128_single_cta":
            scalars = (num_heads, int(has_sinks))
            grid_x = num_query_tokens * 4
        else:
            scalars = (num_heads, 2, int(has_sinks))
            grid_x = num_query_tokens * 8
        getattr(pointer, f"run_{route}")(
            query,
            swa,
            out_rows,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            *scalars,
            grid_x,
            1,
            1,
            stream,
        )
        return out

    if route == "bf16_h64_prefill":
        pointer.run_bf16_h64_prefill(
            query,
            swa,
            compressed,
            out_rows,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            indices.shape[1],
            int(has_sinks),
            num_query_tokens,
            1,
            1,
            stream,
        )
        return out

    if route in ("fp8_lowhead_decode", "fp8_lowhead_prefill"):
        query_u8 = query.view(torch.uint8)
        swa_u8 = swa.view(torch.uint8)
        compressed_u8 = compressed.view(torch.uint8)
        lse = _direct_lse(workspace_buffer, num_query_tokens * num_heads)
        cluster = 2 if route == "fp8_lowhead_decode" else 1
        total_work_items = num_query_tokens * 2
        getattr(pointer, f"run_{route}")(
            query_u8,
            swa_u8,
            compressed_u8,
            out_rows,
            lse,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            num_query_tokens,
            indices.shape[1],
            int(has_sinks),
            total_work_items,
            total_work_items * cluster,
            1,
            1,
            stream,
        )
        return out

    grid_module = _module("grid_constant")
    if route == "bf16_h128_prefill":
        grid_module.run_bf16_h128_prefill(
            query,
            swa,
            compressed,
            swa,
            compressed,
            out_rows,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            num_query_tokens,
            indices.shape[1],
            int(has_sinks),
            num_query_tokens,
            num_query_tokens * 2,
            1,
            1,
            stream,
        )
        return out

    if route == "bf16_h128_topk4x":
        num_splits = 3
        total_work_items = num_query_tokens * num_splits * 2
        lse_elems = num_query_tokens * num_heads * num_splits
        partial_o, partial_lse = _workspace_views(
            workspace_buffer,
            partial_o_elems=lse_elems * _HEAD_DIM,
            partial_lse_elems=lse_elems,
        )
        grid_module.run_bf16_h128_topk4x(
            query,
            swa,
            compressed,
            partial_o,
            partial_lse,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            num_query_tokens,
            indices.shape[1],
            int(has_sinks),
            total_work_items,
            num_splits,
            total_work_items * 2,
            1,
            1,
            stream,
        )
        _reduce(
            grid_module,
            "split_reduce",
            partial_o,
            partial_lse,
            out_rows,
            num_query_tokens,
            num_heads,
            num_splits,
            stream,
        )
        return out

    if route == "fp8_h128":
        num_splits = 4 if indices.shape[1] > 128 and num_query_tokens < 257 else 1
        lse_elems = num_query_tokens * num_heads * num_splits
        if num_splits == 1:
            partial_o = out_rows.reshape(-1)
            partial_lse = _direct_lse(workspace_buffer, lse_elems)
        else:
            partial_o, partial_lse = _workspace_views(
                workspace_buffer,
                partial_o_elems=lse_elems * _HEAD_DIM,
                partial_lse_elems=lse_elems,
            )
        total_work_items = num_query_tokens * num_splits
        grid_module.run_fp8_h128(
            query.view(torch.uint8),
            swa.view(torch.uint8),
            compressed.view(torch.uint8),
            partial_o,
            partial_lse,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            num_query_tokens,
            indices.shape[1],
            int(has_sinks),
            total_work_items,
            num_splits,
            total_work_items * 2,
            1,
            1,
            stream,
        )
        if num_splits > 1:
            _reduce(
                grid_module,
                "split_reduce",
                partial_o,
                partial_lse,
                out_rows,
                num_query_tokens,
                num_heads,
                num_splits,
                stream,
            )
        return out

    raise RuntimeError(f"unhandled CAKE DSv4 route: {route}")


__all__ = ["run_cake_dsv4"]
