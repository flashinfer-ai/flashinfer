"""Source-level CAKE backend for DeepSeek V4 sparse MLA on SM103."""

from __future__ import annotations

import threading
from typing import Literal, Union

import torch


_HEAD_DIM = 512
_TILE_KV = 128
_scale_cache: dict[tuple[int, float], torch.Tensor] = {}
_scale_cache_lock = threading.Lock()
_descriptor_cache: dict[tuple, torch.Tensor] = {}


def _module(kind: str):
    from ..jit.cake_dsv4 import get_cake_dsv4_module

    return get_cake_dsv4_module(kind)


def _variant_module(variant: str):
    return _module(variant)


def _launch_program(variant: str, *, stream: int, **values) -> None:
    from ..jit.cake_dsv4 import (
        get_cake_dsv4_program,
        get_cake_dsv4_program_for_variant,
    )

    selected = get_cake_dsv4_program_for_variant(variant)
    if selected is None:
        raise ValueError(f"CAKE DSv4 variant has no compiled program: {variant}")
    program_id, contract = selected
    signature = contract["signature"]
    names = (
        *signature["tensor_keys"],
        *signature["workspace_keys"],
        *signature["scalar_names"],
    )
    args = [values[name] for name in names]
    getattr(get_cake_dsv4_program(program_id), contract["entry"])(*args, stream)


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
    workspace: torch.Tensor,
    variant: str,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
    num_query_tokens: int,
    num_heads: int,
    num_splits: int,
    stream: int,
) -> None:
    _launch_variant(
        variant,
        partial_o,
        partial_lse,
        out,
        num_heads,
        num_splits,
        grid=(num_query_tokens, num_heads, 1),
        stream=stream,
        workspace=workspace,
    )


def _run_bf16_split(
    *,
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
    _launch_variant(
        variant,
        *args,
        grid=(grid_x, 1, 1),
        stream=stream,
        workspace=workspace,
    )
    if num_splits > 1:
        _reduce(
            workspace,
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
    batch_size: int,
    compressed_page_size: int,
) -> str:
    if max_q_len <= 0:
        raise ValueError("max_q_len must be positive")
    is_swa = sparse_topk == 128
    is_topk4x = not is_swa and compressed_page_size == 64
    is_topk128x = not is_swa and compressed_page_size == 2
    if dtype == torch.float8_e4m3fn:
        if num_heads == 128:
            if (
                batch_size == 2
                and max_q_len == 257
                and ragged
                and is_topk4x
                and sparse_topk == 1152
            ):
                return "fp8_h128_prefill_source_persistent"
            return "fp8_h128"
        if num_heads not in (8, 16, 32, 64):
            raise ValueError(f"unsupported CAKE FP8 DSv4 head count: {num_heads}")
        if (
            num_heads == 64
            and batch_size == 2
            and max_q_len == 257
            and ragged
            and is_topk4x
            and sparse_topk == 640
        ):
            return "fp8_h64_source_exact"
        if max_q_len >= 257:
            return "fp8_lowhead_prefill"
        if is_swa:
            return "fp8_lowhead_prefill"
        if num_heads == 64:
            return "fp8_lowhead_h64"
        return (
            "fp8_lowhead_one_partition" if sparse_topk <= 256 else "fp8_lowhead_split"
        )
    if dtype != torch.bfloat16:
        raise ValueError(f"unsupported CAKE DSv4 dtype: {dtype}")
    if (
        num_heads in (8, 16)
        and batch_size == 3
        and max_q_len == 5
        and ragged
        and (
            (is_topk128x and sparse_topk == 260)
            or (is_topk4x and sparse_topk == (192 if num_heads == 8 else 256))
        )
    ):
        return "bf16_h8_h16_source_exact"
    if num_heads in (8, 16):
        if is_swa:
            return "bf16_h8_swa128_v43" if num_heads == 8 else "bf16_h16_h32_swa128_v44"
        return "bf16_h8_h32"
    if num_heads == 32:
        if is_swa:
            return "bf16_h16_h32_swa128_v44"
        if is_topk4x:
            return "bf16_h32_topk4x_v38"
        if is_topk128x:
            return "bf16_h32_topk128x_early_v47"
        raise ValueError("BF16 H32 compressed cache requires page size 64 or 2")
    if num_heads == 64:
        if not ragged:
            return "bf16_h64_fixed_q"
        if max_q_len >= 257:
            return "bf16_h64_prefill"
        if is_swa:
            return (
                "bf16_h64_guard_q_tma_batch_r25"
                if max_q_len > 5
                else "bf16_swa128_single_cta"
            )
        return "bf16_h64_compressed_q8_v38"
    if num_heads == 128:
        if max_q_len >= 257:
            return "bf16_h128_prefill_v42"
        if is_swa:
            return "bf16_h128_swa128"
        if is_topk4x and sparse_topk == 1152:
            return "bf16_h128_topk4x_v52"
        return "bf16_h128_topk128x"
    raise ValueError(f"unsupported CAKE BF16 DSv4 head count: {num_heads}")


def _descriptor_workspace(
    workspace: torch.Tensor, variant: str, tensors, num_bytes: int
):
    # The native binding caches immutable descriptor addresses. Keep their
    # storage for the module lifetime: a PyTorch suballocation can otherwise be
    # recycled without changing CUDA's allocation id. Retain only descriptors,
    # without extending query/cache tensor lifetimes.
    layout = tuple(
        (
            tensor.device,
            tensor.dtype,
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        for tensor in tensors
    )
    key = ("tma_descriptors", variant, layout)
    entry = _descriptor_cache.get(key)
    if entry is None:
        entry = _descriptor_cache.setdefault(
            key, torch.empty(num_bytes, dtype=torch.uint8, device=workspace.device)
        )
    _workspace_state(workspace)[key] = entry
    return entry


def _launch_variant(
    variant: str,
    *args,
    grid: tuple[int, int, int],
    stream: int,
    workspace: torch.Tensor,
):
    """Bind the generated ABI with descriptors retained for the module lifetime."""
    from ..jit.cake_dsv4 import get_cake_dsv4_spec

    contract = get_cake_dsv4_spec(variant)
    arg_plan = contract["arg_plan"]
    input_plan = [
        (kind, name) for kind, name in arg_plan if kind not in ("workspace", "grid")
    ]
    if len(args) != len(input_plan):
        raise ValueError(
            f"CAKE DSv4 {variant} expects {len(input_plan)} source arguments, got {len(args)}"
        )
    descriptors = None
    if contract["tma_workspace_bytes"]:
        tensors = [
            value
            for (kind, _name), value in zip(input_plan, args, strict=True)
            if kind == "tma_buffer"
        ]
        descriptors = _descriptor_workspace(
            workspace, variant, tensors, contract["tma_workspace_bytes"]
        )
    inputs = iter(args)
    grid_args = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    bound_args = []
    for kind, name in arg_plan:
        if kind == "workspace":
            if name != "tma_descriptor_workspace" or descriptors is None:
                raise ValueError(
                    f"CAKE DSv4 {variant} has an unresolved workspace argument: {name}"
                )
            bound_args.append(descriptors)
        elif kind == "grid":
            bound_args.append(grid_args[name])
        else:
            bound_args.append(next(inputs))
    # Direct-source bindings use the target FFI current stream. The old explicit
    # stream value belongs to Python orchestration, not the generated ABI.
    return getattr(_variant_module(variant), f"run_{variant}")(*bound_args)


def _partition_workspace(workspace, out, num_query_tokens, num_heads, num_splits):
    lse_elems = num_query_tokens * num_heads * num_splits
    if num_splits == 1:
        return out.reshape(-1), _direct_lse(workspace, lse_elems)
    return _workspace_views(
        workspace,
        partial_o_elems=lse_elems * _HEAD_DIM,
        partial_lse_elems=lse_elems,
    )


def _workspace_state(workspace: torch.Tensor) -> dict:
    state = getattr(workspace, "_cake_dsv4_state", None)
    if state is None:
        state = {}
        workspace._cake_dsv4_state = state
    return state


def _partition_arrivals(workspace, variant, merge_groups, num_splits):
    # The producer increments these counters and consumes successive generations.
    # Keep their lifetime with the caller's workspace, as for partial O/LSE.
    state = _workspace_state(workspace)
    key = (variant, merge_groups, num_splits)
    counters = state.get(key)
    if counters is None:
        counters = [
            torch.zeros(merge_groups, dtype=torch.uint32, device=workspace.device),
            0,
        ]
        state[key] = counters
    completion_base = counters[1]
    counters[1] += num_splits
    return counters[0], completion_base


def _padded_sparse_indices(workspace, indices):
    # The last query needs backing storage through the fixed 1152-entry staging
    # envelope. Only the live sparse prefix participates in the computation.
    rows, width = indices.shape
    flat = indices.reshape(-1)
    required = (rows - 1) * width + 1152
    if flat.numel() >= required:
        return flat
    state = _workspace_state(workspace)
    key = ("sparse_index_tail", rows, width)
    padded = state.get(key)
    if padded is None:
        padded = torch.full((required,), -1, dtype=torch.int32, device=indices.device)
        state[key] = padded
    # Public callers may update indices in place between calls.
    padded[: flat.numel()].copy_(flat)
    return padded


def _launch_shared_reduce(
    partial_o,
    partial_lse,
    out,
    num_query_tokens,
    num_heads,
    num_splits,
    stream,
    workspace,
):
    _launch_variant(
        "split_reduce",
        partial_o,
        partial_lse,
        out,
        num_heads,
        num_splits,
        grid=(num_query_tokens, num_heads, 1),
        stream=stream,
        workspace=workspace,
    )


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
    seq_lens: torch.Tensor,
    backend: Literal["cake"],
) -> torch.Tensor:
    if backend != "cake":
        raise ValueError(f"expected backend='cake', got {backend!r}")
    num_query_tokens, num_heads, head_dim = query.shape
    if head_dim != _HEAD_DIM:
        raise ValueError(f"CAKE DSv4 requires head dim {_HEAD_DIM}, got {head_dim}")
    batch_size = seq_lens.numel()
    ragged = cum_seq_lens_q is not None
    query = query.contiguous()
    indices = sparse_indices.reshape(num_query_tokens, -1).contiguous()
    active_lens = sparse_topk_lens.reshape(-1).contiguous()
    sparse_topk = indices.shape[1]
    swa = swa_kv_cache.reshape(-1, _HEAD_DIM).contiguous()
    compressed = compressed_kv_cache.reshape(-1, _HEAD_DIM).contiguous()
    seq_lens = seq_lens.reshape(-1).contiguous()
    if cum_seq_lens_q is not None:
        cum_seq_lens_q = cum_seq_lens_q.reshape(-1).contiguous()
    out_rows = out.reshape(num_query_tokens, num_heads, _HEAD_DIM)
    scale1 = _device_scale(bmm1_scale, device=query.device, name="bmm1_scale")
    scale2 = _device_scale(bmm2_scale, device=query.device, name="bmm2_scale")
    has_sinks = int(sinks is not None)
    sink_tensor = sinks if sinks is not None else scale1
    route = _route(
        dtype=query.dtype,
        num_heads=num_heads,
        max_q_len=max_q_len,
        ragged=ragged,
        sparse_topk=sparse_topk,
        batch_size=batch_size,
        compressed_page_size=compressed_kv_cache.shape[-2],
    )
    stream = _stream_ptr(query.device)

    if route == "bf16_h8_h32":
        # Retain the general low-head path outside the specialized profiles.
        _run_bf16_split(
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

    if route in (
        "bf16_swa128_single_cta",
        "bf16_h128_swa128",
        "bf16_h8_swa128_v43",
        "bf16_h16_h32_swa128_v44",
    ):
        head_tiles = (num_heads + 63) // 64 if route == "bf16_h128_swa128" else 1
        scalars = (
            (num_heads, head_tiles, has_sinks)
            if route == "bf16_h128_swa128"
            else (num_heads, has_sinks)
        )
        _launch_variant(
            route,
            query,
            swa,
            out_rows,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            *scalars,
            grid=(num_query_tokens * head_tiles * 4, 1, 1),
            stream=stream,
            workspace=workspace_buffer,
        )
        return out

    if route == "bf16_h8_h16_source_exact":
        _launch_variant(
            route,
            query,
            swa,
            compressed,
            out_rows,
            indices,
            active_lens,
            seq_lens,
            cum_seq_lens_q,
            sink_tensor,
            scale1,
            scale2,
            sparse_topk,
            max_q_len,
            batch_size,
            has_sinks,
            grid=(max_q_len, (num_heads // 8) * 4, batch_size),
            stream=stream,
            workspace=workspace_buffer,
        )
        return out

    if route == "bf16_h64_guard_q_tma_batch_r25":
        _launch_variant(
            route,
            query,
            swa,
            compressed,
            out_rows,
            indices,
            active_lens,
            seq_lens,
            cum_seq_lens_q,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            sparse_topk,
            batch_size,
            max_q_len,
            int(ragged),
            has_sinks,
            grid=(num_query_tokens, 2, 1),
            stream=stream,
            workspace=workspace_buffer,
        )
        return out

    if route in ("bf16_h64_compressed_q8_v38", "bf16_h64_fixed_q"):
        num_splits = (sparse_topk + 127) // 128
        partial_o, partial_lse = _partition_workspace(
            workspace_buffer, out_rows, num_query_tokens, num_heads, num_splits
        )
        _launch_variant(
            route,
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
            sparse_topk,
            num_splits,
            has_sinks,
            grid=(num_query_tokens * num_splits * 2, 1, 1),
            stream=stream,
            workspace=workspace_buffer,
        )
        if num_splits > 1:
            _launch_variant(
                (
                    "bf16_h64_compressed_reduce"
                    if route == "bf16_h64_compressed_q8_v38"
                    else "bf16_h64_fixed_q_reduce"
                ),
                partial_o,
                partial_lse,
                out_rows,
                num_heads,
                num_splits,
                grid=(num_query_tokens, num_heads, 1),
                stream=stream,
                workspace=workspace_buffer,
            )
        return out

    if route in ("bf16_h32_topk4x_v38", "bf16_h32_topk128x_early_v47"):
        num_splits = (sparse_topk + 127) // 128
        head_tiles = (num_heads + 7) // 8
        partial_o, partial_lse = _partition_workspace(
            workspace_buffer, out_rows, num_query_tokens, num_heads, num_splits
        )
        arrivals, completion_base = _partition_arrivals(
            workspace_buffer, route, num_query_tokens * head_tiles, num_splits
        )
        _launch_variant(
            route,
            query,
            swa,
            compressed,
            partial_o,
            partial_lse,
            out_rows,
            arrivals,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            sparse_topk,
            num_splits,
            head_tiles,
            has_sinks,
            completion_base,
            grid=(num_query_tokens * num_splits * head_tiles, 1, 1),
            stream=stream,
            workspace=workspace_buffer,
        )
        return out

    if route == "bf16_h64_prefill":
        _launch_variant(
            route,
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
            sparse_topk,
            num_query_tokens,
            has_sinks,
            grid=(num_query_tokens, 1, 1),
            stream=stream,
            workspace=workspace_buffer,
        )
        return out

    if route in ("bf16_h128_topk128x", "bf16_h128_topk4x_v52", "bf16_h128_prefill_v42"):
        num_splits = 5 if route == "bf16_h128_topk4x_v52" else 1
        partial_o, partial_lse = _partition_workspace(
            workspace_buffer, out_rows, num_query_tokens, num_heads, num_splits
        )
        packed_indices = (
            _padded_sparse_indices(workspace_buffer, indices)
            if route == "bf16_h128_topk128x"
            else indices
        )
        total_work_items = num_query_tokens * num_splits
        _launch_program(
            route,
            stream=stream,
            Q=query,
            SWA_cache=swa,
            compressed_KV_cache=compressed,
            partial_O=partial_o,
            partial_lse=partial_lse,
            sparse_indices=packed_indices,
            sparse_topk_lens=active_lens,
            sinks=sink_tensor,
            bmm1_scale=scale1,
            bmm2_scale=scale2,
            O=out_rows,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
            sparse_topk=sparse_topk,
            has_sinks=has_sinks,
            total_work_items=total_work_items,
            num_split=num_splits,
        )
        return out

    query_u8 = query.view(torch.uint8)
    swa_u8 = swa.view(torch.uint8)
    compressed_u8 = compressed.view(torch.uint8)

    if route == "fp8_h64_source_exact":
        head_tiles = (num_heads + 63) // 64
        total_work_items = max_q_len * head_tiles * batch_size
        _launch_variant(
            route,
            query_u8,
            swa_u8,
            compressed_u8,
            out_rows,
            cum_seq_lens_q,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            sparse_topk,
            has_sinks,
            total_work_items,
            grid=(max_q_len, head_tiles, batch_size),
            stream=stream,
            workspace=workspace_buffer,
        )
        return out

    if route == "fp8_h128_prefill_source_persistent":
        _launch_variant(
            route,
            query_u8,
            swa_u8,
            compressed_u8,
            out_rows.reshape(-1),
            _direct_lse(workspace_buffer, 1),
            indices,
            active_lens,
            seq_lens,
            cum_seq_lens_q,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            num_query_tokens,
            sparse_topk,
            has_sinks,
            num_query_tokens,
            max_q_len,
            batch_size,
            grid=(num_query_tokens * 2, 1, 1),
            stream=stream,
            workspace=workspace_buffer,
        )
        return out

    if route == "fp8_h128":
        num_splits = 5 if sparse_topk > 128 and num_query_tokens < 128 else 1
        partial_o, partial_lse = _partition_workspace(
            workspace_buffer, out_rows, num_query_tokens, num_heads, num_splits
        )
        total_work_items = num_query_tokens * num_splits
        _launch_program(
            route,
            stream=stream,
            Q=query_u8,
            SWA_cache=swa_u8,
            compressed_KV_cache=compressed_u8,
            partial_O=partial_o,
            partial_lse=partial_lse,
            sparse_indices=indices,
            sparse_topk_lens=active_lens,
            sinks=sink_tensor,
            bmm1_scale=scale1,
            bmm2_scale=scale2,
            O=out_rows,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
            sparse_topk=sparse_topk,
            has_sinks=has_sinks,
            total_work_items=total_work_items,
            num_split=num_splits,
        )
        return out

    if route in (
        "fp8_lowhead_swa",
        "fp8_lowhead_one_partition",
        "fp8_lowhead_split",
        "fp8_lowhead_h64",
        "fp8_lowhead_prefill",
    ):
        num_splits = 2 if route == "fp8_lowhead_split" else 1
        partial_o, partial_lse = _partition_workspace(
            workspace_buffer, out_rows, num_query_tokens, num_heads, num_splits
        )
        work_factor = (
            2 if route in ("fp8_lowhead_swa", "fp8_lowhead_prefill") else num_splits
        )
        total_work_items = num_query_tokens * work_factor
        cluster = 1 if route == "fp8_lowhead_prefill" else 2
        _launch_variant(
            route,
            query_u8,
            swa_u8,
            compressed_u8,
            partial_o,
            partial_lse,
            indices,
            active_lens,
            sink_tensor,
            scale1,
            scale2,
            num_heads,
            num_query_tokens,
            sparse_topk,
            has_sinks,
            total_work_items,
            grid=(total_work_items * cluster, 1, 1),
            stream=stream,
            workspace=workspace_buffer,
        )
        if num_splits > 1:
            _launch_shared_reduce(
                partial_o,
                partial_lse,
                out_rows,
                num_query_tokens,
                num_heads,
                num_splits,
                stream,
                workspace_buffer,
            )
        return out

    raise RuntimeError(f"unhandled CAKE DSv4 route: {route}")


__all__ = ["run_cake_dsv4"]
