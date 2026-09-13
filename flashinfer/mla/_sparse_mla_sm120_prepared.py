from __future__ import annotations

from dataclasses import dataclass, replace
import functools
import math
from typing import Optional, Tuple

import torch

from ..autotuner import AutoTuner
from . import _sparse_mla_sm120_calibration as calibration
from . import _sparse_mla_sm120_policy as policy
from ._sparse_mla_sm120_execution import (
    AttentionMetadata,
    ExecutionPlan,
    get_sparse_mla_sm120_module,
    resolve_attention,
)


def _workspace_tensor_view(
    workspace_buffer: torch.Tensor,
    *,
    byte_offset: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    alignment: int = 16,
) -> Tuple[Optional[torch.Tensor], int]:
    if not workspace_buffer.is_contiguous():
        return None, byte_offset
    # dtype.itemsize avoids creating a tensor on the process-wide default
    # device while partitioning caller-owned storage during graph capture.
    elem_size = dtype.itemsize
    alignment = math.lcm(alignment, elem_size)
    address = workspace_buffer.data_ptr() + byte_offset
    byte_offset += (-address) % alignment
    numel = math.prod(shape)
    byte_end = byte_offset + numel * elem_size
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if byte_end > workspace_bytes:
        return None, byte_offset
    flat = workspace_buffer.view(torch.uint8)
    view = flat[byte_offset:byte_end].view(dtype).view(shape)
    return view, byte_end


@functools.cache
def device_caps(device: torch.device) -> tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count, props.shared_memory_per_block_optin


def tuning_enabled(is_dsv4_nvfp4: bool) -> bool:
    tuner = AutoTuner.get()
    stack = tuner._get_skip_ops_stack()
    name = "sparse_mla_sm120_nvfp4" if is_dsv4_nvfp4 else "sparse_mla_sm120"
    return tuner.is_tuning_mode and not (stack and name in stack[-1])


def tensor_signature(tensor: torch.Tensor | None) -> tuple | None:
    if tensor is None:
        return None
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


def validate_metadata(
    *tensors, model: int, is_dsv4_nvfp4: bool, extra_fp4: bool, value_dim: int = 0
) -> AttentionMetadata:
    from ._sparse_mla_sm120_execution import query

    try:
        values = query(
            "inspect_metadata",
            *tensors,
            model,
            extra_fp4,
            value_dim,
            is_dsv4_nvfp4=is_dsv4_nvfp4,
        )
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    return AttentionMetadata(*values)


def resolve_execution(
    metadata: AttentionMetadata,
    device: torch.device,
    precision: str,
    is_dsv4_nvfp4: bool,
    preference: int,
) -> ExecutionPlan:
    m = metadata
    if is_dsv4_nvfp4:
        from ._sparse_mla_sm120_dsv4_nvfp4_policy import (
            plan_nvfp4_sparse_mla_sm120,
            NVFP4KernelVariant,
        )

        selected = plan_nvfp4_sparse_mla_sm120(
            m.tokens,
            m.heads,
            m.topk,
            m.page_size,
            device,
            extra_topk=m.extra_topk,
            extra_page_size=m.extra_page_size,
            has_topk_length=m.has_lengths,
            has_extra_topk_length=m.has_extra_lengths,
            has_attn_sink=m.has_sink,
        )
        if selected is None:
            raise ValueError("no NVFP4 attention route serves this metadata")
        sm, shared = device_caps(device)
        return get_sparse_mla_sm120_module().dsv4_nvfp4_resolve_attention(
            m._replace(
                variant=int(selected.variant is NVFP4KernelVariant.PREFILL_STREAMING)
            ),
            3,
            max(0, selected.cpb),
            sm,
            shared,
            False,
        )
    selected = (
        policy.profile_selection(m, device, precision) if preference == 0 else None
    )
    selected = selected or policy.plan(
        m.tokens,
        m.heads,
        m.topk,
        m.model,
        m.page_size,
        m.extra_topk > 0,
        preference,
        device,
        extra_topk=m.extra_topk,
        extra_fp4=m.extra_fp4,
        compute_precision=precision,
        extra_page_block_size=m.extra_page_size,
    )
    sm, shared = device_caps(device)
    selected = policy.filter_metadata_selection(selected, m, precision, sm, shared)
    if selected is None:
        raise ValueError("no prefill or decode kernel serves this metadata")
    return resolve_attention(
        **m._replace(variant=int(selected.variant))._asdict(),
        precision=precision,
        cpb=max(0, selected.cpb),
        sm_count=sm,
        max_shared_bytes=shared,
    )


@dataclass(frozen=True)
class PreparedCall:
    plan: ExecutionPlan
    execute_fn: object
    epoch: int
    workspace: tuple
    mid: torch.Tensor | None = None
    mlse: torch.Tensor | None = None
    lse: torch.Tensor | None = None

    def execute(
        self,
        q,
        cache,
        indices,
        output,
        scale,
        lengths,
        sink,
        extra,
        extra_indices,
        extra_lengths,
        mid=None,
        mlse=None,
        lse=None,
    ):
        self.execute_fn(
            self.plan,
            q,
            cache,
            indices,
            self.mid if mid is None else mid,
            self.mlse if mlse is None else mlse,
            output,
            self.lse if lse is None else lse,
            scale,
            lengths,
            sink,
            extra,
            extra_indices,
            extra_lengths,
        )


def _execute_fn(is_dsv4_nvfp4: bool):
    module = get_sparse_mla_sm120_module()
    return (
        module.dsv4_nvfp4_execute_attention
        if is_dsv4_nvfp4
        else module.execute_attention
    )


def prepare(
    metadata,
    device,
    precision,
    is_dsv4_nvfp4,
    preference,
    *,
    owned,
    caller_scratch,
    caller_lse,
    current=None,
):
    plan = resolve_execution(metadata, device, precision, is_dsv4_nvfp4, preference)
    workspace = tuple(
        (tuple(shape), getattr(torch, str(dtype)), int(size), int(alignment))
        for shape, dtype, size, alignment in plan.workspace()
    )
    execute_fn = _execute_fn(is_dsv4_nvfp4)
    if current is not None and current.workspace == workspace:
        return replace(
            current,
            plan=plan,
            execute_fn=execute_fn,
            epoch=calibration._constants_version,
        )
    mid = mlse = lse = None
    if owned:
        if not caller_scratch:
            mid = torch.empty(workspace[0][0], dtype=workspace[0][1], device=device)
            mlse = torch.empty(workspace[1][0], dtype=workspace[1][1], device=device)
        if not caller_lse:
            lse = torch.empty(workspace[2][0], dtype=workspace[2][1], device=device)
    return PreparedCall(
        plan,
        execute_fn,
        calibration._constants_version,
        workspace,
        mid,
        mlse,
        lse,
    )


def prefix_scratch(prepared, mid, mlse):
    for tensor, (_, _, need, _) in zip((mid, mlse), prepared.workspace):
        if tensor is not None and tensor.numel() * tensor.element_size() < need:
            raise ValueError(f"scratch capacity is too small: need {need} bytes")
    return mid, mlse


def wrapper_run(
    wrapper,
    q,
    cache,
    indices,
    output,
    scale,
    *,
    topk_length=None,
    attn_sink=None,
    extra_kv_cache=None,
    extra_indices=None,
    extra_topk_length=None,
    out_lse=None,
    mid_out=None,
    mid_lse=None,
    prefill_impl=None,
    return_lse=False,
):
    from ._sparse_mla_sm120_execution import resolve_model_type as _resolve_model_type

    is_dsv4_nvfp4 = wrapper._kv_cache_format == "nvfp4"
    q = q.squeeze(1) if q.ndim == 4 and q.shape[1] == 1 else q
    output = output.squeeze(1) if output.ndim == 4 and output.shape[1] == 1 else output
    indices = (
        indices.squeeze(1) if indices.ndim == 3 and indices.shape[1] == 1 else indices
    )
    if (
        extra_indices is not None
        and extra_indices.ndim == 3
        and extra_indices.shape[1] == 1
    ):
        extra_indices = extra_indices.squeeze(1)
    if q.ndim != 3:
        raise ValueError("q must be [T,H,D] or [T,1,H,D]")
    t, h, _ = q.shape
    if q.device != wrapper._device:
        raise ValueError("tensors must be on the Wrapper device")
    if (wrapper._max_num_tokens is not None and t > wrapper._max_num_tokens) or (
        wrapper._max_num_heads is not None and h > wrapper._max_num_heads
    ):
        raise ValueError("query exceeds max_num_tokens/max_num_heads")
    if is_dsv4_nvfp4 and prefill_impl not in (None, "auto", "mg"):
        raise ValueError("NVFP4 prefill_impl must be None, auto, or mg")
    pref = 0 if is_dsv4_nvfp4 else policy._normalize_prefill_impl(prefill_impl)
    model = (
        1
        if is_dsv4_nvfp4
        else _resolve_model_type(q.shape[-1], wrapper._kv_scale_format)
    )
    tensors = (
        q,
        cache,
        indices,
        output,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        out_lse,
        mid_out,
        mid_lse,
    )
    key = (pref, tuple(tensor_signature(x) for x in tensors))
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing:
        calibration.refresh_store()
    current = wrapper._prepared_calls.get(key)
    if t == 0:
        if out_lse is not None:
            return out_lse[:0, :h] if return_lse else None
        return (
            torch.empty((0, h), device=q.device, dtype=torch.float32)
            if return_lse
            else None
        )
    if capturing:
        if current is None:
            raise ValueError("warm up this attention shape before CUDA graph capture")
    elif (
        current is None
        or current.epoch != calibration._constants_version
        or tuning_enabled(is_dsv4_nvfp4)
    ):
        m = validate_metadata(
            *tensors,
            model=model,
            is_dsv4_nvfp4=is_dsv4_nvfp4,
            extra_fp4=wrapper._extra_kv_fp4,
            value_dim=wrapper._d_v,
        )
        updated = prepare(
            m,
            q.device,
            wrapper._compute_precision,
            is_dsv4_nvfp4,
            pref,
            owned=True,
            caller_scratch=mid_out is not None,
            caller_lse=out_lse is not None,
            current=current,
        )
        wrapper._prepared_calls[key] = current = updated
    mid_out, mid_lse = prefix_scratch(current, mid_out, mid_lse)
    result_lse = out_lse[:t, :h] if out_lse is not None else current.lse
    current.execute(
        q,
        cache,
        indices,
        output,
        scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        mid_out,
        mid_lse,
        result_lse,
    )
    return result_lse if return_lse else None


_functional_plans: dict[tuple, PreparedCall] = {}


def functional_run(
    q,
    cache,
    indices,
    output,
    workspace,
    scale,
    *,
    lengths=None,
    sink=None,
    extra=None,
    extra_indices=None,
    extra_lengths=None,
    lse=None,
    kv_scale_format="auto",
    is_dsv4_nvfp4=False,
    extra_fp4=False,
):
    from ._sparse_mla_sm120_execution import resolve_model_type as _resolve_model_type

    model = 1 if is_dsv4_nvfp4 else _resolve_model_type(q.shape[-1], kv_scale_format)
    tensors = (
        q,
        cache,
        indices,
        output,
        lengths,
        sink,
        extra,
        extra_indices,
        extra_lengths,
        lse,
        None,
        None,
    )
    key = (model, is_dsv4_nvfp4, extra_fp4, tuple(tensor_signature(x) for x in tensors))
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing:
        calibration.refresh_store()
    current = _functional_plans.get(key)
    if capturing:
        if current is None:
            raise ValueError(
                "warm up functional attention metadata before CUDA graph capture"
            )
    elif (
        current is None
        or current.epoch != calibration._constants_version
        or tuning_enabled(is_dsv4_nvfp4)
    ):
        m = validate_metadata(
            *tensors,
            model=model,
            is_dsv4_nvfp4=is_dsv4_nvfp4,
            extra_fp4=extra_fp4,
        )
        current = prepare(
            m,
            q.device,
            "default",
            is_dsv4_nvfp4,
            0,
            owned=False,
            caller_scratch=True,
            caller_lse=True,
            current=current,
        )
        _functional_plans[key] = current
    requirements = current.workspace[:2] if lse is not None else current.workspace
    views, offset = [], 0
    for shape, dtype, _, alignment in requirements:
        view, offset = _workspace_tensor_view(
            workspace, byte_offset=offset, shape=shape, dtype=dtype, alignment=alignment
        )
        views.append(view)
    if any(view is None for view in views):
        if capturing or is_dsv4_nvfp4:
            raise ValueError("attention workspace insufficient for resolved plan")
        views = [
            torch.empty(shape, dtype=dtype, device=q.device)
            for shape, dtype, _, _ in requirements
        ]
    mid, mlse = views[:2]
    result = lse if lse is not None else views[2]
    current.execute(
        q,
        cache,
        indices,
        output,
        scale,
        lengths,
        sink,
        extra,
        extra_indices,
        extra_lengths,
        mid,
        mlse,
        result,
    )
    return result
