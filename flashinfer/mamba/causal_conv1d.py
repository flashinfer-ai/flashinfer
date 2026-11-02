# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..utils import (
    backend_requirement,
    get_compute_capability,
    supported_compute_capability,
)


@supported_compute_capability([100, 103])
def _supported(x, *args, **kwargs):
    return x.is_cuda and get_compute_capability(x.device) in ((10, 0), (10, 3))


def _prefill_workspace_size(num_sequences):
    return num_sequences + 1 if num_sequences > 32 else 0


def _disjoint(name, destination, *sources):
    storage = destination.untyped_storage().data_ptr()
    if destination.numel() and any(
        tensor is not None
        and tensor.numel()
        and tensor.untyped_storage().data_ptr() == storage
        for tensor in sources
    ):
        raise ValueError(f"{name} must not share storage with another operand")


def _vector(name, tensor, size, dtype, device):
    if (
        tensor.shape != (size,)
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} [{size}] on {device}")


def _validate(x, weight, state, indices, bias, out):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError("x must be BF16 [tokens, channels] with contiguous channels")
    rows, channels = x.shape
    if not 0 < channels <= 2**31 - 1 or rows > 2**31 - 1 or x.stride(0) < channels:
        raise ValueError(
            "x requires positive channels, int32 dimensions and nonoverlapping rows"
        )
    if (
        weight.shape != (channels, 4)
        or weight.dtype not in (torch.float32, torch.bfloat16)
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "weight must be contiguous FP32 or BF16 [channels, 4] on x.device"
        )
    if (
        state.ndim != 3
        or state.shape[1] != channels
        or state.shape[2] < 3
        or state.dtype != torch.bfloat16
        or state.device != x.device
    ):
        raise ValueError(
            "conv_state must be BF16 [slots, channels, history >= 3] on x.device"
        )
    slot_stride, channel_stride, history_stride = state.stride()
    history = state.shape[2]
    if not (
        (history_stride == 1 and channel_stride >= history)
        or (channel_stride == 1 and history_stride >= channels)
    ) or slot_stride < (
        (channels - 1) * channel_stride + (history - 1) * history_stride + 1
    ):
        raise ValueError(
            "conv_state requires nonoverlapping slots and contiguous "
            "channels or history"
        )
    if indices.ndim != 1:
        raise ValueError("state indices must have shape [num_sequences]")
    _vector("state indices", indices, indices.numel(), torch.int32, x.device)
    if (rows and not indices.numel()) or indices.numel() > 2**31 - 1:
        raise ValueError("nonempty x requires an int32, nonzero sequence count")
    if bias is not None:
        if bias.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("bias must be FP32 or BF16")
        _vector("bias", bias, channels, bias.dtype, x.device)
    if out is None:
        out = torch.empty_like(x)
    elif (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or out.stride(1) != 1
        or out.stride(0) < channels
    ):
        raise ValueError(
            "out must match x with contiguous channels and nonoverlapping rows"
        )
    _disjoint("out", out, x, weight, state, indices, bias)
    _disjoint("conv_state", state, x, weight, indices, bias)
    return out


def _run(
    x, weight, state, offsets, indices, initial, accepted, bias, out, workspace, prefill
):
    if x.shape[0] == 0:
        return out
    from ..cute_dsl.causal_conv1d import BLOCK_C, get_conv_kernel, get_prefix_kernel

    with torch.cuda.device(x.device):
        major, minor = get_compute_capability(x.device)
        arch = f"sm_{major}{minor}a"
        kind = "none"
        if bias is not None:
            kind = "fp32" if bias.dtype == torch.float32 else "bf16"
        inline_prefix = prefill and not _prefill_workspace_size(indices.numel())
        block_t = 8 if prefill and x.shape[0] < 128 * indices.numel() else 16
        max_tokens = state.shape[2] - 2 if accepted is not None else 0
        if max_tokens > 16:
            max_tokens = 0
        alignment = 4 * weight.element_size()
        if weight.data_ptr() % alignment:
            alignment = weight.element_size()
        channels = x.shape[1]
        single_sequence = prefill and indices.numel() == 1
        strided_io = x.stride(0) != channels or out.stride(0) != channels
        specialize_strides = strided_io and (
            (prefill and not single_sequence) or max_tokens > 0
        )
        compact_spec = (
            max_tokens > 0 and state.stride(1) == 1 and state.stride(2) == channels
        )
        compact_ds = (
            max_tokens > 0
            and state.stride(2) == 1
            and state.stride(1) == state.shape[2]
        )
        kernel = get_conv_kernel(
            prefill,
            offsets is not None,
            initial is not None,
            accepted is not None,
            weight.dtype == torch.float32,
            kind,
            arch,
            x.device.index,
            state.stride(1) == 1,
            max_tokens,
            inline_prefix,
            alignment,
            block_t,
            full_tile=prefill and x.shape[1] >= 256,
            static_channels=channels
            if specialize_strides or compact_spec or compact_ds
            else 0,
            static_history=state.shape[2] if compact_ds else 0,
            static_strides=(
                x.stride(0),
                out.stride(0),
                state.stride(1),
                state.stride(2),
            )
            if specialize_strides
            else (),
            single_sequence=single_sequence,
            warp_metadata=accepted is not None and offsets is not None,
            vector_state=(
                compact_spec
                and channels % BLOCK_C == 0
                and state.stride(0) % 8 == 0
                and state.data_ptr() % 16 == 0
            ),
        )
        if prefill and not inline_prefix:
            get_prefix_kernel(arch, x.device.index, block_t)(offsets, workspace)
        kernel(
            x,
            weight,
            state,
            indices if offsets is None else offsets,
            indices,
            indices if initial is None else initial.view(torch.uint8),
            indices if accepted is None else accepted,
            x[0] if bias is None else bias,
            out,
            indices if workspace is None else workspace,
        )
    return out


@flashinfer_api
@backend_requirement(backend_checks={}, common_check=_supported)
def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: Optional[torch.Tensor] = None,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = _validate(x, weight, conv_state, cache_indices, bias, out)
    n = cache_indices.numel()
    _vector("cu_seqlens", cu_seqlens, n + 1, torch.int32, x.device)
    if has_initial_state is not None:
        _vector("has_initial_state", has_initial_state, n, torch.bool, x.device)
    _disjoint("out", out, cu_seqlens, has_initial_state)
    _disjoint("conv_state", conv_state, cu_seqlens, has_initial_state)
    workspace_size = _prefill_workspace_size(n)
    if workspace is None and workspace_size:
        workspace = torch.empty(workspace_size, dtype=torch.int32, device=x.device)
    elif workspace is not None and (
        workspace.ndim != 1
        or workspace.numel() < n + 1
        or workspace.dtype != torch.int32
        or workspace.device != x.device
        or not workspace.is_contiguous()
    ):
        raise ValueError(
            "workspace must be contiguous CUDA int32 with at least N+1 elements"
        )
    if workspace is not None:
        _disjoint(
            "workspace",
            workspace,
            x,
            weight,
            conv_state,
            cu_seqlens,
            cache_indices,
            has_initial_state,
            bias,
            out,
        )
    return _run(
        x,
        weight,
        conv_state,
        cu_seqlens,
        cache_indices,
        has_initial_state,
        None,
        bias,
        out,
        workspace,
        True,
    )


@flashinfer_api
@backend_requirement(backend_checks={}, common_check=_supported)
def causal_conv1d_update(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = _validate(x, weight, conv_state, state_indices, bias, out)
    n = state_indices.numel()
    if cu_seqlens is None:
        if x.shape[0] != n:
            raise ValueError("without cu_seqlens, x must have one row per state index")
    else:
        _vector("cu_seqlens", cu_seqlens, n + 1, torch.int32, x.device)
    if num_accepted_tokens is not None:
        _vector("num_accepted_tokens", num_accepted_tokens, n, torch.int32, x.device)
    _disjoint("out", out, cu_seqlens, num_accepted_tokens)
    _disjoint("conv_state", conv_state, cu_seqlens, num_accepted_tokens)
    return _run(
        x,
        weight,
        conv_state,
        cu_seqlens,
        state_indices,
        None,
        num_accepted_tokens,
        bias,
        out,
        None,
        False,
    )
