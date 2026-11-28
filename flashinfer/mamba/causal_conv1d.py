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


def _prefill_block_t(rows, num_sequences):
    threshold = 64 if num_sequences > 32 else 128
    return 8 if rows < threshold * num_sequences else 16


def _prefill_workspace_size(rows, num_sequences):
    if num_sequences <= 32 or rows == 0:
        return 0
    block_t = _prefill_block_t(rows, num_sequences)
    capacity = min(rows, (rows + block_t - 1) // block_t + num_sequences)
    return 1 + 2 * capacity


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
    from ..cute_dsl.causal_conv1d import BLOCK_C, get_chunk_map_kernel, get_conv_kernel

    with torch.cuda.device(x.device):
        major, minor = get_compute_capability(x.device)
        arch = f"sm_{major}{minor}a"
        kind = "none"
        if bias is not None:
            kind = "fp32" if bias.dtype == torch.float32 else "bf16"
        inline_prefix = prefill and indices.numel() <= 32
        block_t = _prefill_block_t(x.shape[0], indices.numel()) if prefill else 16
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
            get_chunk_map_kernel(arch, x.device.index, block_t)(offsets, workspace)
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
    """Apply width-four causal depthwise convolution and SiLU to packed sequences.

    Parameters
    ----------
    x : torch.Tensor
        BF16 CUDA input of shape ``[rows, channels]`` with contiguous channels.
        Rows may have padding. State history stores these raw inputs.
    weight : torch.Tensor
        Contiguous FP32 or BF16 CUDA taps ``[channels, 4]``. Tap 3 multiplies
        the current token. Accumulation and SiLU use FP32.
    conv_state : torch.Tensor
        In-place BF16 pool ``[slots, channels, L]``, ``L >= 3``. Either the
        channel or history axis must be contiguous; slots must not overlap.
        Each nonempty active sequence writes its final three raw history
        values to positions 0:3, leaving the rest of the pool unchanged.
    cu_seqlens : torch.Tensor
        Contiguous CUDA int32 offsets ``[N+1]``, starting at zero,
        nondecreasing, with the last offset at most ``rows``. Rows after the
        last offset are not written. Empty sequences do not update state.
    cache_indices : torch.Tensor
        Contiguous CUDA int32 slot IDs ``[N]``. The -1 sentinel produces zero
        output without accessing state. Live slots must be unique and in
        ``[0, slots)``; slot zero is valid.
    has_initial_state : Optional[torch.Tensor]
        Contiguous CUDA bool ``[N]``. True rows read state positions 0:3;
        false or omitted rows start from zero history.
    bias : Optional[torch.Tensor]
        Contiguous FP32 or BF16 CUDA ``[channels]``, added before the four FMAs.
    out : Optional[torch.Tensor]
        BF16 output matching x, with contiguous channels and nonoverlapping
        rows. Must not share storage with any operand. Allocated if omitted;
        unused trailing rows of an allocated output are unspecified.
    workspace : Optional[torch.Tensor]
        Contiguous CUDA int32 scratch. For ``N > 32`` and nonempty x, needs
        ``1 + 2 * min(rows, ceil(rows / B) + N)`` elements, where B is 8 if
        ``rows < 64*N`` and 16 otherwise. Allocated if omitted. Smaller
        batches need no scratch. The chunk map is rebuilt on every call.

    Returns
    -------
    torch.Tensor
        BF16 output ``[rows, channels]``; conv_state is updated in place.

    Notes
    -----
    Device offset and slot values are caller-owned constraints; no host read
    or validation synchronization occurs. State and workspace must not alias
    other operands. Concurrent calls need disjoint live slots and scratch.
    CUDA graph replay may change device metadata in place within fixed
    capacities. Supply out and workspace before capture to reuse storage.
    After prefill, the first speculative update uses accepted count 1.
    """
    out = _validate(x, weight, conv_state, cache_indices, bias, out)
    n = cache_indices.numel()
    _vector("cu_seqlens", cu_seqlens, n + 1, torch.int32, x.device)
    if has_initial_state is not None:
        _vector("has_initial_state", has_initial_state, n, torch.bool, x.device)
    _disjoint("out", out, cu_seqlens, has_initial_state)
    _disjoint("conv_state", conv_state, cu_seqlens, has_initial_state)
    workspace_size = _prefill_workspace_size(x.shape[0], n)
    if workspace is None and workspace_size:
        workspace = torch.empty(workspace_size, dtype=torch.int32, device=x.device)
    elif workspace is not None and (
        workspace.ndim != 1
        or workspace.numel() < workspace_size
        or workspace.dtype != torch.int32
        or workspace.device != x.device
        or not workspace.is_contiguous()
    ):
        raise ValueError(
            f"workspace must be contiguous CUDA int32 with at least {workspace_size} elements"
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
    """Apply width-four causal convolution and SiLU, updating raw-input history.

    Parameters
    ----------
    x : torch.Tensor
        BF16 CUDA input ``[rows, channels]`` with contiguous channels and
        nonoverlapping, optionally padded rows.
    weight : torch.Tensor
        Contiguous FP32 or BF16 CUDA ``[channels, 4]``. Tap 3 multiplies the
        current token. Accumulation and SiLU use FP32.
    conv_state : torch.Tensor
        In-place BF16 ``[slots, channels, L]``, ``L >= 3``, with nonoverlapping
        slots and either the channel or history axis contiguous. Ordinary
        updates read positions 0:3 and replace them with the final three raw
        inputs/history values; remaining positions are unchanged.
    state_indices : torch.Tensor
        Contiguous CUDA int32 ``[N]``. Live slot IDs must be unique and in
        ``[0, slots)``. The -1 sentinel produces zero output and no state
        access; slot zero is valid.
    cu_seqlens : Optional[torch.Tensor]
        Contiguous CUDA int32 ``[N+1]``. Offsets start at zero, are
        nondecreasing and end at most at rows. Trailing output rows and state
        for empty sequences remain untouched. If omitted, rows must equal N
        and each sequence contains one token.
    num_accepted_tokens : Optional[torch.Tensor]
        Contiguous CUDA int32 ``[N]`` enabling linear speculative updates.
        A positive count a includes the bonus token accepted from the previous
        call and reads history ``state[..., a-1:a+2]``; require ``1 <= a <= L-2``
        and a window written by that call. Nonpositive counts skip the row,
        producing zero output and leaving state unchanged. A current sequence
        of T tokens requires ``T <= L-2`` and writes
        ``[old_h1, old_h2, x0, ..., x(T-1)]`` to positions ``0:T+2``.
    bias : Optional[torch.Tensor]
        Contiguous FP32 or BF16 CUDA ``[channels]``, added before the four FMAs.
    out : Optional[torch.Tensor]
        BF16 storage matching x with contiguous channels and nonoverlapping
        rows. Must not share storage with any operand. Allocated if omitted;
        unused trailing rows of an allocated output are unspecified.

    Returns
    -------
    torch.Tensor
        BF16 output ``[rows, channels]``; conv_state is updated in place.

    Notes
    -----
    Device offset and slot values are caller-owned constraints; no host read
    or validation synchronization occurs. State must not alias other operands.
    CUDA graph replay may change lengths, slots and counts in place within
    fixed capacities. No workspace is needed. An accepted count above L-2
    skips the row. A speculative sequence exceeding L-2 still produces all
    outputs, but stores only the history prefix that fits; it cannot supply
    a complete history for subsequent speculative verification.
    """
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
