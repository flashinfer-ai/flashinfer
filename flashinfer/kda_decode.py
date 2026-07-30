"""
Copyright (c) 2025 by FlashInfer team.

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

"""
Kimi Delta Attention - API Layer
================================

This file provides the public API and dispatch for recurrent KDA decode and
prefill operations.
"""

import math
import threading
from typing import TYPE_CHECKING, Optional

import torch

from .api_logging import flashinfer_api
from .trace.templates.kda import recurrent_kda_trace
from .utils import get_compute_capability

if TYPE_CHECKING:
    from .jit.flash_kda import FlashKDAVariant

try:
    from .kda_kernels.recurrent_kda import run_recurrent_kda as _run_recurrent_kda

    _RECURRENT_KDA_AVAILABLE = True
except (ImportError, RuntimeError):
    _run_recurrent_kda = None
    _RECURRENT_KDA_AVAILABLE = False


_FLASH_KDA_HEAD_DIM = 128
_FLASH_KDA_BETA_TMA_MIN_HEADS = 8
_FLASH_KDA_B200_COMPUTE_CAPABILITY = (10, 0)
_FLASH_KDA_DESCRIPTOR_STORAGE_BYTES = 6 * 128
_flash_kda_tensor_cache: dict[tuple, torch.Tensor] = {}
_flash_kda_tensor_cache_lock = threading.Lock()


class _RecurrentKDAPrefillWorkspaceBase:
    def __init__(self, device: torch.device | str) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError("RecurrentKDAPrefillWorkspace requires a CUDA device")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device
        self._lock = threading.Lock()
        self._state_scratch: Optional[torch.Tensor] = None
        self._beta_padding: Optional[torch.Tensor] = None
        self._descriptor_storages = {
            variant: torch.empty(
                _FLASH_KDA_DESCRIPTOR_STORAGE_BYTES,
                dtype=torch.uint8,
                device=self.device,
            )
            for variant in ("m64", "m128")
        }
        self._descriptor_signatures: dict[str, tuple] = {}
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False


class RecurrentKDAPrefillWorkspace(_RecurrentKDAPrefillWorkspaceBase):
    """Caller-owned storage required for recurrent-KDA CUDA graph capture.

    Construct one workspace per captured :func:`recurrent_kda` invocation on
    the graph's CUDA device. Warm it by invoking :func:`recurrent_kda` eagerly
    with the exact tensors and capture stream, then synchronize that stream
    before capture. The workspace owns optional final-state scratch for calls
    without an initial state, beta padding, and M64/M128 TMA descriptor storage
    for the lifetime of the graph.

    A workspace binds to its first stream. Once it participates in capture it
    cannot be passed to Python again, either eagerly or in another capture.
    Graph replay does not invoke Python and remains valid for the lifetime of
    the workspace.
    """


class _FlashKDAStreamWorkspace(_RecurrentKDAPrefillWorkspaceBase):
    """Internal eager-only workspace for one CUDA stream."""


_flash_kda_stream_workspaces: dict[tuple[int, int], _FlashKDAStreamWorkspace] = {}
_flash_kda_stream_workspaces_lock = threading.Lock()


def _is_plain_multi_token_prefill(
    q: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
) -> bool:
    if num_spec_tokens is not None or not isinstance(q, torch.Tensor) or q.ndim != 4:
        return False
    if cu_seqlens is None:
        return q.shape[1] > 1
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
        return False
    num_sequences = cu_seqlens.numel() - 1
    return num_sequences > 0 and q.shape[1] > num_sequences


def _is_contiguous_cuda_tensor(
    tensor: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.is_contiguous()
    )


def _flash_kda_prefill_is_eligible(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
) -> bool:
    """Return whether the call exactly matches the frozen FlashKDA contract."""

    if not _is_plain_multi_token_prefill(q, cu_seqlens, num_spec_tokens):
        return False
    if (
        ssm_state_indices is not None
        or num_accepted_tokens is not None
        or initial_state_source is not None
        or initial_state_indices is not None
    ):
        return False
    if not (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and beta_is_logit
        and lower_bound is not None
        and math.isfinite(float(lower_bound))
        and float(lower_bound) < 0.0
    ):
        return False
    if (
        not q.is_cuda
        or get_compute_capability(q.device) != _FLASH_KDA_B200_COMPUTE_CAPABILITY
    ):
        return False
    if not _is_contiguous_cuda_tensor(q, dtype=torch.bfloat16, device=q.device):
        return False
    if q.ndim != 4:
        return False
    batch_size, total_or_fixed_tokens, num_heads, head_dim = q.shape
    if (
        batch_size <= 0
        or total_or_fixed_tokens <= 1
        or num_heads <= 0
        or head_dim != _FLASH_KDA_HEAD_DIM
    ):
        return False
    for tensor in (k, v, g):
        if (
            not _is_contiguous_cuda_tensor(
                tensor, dtype=torch.bfloat16, device=q.device
            )
            or tensor.shape != q.shape
        ):
            return False
    if not _is_contiguous_cuda_tensor(
        beta, dtype=torch.bfloat16, device=q.device
    ) or beta.shape != (batch_size, total_or_fixed_tokens, num_heads):
        return False
    if not _is_contiguous_cuda_tensor(
        A_log, dtype=torch.float32, device=q.device
    ) or A_log.shape != (num_heads,):
        return False
    if not _is_contiguous_cuda_tensor(dt_bias, dtype=torch.float32, device=q.device):
        return False
    if dt_bias.numel() != num_heads * _FLASH_KDA_HEAD_DIM or dt_bias.ndim not in (1, 2):
        return False
    if dt_bias.ndim == 2 and dt_bias.shape != (num_heads, _FLASH_KDA_HEAD_DIM):
        return False

    if cu_seqlens is None:
        num_sequences = batch_size
    else:
        if (
            batch_size != 1
            or not cu_seqlens.is_cuda
            or cu_seqlens.device != q.device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences <= 0 or total_or_fixed_tokens <= num_sequences:
            return False

    if initial_state is not None:
        if not _is_contiguous_cuda_tensor(
            initial_state, dtype=torch.bfloat16, device=q.device
        ) or initial_state.shape != (
            num_sequences,
            num_heads,
            _FLASH_KDA_HEAD_DIM,
            _FLASH_KDA_HEAD_DIM,
        ):
            return False
    if output is not None:
        if (
            not _is_contiguous_cuda_tensor(
                output, dtype=torch.bfloat16, device=q.device
            )
            or output.shape != q.shape
        ):
            return False
    return True


def _select_flash_kda_prefill_variant(
    *, fixed_layout: bool, num_sequences: int, num_heads: int
) -> "FlashKDAVariant":
    if fixed_layout and num_sequences == 1 and num_heads == 64:
        return "m64"
    return "m128"


def _cached_tensor(
    key: tuple,
    factory,
    *,
    capture_error: str,
) -> torch.Tensor:
    with _flash_kda_tensor_cache_lock:
        tensor = _flash_kda_tensor_cache.get(key)
        if tensor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(capture_error)
            tensor = factory()
            _flash_kda_tensor_cache[key] = tensor
        return tensor


def _fixed_cu_seqlens(
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    key = ("fixed_cu", *_stream_cache_key(device), batch_size, seq_len)
    return _cached_tensor(
        key,
        lambda: torch.arange(
            0,
            batch_size * seq_len + 1,
            seq_len,
            dtype=torch.int64,
            device=device,
        ),
        capture_error=(
            "fixed-layout recurrent_kda prefill metadata is not warmed for "
            "CUDA graph capture; invoke the same shape once before capture"
        ),
    )


def _identity_seq_order(
    *,
    device: torch.device,
    num_sequences: int,
) -> torch.Tensor:
    key = ("seq_order", *_stream_cache_key(device), num_sequences)
    return _cached_tensor(
        key,
        lambda: torch.arange(num_sequences, dtype=torch.int32, device=device),
        capture_error=(
            "recurrent_kda prefill seq_order is not warmed for CUDA graph "
            "capture; pass a preallocated seq_order or warm the shape first"
        ),
    )


def _dummy_bf16(device: torch.device) -> torch.Tensor:
    key = ("dummy_bf16", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.bfloat16, device=device),
        capture_error=(
            "recurrent_kda prefill dummy state is not warmed for CUDA graph "
            "capture; invoke the same device once before capture"
        ),
    )


def _stream_cache_key(device: torch.device) -> tuple[int, int]:
    stream = torch.cuda.current_stream(device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return device_index, int(stream.cuda_stream)


def _get_stream_workspace(device: torch.device) -> _FlashKDAStreamWorkspace:
    key = _stream_cache_key(device)
    with _flash_kda_stream_workspaces_lock:
        workspace = _flash_kda_stream_workspaces.get(key)
        if workspace is None:
            workspace = _FlashKDAStreamWorkspace(device)
            _flash_kda_stream_workspaces[key] = workspace
        return workspace


def _workspace_buffer(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    attribute: str,
    device: torch.device,
    numel: int,
    capture_error: str,
) -> torch.Tensor:
    buffer = getattr(workspace, attribute)
    capturing = torch.cuda.is_current_stream_capturing()
    if buffer is None or buffer.numel() < numel:
        if capturing:
            raise RuntimeError(capture_error)
        buffer = torch.empty(numel, dtype=torch.bfloat16, device=device)
        setattr(workspace, attribute, buffer)
    return buffer[:numel]


def _state_scratch(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    numel = math.prod(shape)
    return _workspace_buffer(
        workspace=workspace,
        attribute="_state_scratch",
        device=device,
        numel=numel,
        capture_error=(
            "recurrent_kda prefill final-state workspace is not large enough "
            "for CUDA graph capture; warm the largest shape on this stream "
            "before capture"
        ),
    ).view(shape)


def _beta_tma_source(
    beta: torch.Tensor,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
) -> torch.Tensor:
    batch_size, seq_len, num_heads = beta.shape
    total_tokens = batch_size * seq_len
    beta_flat = beta.reshape(total_tokens, num_heads)
    padded_tokens = max(total_tokens, 32)
    padded_heads = max(num_heads, _FLASH_KDA_BETA_TMA_MIN_HEADS)
    if padded_tokens == total_tokens and padded_heads == num_heads:
        return beta_flat
    shape = (padded_tokens, padded_heads)
    padded = _workspace_buffer(
        workspace=workspace,
        attribute="_beta_padding",
        device=beta.device,
        numel=math.prod(shape),
        capture_error=(
            "recurrent_kda prefill beta TMA workspace is not large enough for "
            "CUDA graph capture; warm the largest padded token/head shape on "
            "this stream before capture"
        ),
    ).view(shape)
    # The descriptor may fetch a full (32 token, 8 head) box at the tail.
    # Clear reused padding so every legal OOB-adjacent fetch has defined data.
    padded.zero_()
    padded[:total_tokens, :num_heads].copy_(beta_flat)
    return padded


def _tensor_descriptor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _descriptor_signature(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_tma: torch.Tensor,
    out: torch.Tensor,
) -> tuple:
    return tuple(
        _tensor_descriptor_signature(tensor) for tensor in (q, k, v, g, beta_tma, out)
    )


def _bind_workspace(
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    *,
    device: torch.device,
    stream_ptr: int,
    capturing: bool,
    explicit: bool,
) -> None:
    if workspace.device != device:
        raise ValueError(
            "RecurrentKDAPrefillWorkspace is bound to "
            f"{workspace.device}, but recurrent_kda inputs are on {device}"
        )
    if workspace._bound_stream_ptr is None:
        workspace._bound_stream_ptr = stream_ptr
    elif workspace._bound_stream_ptr != stream_ptr:
        raise RuntimeError(
            "RecurrentKDAPrefillWorkspace is bound to a different CUDA "
            "stream; warm and capture it on one stream"
        )
    if explicit and workspace._captured:
        reuse_kind = "captured by another CUDA graph" if capturing else "reused eagerly"
        raise RuntimeError(
            "RecurrentKDAPrefillWorkspace has participated in CUDA graph "
            f"capture and cannot be {reuse_kind} or mutated"
        )


def _storage_ranges_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    left_start = left.data_ptr()
    right_start = right.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def _check_output_does_not_overlap_inputs(
    output: torch.Tensor,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> None:
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", initial_state),
    ):
        if tensor is not None and _storage_ranges_overlap(output, tensor):
            raise ValueError(
                f"output must not overlap {name} for frozen recurrent_kda prefill"
            )


def _validate_prefill_seq_order(
    seq_order: Optional[torch.Tensor],
    *,
    fixed_layout: bool,
    num_sequences: int,
    device: torch.device,
) -> torch.Tensor:
    if seq_order is None:
        return _identity_seq_order(device=device, num_sequences=num_sequences)
    if fixed_layout:
        raise ValueError("seq_order is only supported for packed recurrent_kda prefill")
    if not isinstance(seq_order, torch.Tensor):
        raise TypeError("seq_order must be a torch.Tensor")
    if (
        not seq_order.is_cuda
        or seq_order.device != device
        or seq_order.dtype != torch.int32
        or seq_order.ndim != 1
        or not seq_order.is_contiguous()
        or seq_order.numel() != num_sequences
    ):
        raise ValueError(
            "seq_order must be a contiguous CUDA int32 tensor with one "
            f"entry per sequence ({num_sequences})"
        )
    return seq_order


def _get_flash_kda_prefill_module(variant: "FlashKDAVariant"):
    from .jit.flash_kda import get_flash_kda_prefill_module

    return get_flash_kda_prefill_module(variant)


def _run_flash_kda_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    lower_bound: float,
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and prefill_workspace is None:
        raise RuntimeError(
            "CUDA graph capture of recurrent_kda prefill requires an explicit "
            "RecurrentKDAPrefillWorkspace warmed with the exact tensors on "
            "the capture stream"
        )
    batch_size, seq_len, num_heads, _ = q.shape
    fixed_layout = cu_seqlens is None
    num_sequences = batch_size if fixed_layout else cu_seqlens.numel() - 1
    if fixed_layout:
        cu_seqlens_i64 = _fixed_cu_seqlens(
            device=q.device, batch_size=batch_size, seq_len=seq_len
        )
    else:
        assert cu_seqlens is not None
        if cu_seqlens.dtype == torch.int32 and capturing:
            raise RuntimeError(
                "packed recurrent_kda prefill requires int64 cu_seqlens "
                "during CUDA graph capture; convert it before capture"
            )
        cu_seqlens_i64 = (
            cu_seqlens
            if cu_seqlens.dtype == torch.int64
            else cu_seqlens.to(torch.int64)
        )
    seq_order_i32 = _validate_prefill_seq_order(
        seq_order,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        device=q.device,
    )
    dummy_state = _dummy_bf16(q.device)

    if output is None:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires a preallocated output tensor for "
                "recurrent_kda prefill"
            )
        out_buf = torch.empty_like(q)
    else:
        out_buf = output
    _check_output_does_not_overlap_inputs(
        out_buf,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
    )

    state_shape = (
        num_sequences,
        num_heads,
        _FLASH_KDA_HEAD_DIM,
        _FLASH_KDA_HEAD_DIM,
    )
    use_initial_state = initial_state is not None
    if initial_state is not None:
        initial_state_arg = initial_state
        final_state_arg = initial_state
        store_final_state = True
        returned_state = initial_state
    elif output_final_state:
        initial_state_arg = dummy_state
        if prefill_workspace is None:
            final_state_arg = torch.empty(
                state_shape, dtype=torch.bfloat16, device=q.device
            )
            returned_state = final_state_arg
        else:
            # Assigned to caller-owned stable state scratch under its lock.
            returned_state = None
        store_final_state = True
    else:
        initial_state_arg = dummy_state
        final_state_arg = dummy_state
        store_final_state = False
        returned_state = None

    scale_value = (
        1.0 / math.sqrt(_FLASH_KDA_HEAD_DIM) if scale is None else float(scale)
    )
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")
    variant = _select_flash_kda_prefill_variant(
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
    )
    stream_ptr = int(torch.cuda.current_stream(q.device).cuda_stream)
    explicit_workspace = prefill_workspace is not None
    workspace: _RecurrentKDAPrefillWorkspaceBase
    if prefill_workspace is None:
        workspace = _get_stream_workspace(q.device)
    else:
        workspace = prefill_workspace
    # TVM FFI may release the GIL. Serialize the complete shared-workspace
    # enqueue sequence so two host threads cannot interleave preparation or
    # launch on the same CUDA stream.
    with workspace._lock:
        _bind_workspace(
            workspace,
            device=q.device,
            stream_ptr=stream_ptr,
            capturing=capturing,
            explicit=explicit_workspace,
        )
        beta_tma = _beta_tma_source(beta, workspace)
        if initial_state is None and output_final_state and explicit_workspace:
            final_state_arg = _state_scratch(
                workspace=workspace,
                device=q.device,
                shape=state_shape,
            )
            if initial_state is None:
                returned_state = final_state_arg
        signature = _descriptor_signature(
            q=q,
            k=k,
            v=v,
            g=g,
            beta_tma=beta_tma,
            out=out_buf,
        )
        warmed_signature = workspace._descriptor_signatures.get(variant)
        if capturing:
            if warmed_signature != signature:
                raise RuntimeError(
                    "RecurrentKDAPrefillWorkspace is not warmed for the exact "
                    f"{variant} descriptor signature; eagerly invoke the same "
                    "call on this stream before capture"
                )
            prepare_descriptors = 0
        else:
            prepare_descriptors = int(warmed_signature != signature)
        descriptor_storage = workspace._descriptor_storages[variant]
        module = _get_flash_kda_prefill_module(variant)
        try:
            module.run(
                q,
                k,
                v,
                g,
                beta,
                beta_tma,
                A_log,
                dt_bias,
                cu_seqlens_i64,
                seq_order_i32,
                initial_state_arg,
                out_buf,
                final_state_arg,
                descriptor_storage,
                prepare_descriptors,
                num_heads,
                int(use_initial_state),
                int(store_final_state),
                scale_value,
                float(lower_bound),
                stream_ptr,
            )
        except Exception:
            if prepare_descriptors:
                workspace._descriptor_signatures.pop(variant, None)
            raise
        if prepare_descriptors:
            workspace._descriptor_signatures[variant] = signature
        if capturing and explicit_workspace:
            workspace._captured = True
    return (
        out_buf,
        returned_state if output_final_state else None,
    )


@flashinfer_api(trace=recurrent_kda_trace)
def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_spec_tokens: Optional[int] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    seq_order: Optional[torch.Tensor] = None,
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Recurrent KDA (Kimi Delta Attention) decode and prefill kernel.

    This is the public API layer for the CuTe DSL implementation in
    ``flashinfer.kda_kernels.recurrent_kda``. It supports single-token decode,
    fused speculative decode, GQA, optional cu_seqlens packing, and the same
    gate modes as the backend implementation. On NVIDIA B200, the exact
    FlashKDA-compatible subset of ordinary multi-token prefill is dispatched to
    frozen SM100a kernels. All existing decode and speculative-decode calls
    retain the CuTe DSL backend.

    Args:
        q (torch.Tensor):
            Query of shape ``[B, T, H, K]``, or
            ``[1, total_tokens, H, K]`` when using ``cu_seqlens``. Must be
            bfloat16. ``T=1`` selects decode; eligible ``T>1`` calls may select
            the frozen prefill backend.
        k (torch.Tensor):
            Key with the same shape as ``q``. Must be bfloat16.
        v (torch.Tensor):
            Value of shape ``[B, T, HV, V]``, or
            ``[1, total_tokens, HV, V]`` when packed. Must be bfloat16. GQA is
            applied when ``HV != H``.
        g (torch.Tensor):
            Per-K-dimension gate of shape ``[B, T, HV, K]``, or
            ``[1, total_tokens, HV, K]`` when packed. Must be bfloat16.
            Log-space if pre-computed, raw input if
            ``use_gate_in_kernel=True``.
        beta (torch.Tensor):
            Delta-rule learning rate of shape ``[B, T, HV]``, or
            ``[1, total_tokens, HV]`` when packed. Must be bfloat16.
            Pre-sigmoided unless ``beta_is_logit=True``.
        A_log (Optional[torch.Tensor]):
            Log decay parameter of shape ``[H]``. Must be float32.
            Required when ``use_gate_in_kernel=True``.
        dt_bias (Optional[torch.Tensor]):
            Per-head-K decay bias of shape ``[H*K]`` or ``[H, K]``. Must be
            float32.
        scale (Optional[float]):
            Scale factor for queries. If ``None``, defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[N, HV, V, K]``. Must be bfloat16.
            If ``None``, zero-initialized. Updated in-place. For batched spec
            decode without ``cu_seqlens``, ``N`` is the packed checkpoint-slot
            count ``B * (1 + num_spec_tokens)`` when ``ssm_state_indices`` is
            omitted.
        output_final_state (bool):
            Whether to return the final state. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2 normalization to Q and K. Default: ``True``.
        use_gate_in_kernel (bool):
            Whether to compute the gate inside the kernel from ``A_log`` and
            ``g``. Default: ``False``.
        lower_bound (Optional[float]):
            If set, uses ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
            gate formula instead of softplus. Must be negative.
        cu_seqlens (Optional[torch.Tensor]):
            Contiguous CUDA cumulative sequence lengths of shape ``[N+1]``.
            May be int32 or int64. Frozen prefill converts int32 offsets to
            int64 outside graph capture; graph capture requires caller-provided
            int64 offsets. For frozen prefill, values must start at zero, be
            strictly increasing, and end at the total token count. This value
            contract is not host-validated to avoid a device synchronization.
        ssm_state_indices (Optional[torch.Tensor]):
            State cache indices. Shape ``[N]`` int32 for standard decode, or
            ``[N, 1+S]`` int32 for spec decode (``num_spec_tokens`` must also
            be set).
        num_spec_tokens (Optional[int]):
            Number of speculative tokens (S). When set, processes 1+S tokens in
            a single fused kernel launch. Must be >= 1.
        num_accepted_tokens (Optional[torch.Tensor]):
            Per-sequence accepted token count from the previous spec decode
            round. Shape ``[N]`` int32. If ``None``, initial state is loaded
            from ``ssm_state_indices[n, 0]``. Values above ``1+S`` are clamped
            to the final checkpoint slot.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor. Shape ``[B, T, HV, V]`` for fixed
            layout, or the corresponding packed/speculative shape when using
            ``cu_seqlens``. If ``None``, a new tensor is allocated. Frozen
            prefill requires storage disjoint from Q, K, V, G, beta, and
            ``initial_state``.
        initial_state_source (Optional[torch.Tensor]):
            Optional read-only committed state pool ``[N0, HV, V, K]``. When
            provided, token 0 is loaded from this pool instead of
            ``initial_state``.
        initial_state_indices (Optional[torch.Tensor]):
            Source slot per sequence, shape ``[N]`` int32. Required together
            with ``initial_state_source``.
        beta_is_logit (bool):
            If ``True``, apply sigmoid to ``beta`` inside the recurrent kernel.
        seq_order (Optional[torch.Tensor]):
            Optional packed-prefill sequence order, as a contiguous CUDA int32
            permutation of shape ``[N]``. Sorting by descending sequence length
            improves tail utilization. It is only consumed by the frozen
            FlashKDA prefill backend; prepare it before CUDA graph capture or
            timed launches. Fixed-layout prefill and decode calls must leave it
            as ``None``.
        prefill_workspace (Optional[RecurrentKDAPrefillWorkspace]):
            Caller-owned workspace for the frozen B200 prefill backend. It is
            optional for eager execution and required for CUDA graph capture.
            Warm it eagerly with the exact tensors on the capture stream before
            capture. Use one workspace per captured ``recurrent_kda``
            invocation.

    Returns:
        Tuple of ``(output, final_state)`` where ``final_state`` is ``None``
        when ``output_final_state=False``. See
        :func:`flashinfer.kda_kernels.recurrent_kda.run_recurrent_kda` for the
        backend implementation.
    """
    if prefill_workspace is not None and not isinstance(
        prefill_workspace, RecurrentKDAPrefillWorkspace
    ):
        raise TypeError("prefill_workspace must be a RecurrentKDAPrefillWorkspace")

    use_flash_kda_prefill = _flash_kda_prefill_is_eligible(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=num_accepted_tokens,
        output=output,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        beta_is_logit=beta_is_logit,
    )
    if use_flash_kda_prefill:
        assert A_log is not None
        assert dt_bias is not None
        assert lower_bound is not None
        return _run_flash_kda_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            output=output,
            seq_order=seq_order,
            prefill_workspace=prefill_workspace,
        )

    if prefill_workspace is not None:
        raise ValueError(
            "prefill_workspace is only supported by eligible ordinary "
            "prefill on the frozen B200 FlashKDA backend"
        )
    if seq_order is not None:
        raise ValueError(
            "seq_order is only supported by eligible packed ordinary prefill "
            "on the frozen B200 FlashKDA backend"
        )
    if _run_recurrent_kda is None:
        raise NotImplementedError("recurrent KDA backend is unavailable")

    return _run_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=num_accepted_tokens,
        output=output,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        beta_is_logit=beta_is_logit,
    )
