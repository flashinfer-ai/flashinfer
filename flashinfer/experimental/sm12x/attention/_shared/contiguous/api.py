# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/contiguous/api.py @ 149d6bb2 (2026-06-12) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Planner-backed public attention entrypoints for contiguous attention."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32

from flashinfer.experimental.sm12x.attention._shared.contiguous.forward import (
    ContiguousAttentionForwardKernel,
)
from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    compile as sm12x_compile,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream, make_ptr
from flashinfer.experimental.sm12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)


_ARENA_ALIGN_BYTES = 1024


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise TypeError(
        f"unsupported dtype {dtype}; expected torch.bfloat16 or torch.float16"
    )


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for idx in range(len(shape) - 1, -1, -1):
        stride[idx] = running
        running *= shape[idx]
    return tuple(stride)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


def _lse_shape(q_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(q_shape) == 3:
        seqlen_q, q_heads, _ = q_shape
        return (q_heads, seqlen_q)
    batch, seqlen_q, q_heads, _ = q_shape
    return (batch, q_heads, seqlen_q)


def _seq_dims(shape: tuple[int, ...]) -> tuple[tuple[int, ...], int, int, int]:
    if len(shape) == 3:
        seqlen, num_heads, head_dim = shape
        return (), seqlen, num_heads, head_dim
    if len(shape) == 4:
        batch, seqlen, num_heads, head_dim = shape
        return (batch,), seqlen, num_heads, head_dim
    raise ValueError(f"expected rank-3 or rank-4 tensor shape, got {shape}")


def _normalize_window_size(
    window_size: int | tuple[int, int] | None,
) -> tuple[int, int]:
    if window_size is None:
        return (-1, -1)
    if isinstance(window_size, int):
        left = right = int(window_size)
    else:
        if len(window_size) != 2:
            raise ValueError(f"window_size must have two elements, got {window_size}")
        left, right = (int(window_size[0]), int(window_size[1]))
    if left < -1 or right < -1:
        raise ValueError(f"window_size values must be >= -1, got {(left, right)}")
    return left, right


def _resolve_max_seqlen(
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None,
    *,
    name: str,
) -> int:
    if max_seqlen is not None:
        resolved = int(max_seqlen)
        if resolved < 0:
            raise ValueError(f"{name} must be non-negative, got {resolved}")
        return resolved
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu()
    return int(lengths.max().item()) if lengths.numel() else 0


def _attention_logical_dims(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
) -> tuple[int, int, int, int, int, int, int, int]:
    batch_dims, seqlen_q, q_heads, _ = _seq_dims(q_shape)
    _, seqlen_k, kv_heads, _ = _seq_dims(k_shape)
    num_batch = batch_dims[0] if batch_dims else 1
    qhead_per_kvhead = q_heads // kv_heads
    logical_q_rows_static = seqlen_q * qhead_per_kvhead
    logical_total_q_rows = logical_q_rows_static * num_batch
    return (
        num_batch,
        q_heads,
        kv_heads,
        qhead_per_kvhead,
        seqlen_q,
        seqlen_k,
        logical_q_rows_static,
        logical_total_q_rows,
    )


def _varlen_attention_logical_dims(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[int, int, int, int, int, int]:
    if len(q_shape) != 3 or len(k_shape) != 3:
        raise ValueError("varlen attention expects packed rank-3 q/k/v tensors")
    _, q_heads, _ = q_shape
    total_q, _, _ = q_shape
    _, kv_heads, _ = k_shape
    num_batch = cu_seqlens_q_shape[0] - 1
    qhead_per_kvhead = q_heads // kv_heads
    logical_q_rows_static = max_seqlen_q * qhead_per_kvhead
    logical_total_q_rows = total_q * qhead_per_kvhead
    return (
        num_batch,
        q_heads,
        kv_heads,
        qhead_per_kvhead,
        logical_q_rows_static,
        logical_total_q_rows,
    )


def _select_tile_shape(head_dim: int, *, causal: bool) -> tuple[int, int]:
    if head_dim <= 64:
        return (128, 128)
    if head_dim <= 128:
        return (128, 64)
    if head_dim == 256:
        return (64, 32 if causal else 48)
    raise ValueError(
        f"unsupported head_dim={head_dim} for the current sm12x attention path"
    )


def _normalize_tensor_shape(t: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in t.shape)


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"expected CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


@functools.cache
def _attention_sink_placeholder(device_index: int) -> torch.Tensor:
    return torch.empty(
        (1,), dtype=torch.float32, device=torch.device("cuda", device_index)
    )


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.device, torch.dtype
]:
    if q.ndim not in (3, 4):
        raise ValueError(f"q must be rank-3 or rank-4, got rank {q.ndim}")
    if q.ndim != k.ndim or q.ndim != v.ndim:
        raise ValueError("q, k, and v must have the same rank")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must all be CUDA tensors on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"unsupported dtype {q.dtype}; expected torch.bfloat16 or torch.float16"
        )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, and v must all be contiguous")

    q_shape = _normalize_tensor_shape(q)
    k_shape = _normalize_tensor_shape(k)
    v_shape = _normalize_tensor_shape(v)
    batch_q, _, q_heads, q_head_dim = _seq_dims(q_shape)
    batch_k, _, kv_heads, k_head_dim = _seq_dims(k_shape)
    batch_v, _, v_heads, v_head_dim = _seq_dims(v_shape)
    if batch_q != batch_k or batch_q != batch_v:
        raise ValueError("q, k, and v must have matching batch dimensions")
    if q_head_dim != k_head_dim or q_head_dim != v_head_dim:
        raise ValueError(
            "q, k, and v must have matching head dimensions in the initial path"
        )
    if kv_heads != v_heads:
        raise ValueError("k and v must have the same number of KV heads")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
    return q_shape, k_shape, v_shape, q.device, q.dtype


def _prepare_attention_sink_bias(
    attention_sink_bias: torch.Tensor | None,
    *,
    q_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor | None:
    if attention_sink_bias is None:
        return None
    _, _, q_heads, _ = _seq_dims(q_shape)
    if attention_sink_bias.ndim != 1:
        raise ValueError(
            "attention_sink_bias must be rank-1 [num_q_heads], "
            f"got {tuple(attention_sink_bias.shape)}"
        )
    if int(attention_sink_bias.shape[0]) != q_heads:
        raise ValueError(
            f"attention_sink_bias must have {q_heads} elements, "
            f"got {int(attention_sink_bias.shape[0])}"
        )
    if attention_sink_bias.device != device:
        raise ValueError("attention_sink_bias must be on the same CUDA device as q")
    if attention_sink_bias.dtype != torch.float32:
        attention_sink_bias = attention_sink_bias.to(torch.float32)
    if not attention_sink_bias.is_contiguous():
        attention_sink_bias = attention_sink_bias.contiguous()
    return attention_sink_bias


def _validate_attention_inputs_against_plan(
    *,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    plan: AttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_shape,
        plan.v_shape,
        plan.device,
        plan.dtype,
    )
    actual = (q_shape, k_shape, v_shape, device, dtype)
    if expected != actual:
        raise ValueError(
            "attention plan mismatch: "
            f"expected q/k/v/device/dtype={expected}, got {actual}"
        )


def _validate_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    torch.device,
    torch.dtype,
]:
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    if len(q_shape) != 3:
        raise ValueError("varlen attention expects packed rank-3 q/k/v tensors")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1 tensors")
    if cu_seqlens_q.device != device or cu_seqlens_k.device != device:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must be CUDA tensors on the input device"
        )
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.int32")
    if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be contiguous")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError(
            "the restored varlen contiguous path currently requires matching q/k segment counts"
        )
    return (
        q_shape,
        k_shape,
        v_shape,
        _normalize_tensor_shape(cu_seqlens_q),
        _normalize_tensor_shape(cu_seqlens_k),
        device,
        dtype,
    )


def _validate_varlen_inputs_against_plan(
    *,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    cu_seqlens_k_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    plan: VarlenAttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_shape,
        plan.v_shape,
        plan.cu_seqlens_q_shape,
        plan.cu_seqlens_k_shape,
        plan.device,
        plan.dtype,
    )
    actual = (
        q_shape,
        k_shape,
        v_shape,
        cu_seqlens_q_shape,
        cu_seqlens_k_shape,
        device,
        dtype,
    )
    if expected != actual:
        raise ValueError(
            "varlen attention plan mismatch: "
            "expected q/k/v/cu_q/cu_k/device/dtype="
            f"{expected}, got {actual}"
        )


@dataclass(frozen=True, kw_only=True)
class AttentionPlanKey:
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    device_index: int
    dtype: torch.dtype
    causal: bool
    window_size_left: int
    window_size_right: int
    has_attention_sink_bias: bool
    tile_m: int
    tile_n: int
    num_batch: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    seqlen_q_static: int
    seqlen_k_static: int
    logical_q_rows_static: int
    logical_total_q_rows: int


@dataclass(frozen=True, kw_only=True)
class VarlenAttentionPlanKey:
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    cu_seqlens_q_shape: tuple[int, ...]
    cu_seqlens_k_shape: tuple[int, ...]
    device_index: int
    dtype: torch.dtype
    causal: bool
    window_size_left: int
    window_size_right: int
    has_attention_sink_bias: bool
    tile_m: int
    tile_n: int
    max_seqlen_q: int
    max_seqlen_k: int
    num_batch: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    logical_q_rows_static: int
    logical_total_q_rows: int


@dataclass(frozen=True, kw_only=True)
class AttentionPlan:
    """Exact-shape launch contract for one contiguous attention shape."""

    key: AttentionPlanKey
    compiled: object = field(repr=False, compare=False)
    cutlass_dtype: type[cutlass.Numeric] = field(repr=False, compare=False)

    def __getattr__(self, name: str):
        return getattr(self.key, name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.key.device_index)


@dataclass(frozen=True, kw_only=True)
class VarlenAttentionPlan:
    """Exact-shape launch contract for one packed varlen contiguous attention shape."""

    key: VarlenAttentionPlanKey
    compiled: object = field(repr=False, compare=False)
    cutlass_dtype: type[cutlass.Numeric] = field(repr=False, compare=False)

    def __getattr__(self, name: str):
        return getattr(self.key, name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.key.device_index)


@dataclass(kw_only=True)
class AttentionWorkspace:
    """Reusable exact-shape output buffers for one contiguous attention plan."""

    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    causal: bool
    window_size_left: int
    window_size_right: int
    has_attention_sink_bias: bool
    tile_m: int
    tile_n: int
    output: torch.Tensor
    lse: torch.Tensor
    plan_key: AttentionPlanKey | None = None

    def bind(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: AttentionPlan | None = None,
        softmax_scale: float | None = None,
        attention_sink_bias: torch.Tensor | None = None,
    ) -> "AttentionBinding":
        return build_attention_binding(
            workspace=self,
            q=q,
            k=k,
            v=v,
            plan=plan,
            softmax_scale=softmax_scale,
            attention_sink_bias=attention_sink_bias,
        )


@dataclass(kw_only=True)
class VarlenAttentionWorkspace:
    """Reusable output buffers for one packed varlen contiguous attention plan."""

    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    cu_seqlens_q_shape: tuple[int, ...]
    cu_seqlens_k_shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    causal: bool
    window_size_left: int
    window_size_right: int
    has_attention_sink_bias: bool
    tile_m: int
    tile_n: int
    max_seqlen_q: int
    max_seqlen_k: int
    output: torch.Tensor
    lse: torch.Tensor
    plan_key: VarlenAttentionPlanKey | None = None

    def bind(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor | None = None,
        plan: VarlenAttentionPlan | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        softmax_scale: float | None = None,
        causal: bool | None = None,
        window_size: int | tuple[int, int] | None = None,
        attention_sink_bias: torch.Tensor | None = None,
    ) -> "VarlenAttentionBinding":
        return build_varlen_attention_binding(
            workspace=self,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            plan=plan,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_sink_bias=attention_sink_bias,
        )


@dataclass
class AttentionWorkspacePool:
    """Caller-owned exact-shape workspace cache partitioned by CUDA stream."""

    workspaces: dict[tuple[int, AttentionPlanKey], AttentionWorkspace] = field(
        default_factory=dict
    )

    def clear(self) -> None:
        self.workspaces.clear()

    def bind(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: AttentionPlan,
        softmax_scale: float | None = None,
        attention_sink_bias: torch.Tensor | None = None,
    ) -> "AttentionBinding":
        return build_attention_binding(
            workspace=self,
            q=q,
            k=k,
            v=v,
            plan=plan,
            softmax_scale=softmax_scale,
            attention_sink_bias=attention_sink_bias,
        )


@dataclass(frozen=True, kw_only=True)
class AttentionBinding:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    plan: AttentionPlan
    softmax_scale: float | None = None
    attention_sink_bias: torch.Tensor | None = None

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        return sm12x_attention_forward(binding=self)


@dataclass(frozen=True, kw_only=True)
class VarlenAttentionBinding:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    plan: VarlenAttentionPlan
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    softmax_scale: float | None = None
    causal: bool | None = None
    window_size: int | tuple[int, int] | None = None
    attention_sink_bias: torch.Tensor | None = None

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        return sm12x_varlen_attention_forward(binding=self)


@dataclass(frozen=True, kw_only=True)
class _AttentionScratchLayout:
    nbytes: int
    output_offset_bytes: int
    lse_offset_bytes: int


@dataclass(frozen=True)
class _AttentionScratchViews:
    output: torch.Tensor
    lse: torch.Tensor


@dataclass(frozen=True)
class AttentionScratchPlan:
    plan: AttentionPlan
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    _layout: _AttentionScratchLayout

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        attention_sink_bias: torch.Tensor | None = None,
    ) -> AttentionBinding:
        views = _attention_scratch_views_from_scratch_plan(
            self,
            scratch=scratch,
        )
        return _build_attention_binding_from_views(
            output=views.output,
            lse=views.lse,
            q=q,
            k=k,
            v=v,
            plan=self.plan,
            softmax_scale=softmax_scale,
            attention_sink_bias=attention_sink_bias,
        )


@dataclass(frozen=True)
class VarlenAttentionScratchPlan:
    plan: VarlenAttentionPlan
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    _layout: _AttentionScratchLayout

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        softmax_scale: float | None = None,
        causal: bool | None = None,
        window_size: int | tuple[int, int] | None = None,
        attention_sink_bias: torch.Tensor | None = None,
    ) -> VarlenAttentionBinding:
        views = _varlen_attention_scratch_views_from_scratch_plan(
            self,
            scratch=scratch,
        )
        return _build_varlen_attention_binding_from_views(
            output=views.output,
            lse=views.lse,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            plan=self.plan,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_sink_bias=attention_sink_bias,
        )


class _AttentionForwardLaunch:
    def __init__(
        self,
        *,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        dtype: torch.dtype,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        has_attention_sink_bias: bool,
        tile_m: int,
        tile_n: int,
    ):
        self._q_shape = q_shape
        self._k_shape = k_shape
        self._v_shape = v_shape
        self._o_shape = q_shape
        self._lse_shape = _lse_shape(q_shape)
        self._attention_sink_bias_shape = (q_shape[-2],)
        self._q_stride = _contiguous_stride(q_shape)
        self._k_stride = _contiguous_stride(k_shape)
        self._v_stride = _contiguous_stride(v_shape)
        self._o_stride = _contiguous_stride(q_shape)
        self._lse_stride = _contiguous_stride(self._lse_shape)
        self._attention_sink_bias_stride = _contiguous_stride(
            self._attention_sink_bias_shape
        )
        self._dtype = _torch_to_cutlass_dtype(dtype)
        self._window_size_left = window_size_left
        self._window_size_right = window_size_right
        self._has_attention_sink_bias = bool(has_attention_sink_bias)
        is_local = window_size_left != -1 or window_size_right != -1
        (
            self._num_batch,
            q_heads,
            kv_heads,
            qhead_per_kvhead,
            self._seqlen_q_static,
            self._seqlen_k_static,
            self._logical_q_rows_static,
            self._logical_total_q_rows,
        ) = _attention_logical_dims(q_shape, k_shape)
        _, _, _, head_dim = _seq_dims(q_shape)
        _, _, _, head_dim_k = _seq_dims(k_shape)
        _, _, _, head_dim_v = _seq_dims(v_shape)
        if not ContiguousAttentionForwardKernel.can_implement(
            self._dtype,
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            1,
            160,
            causal,
        ):
            raise TypeError(
                "sm12x attention launch is unsupported with "
                f"dtype={dtype}, q_shape={q_shape}, k_shape={k_shape}, v_shape={v_shape}, "
                f"causal={causal}, tile=({tile_m}, {tile_n})"
            )
        self._kernel = ContiguousAttentionForwardKernel(
            self._dtype,
            head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=is_local,
            pack_gqa=qhead_per_kvhead != 1,
            tile_m=tile_m,
            tile_n=tile_n,
        )
        assert head_dim == head_dim_k

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        attention_sink_bias_ptr: cute.Pointer,
        softmax_scale: float,
        current_stream: cuda.CUstream,
    ):
        q_tensor = cute.make_tensor(
            q_ptr, layout=cute.make_layout(self._q_shape, stride=self._q_stride)
        )
        k_tensor = cute.make_tensor(
            k_ptr, layout=cute.make_layout(self._k_shape, stride=self._k_stride)
        )
        v_tensor = cute.make_tensor(
            v_ptr, layout=cute.make_layout(self._v_shape, stride=self._v_stride)
        )
        o_tensor = cute.make_tensor(
            o_ptr, layout=cute.make_layout(self._o_shape, stride=self._o_stride)
        )
        lse_tensor = cute.make_tensor(
            lse_ptr,
            layout=cute.make_layout(self._lse_shape, stride=self._lse_stride),
        )
        attention_sink_bias_tensor = cute.make_tensor(
            attention_sink_bias_ptr,
            layout=cute.make_layout(
                self._attention_sink_bias_shape,
                stride=self._attention_sink_bias_stride,
            ),
        )
        self._kernel(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            learnable_sink=attention_sink_bias_tensor,
            has_attention_sink_bias=self._has_attention_sink_bias,
            logical_num_batch_static=self._num_batch,
            logical_seqlen_q_static=self._seqlen_q_static,
            logical_seqlen_k_static=self._seqlen_k_static,
            window_size_left=(
                None if self._window_size_left == -1 else Int32(self._window_size_left)
            ),
            window_size_right=(
                None
                if self._window_size_right == -1
                else Int32(self._window_size_right)
            ),
            stream=current_stream,
        )


class _VarlenAttentionForwardLaunch:
    def __init__(
        self,
        *,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
        cu_seqlens_k_shape: tuple[int, ...],
        dtype: torch.dtype,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        has_attention_sink_bias: bool,
        max_seqlen_q: int,
        max_seqlen_k: int,
        tile_m: int,
        tile_n: int,
    ):
        self._q_shape = q_shape
        self._k_shape = k_shape
        self._v_shape = v_shape
        self._cu_seqlens_q_shape = cu_seqlens_q_shape
        self._cu_seqlens_k_shape = cu_seqlens_k_shape
        self._o_shape = q_shape
        self._lse_shape = _lse_shape(q_shape)
        self._attention_sink_bias_shape = (q_shape[-2],)
        self._q_stride = _contiguous_stride(q_shape)
        self._k_stride = _contiguous_stride(k_shape)
        self._v_stride = _contiguous_stride(v_shape)
        self._cu_seqlens_q_stride = _contiguous_stride(cu_seqlens_q_shape)
        self._cu_seqlens_k_stride = _contiguous_stride(cu_seqlens_k_shape)
        self._o_stride = _contiguous_stride(q_shape)
        self._lse_stride = _contiguous_stride(self._lse_shape)
        self._attention_sink_bias_stride = _contiguous_stride(
            self._attention_sink_bias_shape
        )
        self._dtype = _torch_to_cutlass_dtype(dtype)
        self._window_size_left = window_size_left
        self._window_size_right = window_size_right
        self._has_attention_sink_bias = bool(has_attention_sink_bias)
        self._max_seqlen_q = max_seqlen_q
        self._max_seqlen_k = max_seqlen_k
        is_local = window_size_left != -1 or window_size_right != -1
        (
            self._num_batch,
            q_heads,
            kv_heads,
            qhead_per_kvhead,
            self._logical_q_rows_static,
            self._logical_total_q_rows,
        ) = _varlen_attention_logical_dims(
            q_shape,
            k_shape,
            cu_seqlens_q_shape,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        _, _, head_dim = q_shape
        _, _, head_dim_k = k_shape
        _, _, head_dim_v = v_shape
        if not ContiguousAttentionForwardKernel.can_implement(
            self._dtype,
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            1,
            160,
            causal,
        ):
            raise TypeError(
                "sm12x varlen attention launch is unsupported with "
                f"dtype={dtype}, q_shape={q_shape}, k_shape={k_shape}, v_shape={v_shape}, "
                f"causal={causal}, window=({window_size_left}, {window_size_right}), "
                f"max_seqlen=({max_seqlen_q}, {max_seqlen_k}), tile=({tile_m}, {tile_n})"
            )
        self._kernel = ContiguousAttentionForwardKernel(
            self._dtype,
            head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=is_local,
            pack_gqa=qhead_per_kvhead != 1,
            tile_m=tile_m,
            tile_n=tile_n,
        )
        assert head_dim == head_dim_k

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        cu_seqlens_q_ptr: cute.Pointer,
        cu_seqlens_k_ptr: cute.Pointer,
        attention_sink_bias_ptr: cute.Pointer,
        softmax_scale: float,
        current_stream: cuda.CUstream,
    ):
        q_tensor = cute.make_tensor(
            q_ptr, layout=cute.make_layout(self._q_shape, stride=self._q_stride)
        )
        k_tensor = cute.make_tensor(
            k_ptr, layout=cute.make_layout(self._k_shape, stride=self._k_stride)
        )
        v_tensor = cute.make_tensor(
            v_ptr, layout=cute.make_layout(self._v_shape, stride=self._v_stride)
        )
        o_tensor = cute.make_tensor(
            o_ptr, layout=cute.make_layout(self._o_shape, stride=self._o_stride)
        )
        lse_tensor = cute.make_tensor(
            lse_ptr,
            layout=cute.make_layout(self._lse_shape, stride=self._lse_stride),
        )
        cu_seqlens_q_tensor = cute.make_tensor(
            cu_seqlens_q_ptr,
            layout=cute.make_layout(
                self._cu_seqlens_q_shape,
                stride=self._cu_seqlens_q_stride,
            ),
        )
        cu_seqlens_k_tensor = cute.make_tensor(
            cu_seqlens_k_ptr,
            layout=cute.make_layout(
                self._cu_seqlens_k_shape,
                stride=self._cu_seqlens_k_stride,
            ),
        )
        attention_sink_bias_tensor = cute.make_tensor(
            attention_sink_bias_ptr,
            layout=cute.make_layout(
                self._attention_sink_bias_shape,
                stride=self._attention_sink_bias_stride,
            ),
        )
        self._kernel(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            mCuSeqlensQ=cu_seqlens_q_tensor,
            mCuSeqlensK=cu_seqlens_k_tensor,
            learnable_sink=attention_sink_bias_tensor,
            has_attention_sink_bias=self._has_attention_sink_bias,
            logical_num_batch_static=self._num_batch,
            logical_seqlen_q_static=self._max_seqlen_q,
            logical_seqlen_k_static=self._max_seqlen_k,
            window_size_left=(
                None if self._window_size_left == -1 else Int32(self._window_size_left)
            ),
            window_size_right=(
                None
                if self._window_size_right == -1
                else Int32(self._window_size_right)
            ),
            stream=current_stream,
        )


@functools.cache
def _compile_attention(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    dtype: torch.dtype,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    has_attention_sink_bias: bool,
    tile_m: int,
    tile_n: int,
):
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    launch = _AttentionForwardLaunch(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        dtype=dtype,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        has_attention_sink_bias=has_attention_sink_bias,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    return sm12x_compile(
        launch,
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        1.0,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "attention.contiguous.forward",
            1,
            (
                q_shape,
                k_shape,
                v_shape,
                dtype,
                causal,
                window_size_left,
                window_size_right,
                has_attention_sink_bias,
                tile_m,
                tile_n,
            ),
        ),
    )


@functools.cache
def _compile_varlen_attention(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    cu_seqlens_k_shape: tuple[int, ...],
    dtype: torch.dtype,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    has_attention_sink_bias: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    tile_m: int,
    tile_n: int,
):
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    launch = _VarlenAttentionForwardLaunch(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        cu_seqlens_q_shape=cu_seqlens_q_shape,
        cu_seqlens_k_shape=cu_seqlens_k_shape,
        dtype=dtype,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        has_attention_sink_bias=has_attention_sink_bias,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    return sm12x_compile(
        launch,
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        1.0,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "attention.contiguous.varlen_forward",
            1,
            (
                q_shape,
                k_shape,
                v_shape,
                cu_seqlens_q_shape,
                cu_seqlens_k_shape,
                dtype,
                causal,
                window_size_left,
                window_size_right,
                has_attention_sink_bias,
                max_seqlen_q,
                max_seqlen_k,
                tile_m,
                tile_n,
            ),
        ),
    )


@functools.cache
def _get_attention_plan(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device_index: int,
    dtype: torch.dtype,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    has_attention_sink_bias: bool,
    tile_m: int,
    tile_n: int,
) -> AttentionPlan:
    (
        num_batch,
        num_q_heads,
        num_kv_heads,
        qhead_per_kvhead,
        seqlen_q_static,
        seqlen_k_static,
        logical_q_rows_static,
        logical_total_q_rows,
    ) = _attention_logical_dims(q_shape, k_shape)
    return AttentionPlan(
        key=AttentionPlanKey(
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            device_index=device_index,
            dtype=dtype,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            has_attention_sink_bias=has_attention_sink_bias,
            tile_m=tile_m,
            tile_n=tile_n,
            num_batch=num_batch,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            qhead_per_kvhead=qhead_per_kvhead,
            seqlen_q_static=seqlen_q_static,
            seqlen_k_static=seqlen_k_static,
            logical_q_rows_static=logical_q_rows_static,
            logical_total_q_rows=logical_total_q_rows,
        ),
        compiled=_compile_attention(
            q_shape,
            k_shape,
            v_shape,
            dtype,
            causal,
            window_size_left,
            window_size_right,
            has_attention_sink_bias,
            tile_m,
            tile_n,
        ),
        cutlass_dtype=_torch_to_cutlass_dtype(dtype),
    )


@functools.cache
def _get_varlen_attention_plan(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    cu_seqlens_k_shape: tuple[int, ...],
    device_index: int,
    dtype: torch.dtype,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    has_attention_sink_bias: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    tile_m: int,
    tile_n: int,
) -> VarlenAttentionPlan:
    (
        num_batch,
        num_q_heads,
        num_kv_heads,
        qhead_per_kvhead,
        logical_q_rows_static,
        logical_total_q_rows,
    ) = _varlen_attention_logical_dims(
        q_shape,
        k_shape,
        cu_seqlens_q_shape,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return VarlenAttentionPlan(
        key=VarlenAttentionPlanKey(
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            cu_seqlens_q_shape=cu_seqlens_q_shape,
            cu_seqlens_k_shape=cu_seqlens_k_shape,
            device_index=device_index,
            dtype=dtype,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            has_attention_sink_bias=has_attention_sink_bias,
            tile_m=tile_m,
            tile_n=tile_n,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_batch=num_batch,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            qhead_per_kvhead=qhead_per_kvhead,
            logical_q_rows_static=logical_q_rows_static,
            logical_total_q_rows=logical_total_q_rows,
        ),
        compiled=_compile_varlen_attention(
            q_shape,
            k_shape,
            v_shape,
            cu_seqlens_q_shape,
            cu_seqlens_k_shape,
            dtype,
            causal,
            window_size_left,
            window_size_right,
            has_attention_sink_bias,
            max_seqlen_q,
            max_seqlen_k,
            tile_m,
            tile_n,
        ),
        cutlass_dtype=_torch_to_cutlass_dtype(dtype),
    )


def clear_attention_caches() -> None:
    """Clear global compile caches owned by the sm12x attention integration."""
    _compile_attention.cache_clear()
    _compile_varlen_attention.cache_clear()
    _get_attention_plan.cache_clear()
    _get_varlen_attention_plan.cache_clear()


def _validate_workspace(
    workspace: AttentionWorkspace,
    *,
    plan: AttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_shape,
        plan.v_shape,
        plan.device,
        plan.dtype,
        plan.causal,
        plan.window_size_left,
        plan.window_size_right,
        plan.has_attention_sink_bias,
        plan.tile_m,
        plan.tile_n,
    )
    actual = (
        workspace.q_shape,
        workspace.k_shape,
        workspace.v_shape,
        workspace.device,
        workspace.dtype,
        workspace.causal,
        workspace.window_size_left,
        workspace.window_size_right,
        workspace.has_attention_sink_bias,
        workspace.tile_m,
        workspace.tile_n,
    )
    if expected != actual:
        raise ValueError(
            "workspace shape mismatch: "
            "expected q/k/v/device/dtype/causal/window/sink/tile="
            f"{expected}, got {actual}"
        )
    if workspace.plan_key is not None and workspace.plan_key != plan.key:
        raise ValueError(
            f"workspace plan mismatch: expected {workspace.plan_key}, got {plan.key}"
        )


def _validate_varlen_workspace(
    workspace: VarlenAttentionWorkspace,
    *,
    plan: VarlenAttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_shape,
        plan.v_shape,
        plan.cu_seqlens_q_shape,
        plan.cu_seqlens_k_shape,
        plan.device,
        plan.dtype,
        plan.causal,
        plan.window_size_left,
        plan.window_size_right,
        plan.has_attention_sink_bias,
        plan.tile_m,
        plan.tile_n,
        plan.max_seqlen_q,
        plan.max_seqlen_k,
    )
    actual = (
        workspace.q_shape,
        workspace.k_shape,
        workspace.v_shape,
        workspace.cu_seqlens_q_shape,
        workspace.cu_seqlens_k_shape,
        workspace.device,
        workspace.dtype,
        workspace.causal,
        workspace.window_size_left,
        workspace.window_size_right,
        workspace.has_attention_sink_bias,
        workspace.tile_m,
        workspace.tile_n,
        workspace.max_seqlen_q,
        workspace.max_seqlen_k,
    )
    if expected != actual:
        raise ValueError(
            "varlen workspace shape mismatch: "
            "expected q/k/v/cu_q/cu_k/device/dtype/causal/window/sink/tile/max_seqlen="
            f"{expected}, got {actual}"
        )
    if workspace.plan_key is not None and workspace.plan_key != plan.key:
        raise ValueError(
            "varlen workspace plan mismatch: "
            f"expected {workspace.plan_key}, got {plan.key}"
        )


def _validate_attention_output_lse(
    *,
    output: torch.Tensor,
    lse: torch.Tensor,
    plan: AttentionPlan | VarlenAttentionPlan,
) -> None:
    if output.shape != plan.q_shape:
        raise ValueError(
            f"attention output must have shape {plan.q_shape}, got {tuple(output.shape)}"
        )
    if output.device != plan.device:
        raise ValueError(
            f"attention output device {output.device} does not match plan device {plan.device}"
        )
    if output.dtype != plan.dtype:
        raise ValueError(
            f"attention output dtype {output.dtype} does not match plan dtype {plan.dtype}"
        )
    expected_lse_shape = _lse_shape(plan.q_shape)
    if lse.shape != expected_lse_shape:
        raise ValueError(
            f"attention lse must have shape {expected_lse_shape}, got {tuple(lse.shape)}"
        )
    if lse.device != plan.device:
        raise ValueError(
            f"attention lse device {lse.device} does not match plan device {plan.device}"
        )
    if lse.dtype != torch.float32:
        raise ValueError(
            f"attention lse must have dtype torch.float32, got {lse.dtype}"
        )


def _attention_scratch_layout(
    *,
    q_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> _AttentionScratchLayout:
    cursor = 0
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)
    output_offset_bytes = cursor
    cursor += _shape_numel(q_shape) * _dtype_nbytes(dtype)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)
    lse_offset_bytes = cursor
    cursor += _shape_numel(_lse_shape(q_shape)) * _dtype_nbytes(torch.float32)
    return _AttentionScratchLayout(
        nbytes=max(int(cursor), 1),
        output_offset_bytes=output_offset_bytes,
        lse_offset_bytes=lse_offset_bytes,
    )


def _arena_view(
    arena: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    offset_bytes = _align_up(
        offset_bytes, max(_ARENA_ALIGN_BYTES, _dtype_nbytes(dtype))
    )
    nbytes = _shape_numel(shape) * _dtype_nbytes(dtype)
    return arena.narrow(0, offset_bytes, nbytes).view(dtype).view(shape)


def _attention_workspace_from_arena(
    plan: AttentionPlan,
    *,
    layout: _AttentionScratchLayout,
    arena: torch.Tensor,
) -> AttentionWorkspace:
    views = _attention_scratch_views_from_arena(plan, layout=layout, arena=arena)
    return AttentionWorkspace(
        q_shape=plan.q_shape,
        k_shape=plan.k_shape,
        v_shape=plan.v_shape,
        device=plan.device,
        dtype=plan.dtype,
        causal=plan.causal,
        window_size_left=plan.window_size_left,
        window_size_right=plan.window_size_right,
        has_attention_sink_bias=plan.has_attention_sink_bias,
        tile_m=plan.tile_m,
        tile_n=plan.tile_n,
        output=views.output,
        lse=views.lse,
        plan_key=getattr(plan, "key", None),
    )


def _attention_scratch_views_from_arena(
    plan: AttentionPlan,
    *,
    layout: _AttentionScratchLayout,
    arena: torch.Tensor,
) -> _AttentionScratchViews:
    output = _arena_view(
        arena,
        offset_bytes=layout.output_offset_bytes,
        shape=plan.q_shape,
        dtype=plan.dtype,
    )
    lse = _arena_view(
        arena,
        offset_bytes=layout.lse_offset_bytes,
        shape=_lse_shape(plan.q_shape),
        dtype=torch.float32,
    )
    return _AttentionScratchViews(output=output, lse=lse)


def _varlen_attention_workspace_from_arena(
    plan: VarlenAttentionPlan,
    *,
    layout: _AttentionScratchLayout,
    arena: torch.Tensor,
) -> VarlenAttentionWorkspace:
    views = _varlen_attention_scratch_views_from_arena(
        plan,
        layout=layout,
        arena=arena,
    )
    return VarlenAttentionWorkspace(
        q_shape=plan.q_shape,
        k_shape=plan.k_shape,
        v_shape=plan.v_shape,
        cu_seqlens_q_shape=plan.cu_seqlens_q_shape,
        cu_seqlens_k_shape=plan.cu_seqlens_k_shape,
        device=plan.device,
        dtype=plan.dtype,
        causal=plan.causal,
        window_size_left=plan.window_size_left,
        window_size_right=plan.window_size_right,
        has_attention_sink_bias=plan.has_attention_sink_bias,
        tile_m=plan.tile_m,
        tile_n=plan.tile_n,
        max_seqlen_q=plan.max_seqlen_q,
        max_seqlen_k=plan.max_seqlen_k,
        output=views.output,
        lse=views.lse,
        plan_key=getattr(plan, "key", None),
    )


def _varlen_attention_scratch_views_from_arena(
    plan: VarlenAttentionPlan,
    *,
    layout: _AttentionScratchLayout,
    arena: torch.Tensor,
) -> _AttentionScratchViews:
    output = _arena_view(
        arena,
        offset_bytes=layout.output_offset_bytes,
        shape=plan.q_shape,
        dtype=plan.dtype,
    )
    lse = _arena_view(
        arena,
        offset_bytes=layout.lse_offset_bytes,
        shape=_lse_shape(plan.q_shape),
        dtype=torch.float32,
    )
    return _AttentionScratchViews(output=output, lse=lse)


def _attention_workspace_from_scratch_plan(
    scratch_plan: AttentionScratchPlan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> AttentionWorkspace:
    arena = scratch_tensor(
        scratch,
        scratch_plan._scratch_specs,
        owner="contiguous attention",
    )
    return _attention_workspace_from_arena(
        scratch_plan.plan,
        layout=scratch_plan._layout,
        arena=arena,
    )


def _attention_scratch_views_from_scratch_plan(
    scratch_plan: AttentionScratchPlan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> _AttentionScratchViews:
    arena = scratch_tensor(
        scratch,
        scratch_plan._scratch_specs,
        owner="contiguous attention",
    )
    return _attention_scratch_views_from_arena(
        scratch_plan.plan,
        layout=scratch_plan._layout,
        arena=arena,
    )


def _varlen_attention_workspace_from_scratch_plan(
    scratch_plan: VarlenAttentionScratchPlan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> VarlenAttentionWorkspace:
    arena = scratch_tensor(
        scratch,
        scratch_plan._scratch_specs,
        owner="varlen contiguous attention",
    )
    return _varlen_attention_workspace_from_arena(
        scratch_plan.plan,
        layout=scratch_plan._layout,
        arena=arena,
    )


def _varlen_attention_scratch_views_from_scratch_plan(
    scratch_plan: VarlenAttentionScratchPlan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> _AttentionScratchViews:
    arena = scratch_tensor(
        scratch,
        scratch_plan._scratch_specs,
        owner="varlen contiguous attention",
    )
    return _varlen_attention_scratch_views_from_arena(
        scratch_plan.plan,
        layout=scratch_plan._layout,
        arena=arena,
    )


def build_attention_binding(
    *,
    workspace: AttentionWorkspace | AttentionWorkspacePool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: AttentionPlan | None = None,
    softmax_scale: float | None = None,
    attention_sink_bias: torch.Tensor | None = None,
) -> AttentionBinding:
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    has_attention_sink_bias = attention_sink_bias is not None
    if plan is None:
        if isinstance(workspace, AttentionWorkspacePool):
            raise TypeError("workspace pools require an explicit AttentionPlan")
        if has_attention_sink_bias != workspace.has_attention_sink_bias:
            raise ValueError(
                "attention_sink_bias mismatch: "
                f"workspace expects {workspace.has_attention_sink_bias}, "
                f"got {has_attention_sink_bias}"
            )
        plan = _get_attention_plan(
            q_shape,
            k_shape,
            v_shape,
            _cuda_device_index(workspace.device),
            workspace.dtype,
            workspace.causal,
            workspace.window_size_left,
            workspace.window_size_right,
            workspace.has_attention_sink_bias,
            workspace.tile_m,
            workspace.tile_n,
        )
    if has_attention_sink_bias != plan.has_attention_sink_bias:
        raise ValueError(
            "attention_sink_bias mismatch: "
            f"plan expects {plan.has_attention_sink_bias}, got {has_attention_sink_bias}"
        )
    _validate_attention_inputs_against_plan(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        plan=plan,
    )
    resolved_workspace = _resolve_attention_workspace(workspace, plan=plan)
    _validate_attention_output_lse(
        output=resolved_workspace.output,
        lse=resolved_workspace.lse,
        plan=plan,
    )
    return _build_attention_binding_from_views(
        output=resolved_workspace.output,
        lse=resolved_workspace.lse,
        q=q,
        k=k,
        v=v,
        plan=plan,
        softmax_scale=softmax_scale,
        attention_sink_bias=attention_sink_bias,
    )


def _build_attention_binding_from_views(
    *,
    output: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: AttentionPlan,
    softmax_scale: float | None = None,
    attention_sink_bias: torch.Tensor | None = None,
) -> AttentionBinding:
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    has_attention_sink_bias = attention_sink_bias is not None
    if has_attention_sink_bias != plan.has_attention_sink_bias:
        raise ValueError(
            "attention_sink_bias mismatch: "
            f"plan expects {plan.has_attention_sink_bias}, got {has_attention_sink_bias}"
        )
    _validate_attention_inputs_against_plan(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        plan=plan,
    )
    _validate_attention_output_lse(output=output, lse=lse, plan=plan)
    return AttentionBinding(
        q=q,
        k=k,
        v=v,
        output=output,
        lse=lse,
        plan=plan,
        softmax_scale=softmax_scale,
        attention_sink_bias=attention_sink_bias,
    )


def build_varlen_attention_binding(
    *,
    workspace: VarlenAttentionWorkspace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    plan: VarlenAttentionPlan | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool | None = None,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
) -> VarlenAttentionBinding:
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    (
        q_shape,
        k_shape,
        v_shape,
        cu_seqlens_q_shape,
        cu_seqlens_k_shape,
        device,
        dtype,
    ) = _validate_varlen_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    has_attention_sink_bias = attention_sink_bias is not None
    if plan is None:
        if has_attention_sink_bias != workspace.has_attention_sink_bias:
            raise ValueError(
                "attention_sink_bias mismatch: "
                f"workspace expects {workspace.has_attention_sink_bias}, "
                f"got {has_attention_sink_bias}"
            )
        resolved_max_seqlen_q = (
            workspace.max_seqlen_q
            if max_seqlen_q is None
            else _resolve_max_seqlen(cu_seqlens_q, max_seqlen_q, name="max_seqlen_q")
        )
        resolved_max_seqlen_k = (
            workspace.max_seqlen_k
            if max_seqlen_k is None
            else _resolve_max_seqlen(cu_seqlens_k, max_seqlen_k, name="max_seqlen_k")
        )
        resolved_causal = workspace.causal if causal is None else bool(causal)
        if window_size is None:
            window_size_left = workspace.window_size_left
            window_size_right = workspace.window_size_right
        else:
            window_size_left, window_size_right = _normalize_window_size(window_size)
        plan = _get_varlen_attention_plan(
            q_shape,
            k_shape,
            v_shape,
            cu_seqlens_q_shape,
            cu_seqlens_k_shape,
            _cuda_device_index(workspace.device),
            workspace.dtype,
            resolved_causal,
            window_size_left,
            window_size_right,
            workspace.has_attention_sink_bias,
            resolved_max_seqlen_q,
            resolved_max_seqlen_k,
            workspace.tile_m,
            workspace.tile_n,
        )
    if has_attention_sink_bias != plan.has_attention_sink_bias:
        raise ValueError(
            "attention_sink_bias mismatch: "
            f"plan expects {plan.has_attention_sink_bias}, got {has_attention_sink_bias}"
        )
    if max_seqlen_q is not None and int(max_seqlen_q) != plan.max_seqlen_q:
        raise ValueError(
            f"max_seqlen_q mismatch: plan has {plan.max_seqlen_q}, got {max_seqlen_q}"
        )
    if max_seqlen_k is not None and int(max_seqlen_k) != plan.max_seqlen_k:
        raise ValueError(
            f"max_seqlen_k mismatch: plan has {plan.max_seqlen_k}, got {max_seqlen_k}"
        )
    if causal is not None and bool(causal) != plan.causal:
        raise ValueError(f"causal mismatch: plan has {plan.causal}, got {causal}")
    if window_size is not None:
        window_size_left, window_size_right = _normalize_window_size(window_size)
        if (
            window_size_left != plan.window_size_left
            or window_size_right != plan.window_size_right
        ):
            raise ValueError(
                "window_size mismatch: "
                f"plan has {(plan.window_size_left, plan.window_size_right)}, "
                f"got {(window_size_left, window_size_right)}"
            )
    _validate_varlen_inputs_against_plan(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        cu_seqlens_q_shape=cu_seqlens_q_shape,
        cu_seqlens_k_shape=cu_seqlens_k_shape,
        device=device,
        dtype=dtype,
        plan=plan,
    )
    _validate_varlen_workspace(workspace, plan=plan)
    _validate_attention_output_lse(
        output=workspace.output, lse=workspace.lse, plan=plan
    )
    return _build_varlen_attention_binding_from_views(
        output=workspace.output,
        lse=workspace.lse,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        plan=plan,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_sink_bias=attention_sink_bias,
    )


def _build_varlen_attention_binding_from_views(
    *,
    output: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    plan: VarlenAttentionPlan,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool | None = None,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
) -> VarlenAttentionBinding:
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    (
        q_shape,
        k_shape,
        v_shape,
        cu_seqlens_q_shape,
        cu_seqlens_k_shape,
        device,
        dtype,
    ) = _validate_varlen_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    has_attention_sink_bias = attention_sink_bias is not None
    if has_attention_sink_bias != plan.has_attention_sink_bias:
        raise ValueError(
            "attention_sink_bias mismatch: "
            f"plan expects {plan.has_attention_sink_bias}, got {has_attention_sink_bias}"
        )
    if max_seqlen_q is not None and int(max_seqlen_q) != plan.max_seqlen_q:
        raise ValueError(
            f"max_seqlen_q mismatch: plan has {plan.max_seqlen_q}, got {max_seqlen_q}"
        )
    if max_seqlen_k is not None and int(max_seqlen_k) != plan.max_seqlen_k:
        raise ValueError(
            f"max_seqlen_k mismatch: plan has {plan.max_seqlen_k}, got {max_seqlen_k}"
        )
    if causal is not None and bool(causal) != plan.causal:
        raise ValueError(f"causal mismatch: plan has {plan.causal}, got {causal}")
    if window_size is not None:
        window_size_left, window_size_right = _normalize_window_size(window_size)
        if (
            window_size_left != plan.window_size_left
            or window_size_right != plan.window_size_right
        ):
            raise ValueError(
                "window_size mismatch: "
                f"plan has {(plan.window_size_left, plan.window_size_right)}, "
                f"got {(window_size_left, window_size_right)}"
            )
    _validate_varlen_inputs_against_plan(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        cu_seqlens_q_shape=cu_seqlens_q_shape,
        cu_seqlens_k_shape=cu_seqlens_k_shape,
        device=device,
        dtype=dtype,
        plan=plan,
    )
    _validate_attention_output_lse(output=output, lse=lse, plan=plan)
    return VarlenAttentionBinding(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        output=output,
        lse=lse,
        plan=plan,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_sink_bias=attention_sink_bias,
    )


def plan_attention_scratch(plan: AttentionPlan) -> AttentionScratchPlan:
    layout = _attention_scratch_layout(q_shape=plan.q_shape, dtype=plan.dtype)
    return AttentionScratchPlan(
        plan=plan,
        _layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "contiguous_attention.scratch",
                nbytes=layout.nbytes,
                device=plan.device,
            ),
        ),
    )


def plan_varlen_attention_scratch(
    plan: VarlenAttentionPlan,
) -> VarlenAttentionScratchPlan:
    layout = _attention_scratch_layout(q_shape=plan.q_shape, dtype=plan.dtype)
    return VarlenAttentionScratchPlan(
        plan=plan,
        _layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "varlen_contiguous_attention.scratch",
                nbytes=layout.nbytes,
                device=plan.device,
            ),
        ),
    )


def allocate_attention_workspace_for_plan(plan: AttentionPlan) -> AttentionWorkspace:
    """Allocate reusable scratch for one exact contiguous attention plan."""
    output = torch.empty(plan.q_shape, dtype=plan.dtype, device=plan.device)
    lse = torch.empty(_lse_shape(plan.q_shape), dtype=torch.float32, device=plan.device)
    return AttentionWorkspace(
        q_shape=plan.q_shape,
        k_shape=plan.k_shape,
        v_shape=plan.v_shape,
        device=plan.device,
        dtype=plan.dtype,
        causal=plan.causal,
        window_size_left=plan.window_size_left,
        window_size_right=plan.window_size_right,
        has_attention_sink_bias=plan.has_attention_sink_bias,
        tile_m=plan.tile_m,
        tile_n=plan.tile_n,
        output=output,
        lse=lse,
        plan_key=plan.key,
    )


def allocate_varlen_attention_workspace_for_plan(
    plan: VarlenAttentionPlan,
) -> VarlenAttentionWorkspace:
    """Allocate reusable scratch for one exact packed varlen attention plan."""
    output = torch.empty(plan.q_shape, dtype=plan.dtype, device=plan.device)
    lse = torch.empty(_lse_shape(plan.q_shape), dtype=torch.float32, device=plan.device)
    return VarlenAttentionWorkspace(
        q_shape=plan.q_shape,
        k_shape=plan.k_shape,
        v_shape=plan.v_shape,
        cu_seqlens_q_shape=plan.cu_seqlens_q_shape,
        cu_seqlens_k_shape=plan.cu_seqlens_k_shape,
        device=plan.device,
        dtype=plan.dtype,
        causal=plan.causal,
        window_size_left=plan.window_size_left,
        window_size_right=plan.window_size_right,
        has_attention_sink_bias=plan.has_attention_sink_bias,
        tile_m=plan.tile_m,
        tile_n=plan.tile_n,
        max_seqlen_q=plan.max_seqlen_q,
        max_seqlen_k=plan.max_seqlen_k,
        output=output,
        lse=lse,
        plan_key=plan.key,
    )


def allocate_attention_workspace_pool() -> AttentionWorkspacePool:
    """Allocate an explicit caller-owned workspace pool for contiguous attention."""
    return AttentionWorkspacePool()


def _resolve_attention_workspace(
    workspace: AttentionWorkspace | AttentionWorkspacePool,
    *,
    plan: AttentionPlan,
) -> AttentionWorkspace:
    if isinstance(workspace, AttentionWorkspace):
        _validate_workspace(workspace, plan=plan)
        return workspace
    if not isinstance(workspace, AttentionWorkspacePool):
        raise TypeError(
            "workspace must be an AttentionWorkspace or AttentionWorkspacePool"
        )

    stream_key = int(torch.cuda.current_stream(plan.device).cuda_stream)
    key = (stream_key, plan.key)
    resolved = workspace.workspaces.get(key)
    if resolved is None:
        resolved = allocate_attention_workspace_for_plan(plan)
        workspace.workspaces[key] = resolved
    return resolved


def create_attention_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    tile_shape: tuple[int, int] | None = None,
) -> AttentionPlan:
    """Create one exact contiguous attention launch plan."""
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    _, _, _, head_dim = _seq_dims(q_shape)
    tile_m, tile_n = tile_shape or _select_tile_shape(head_dim, causal=causal)
    window_size_left, window_size_right = _normalize_window_size(window_size)
    return _get_attention_plan(
        q_shape,
        k_shape,
        v_shape,
        _cuda_device_index(device),
        dtype,
        causal,
        window_size_left,
        window_size_right,
        attention_sink_bias is not None,
        tile_m,
        tile_n,
    )


def create_varlen_attention_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    causal: bool = False,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    tile_shape: tuple[int, int] | None = None,
) -> VarlenAttentionPlan:
    """Create one exact packed varlen contiguous attention launch plan."""
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    (
        q_shape,
        k_shape,
        v_shape,
        cu_seqlens_q_shape,
        cu_seqlens_k_shape,
        device,
        dtype,
    ) = _validate_varlen_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    max_seqlen_q = _resolve_max_seqlen(
        cu_seqlens_q,
        max_seqlen_q,
        name="max_seqlen_q",
    )
    max_seqlen_k = _resolve_max_seqlen(
        cu_seqlens_k,
        max_seqlen_k,
        name="max_seqlen_k",
    )
    _, _, head_dim = q_shape
    tile_m, tile_n = tile_shape or _select_tile_shape(head_dim, causal=causal)
    window_size_left, window_size_right = _normalize_window_size(window_size)
    return _get_varlen_attention_plan(
        q_shape,
        k_shape,
        v_shape,
        cu_seqlens_q_shape,
        cu_seqlens_k_shape,
        _cuda_device_index(device),
        dtype,
        causal,
        window_size_left,
        window_size_right,
        attention_sink_bias is not None,
        max_seqlen_q,
        max_seqlen_k,
        tile_m,
        tile_n,
    )


def allocate_attention_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    tile_shape: tuple[int, int] | None = None,
) -> AttentionWorkspace:
    """Allocate one exact-shape workspace for `sm12x_attention_forward`."""
    plan = create_attention_plan(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        attention_sink_bias=attention_sink_bias,
        tile_shape=tile_shape,
    )
    return allocate_attention_workspace_for_plan(plan)


def allocate_varlen_attention_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    causal: bool = False,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    tile_shape: tuple[int, int] | None = None,
) -> VarlenAttentionWorkspace:
    """Allocate one exact-shape workspace for `sm12x_varlen_attention_forward`."""
    plan = create_varlen_attention_plan(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size=window_size,
        attention_sink_bias=attention_sink_bias,
        tile_shape=tile_shape,
    )
    return allocate_varlen_attention_workspace_for_plan(plan)


def sm12x_attention_forward(
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    *,
    plan: AttentionPlan | None = None,
    softmax_scale: float | None = None,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    binding: AttentionBinding | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute contiguous self-attention using the restored forward kernel."""
    output: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q", q),
                ("k", k),
                ("v", v),
                ("plan", plan),
                ("softmax_scale", softmax_scale),
                ("attention_sink_bias", attention_sink_bias),
            )
            if value is not None
        ]
        if window_size is not None:
            extras.append("window_size")
        if extras:
            raise ValueError(
                "attention binding owns runtime tensors, outputs, plan, and options; "
                f"do not also pass {', '.join(extras)}"
            )
        q = binding.q
        k = binding.k
        v = binding.v
        output = binding.output
        lse = binding.lse
        plan = binding.plan
        softmax_scale = binding.softmax_scale
        attention_sink_bias = binding.attention_sink_bias
    else:
        raise TypeError("sm12x_attention_forward requires binding")
    if q is None or k is None or v is None:
        raise TypeError("attention binding is missing q, k, or v")
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    has_attention_sink_bias = attention_sink_bias is not None
    if plan is None:
        raise TypeError("attention binding is missing plan")
    resolved_plan = plan
    if has_attention_sink_bias != resolved_plan.has_attention_sink_bias:
        raise ValueError(
            "attention_sink_bias mismatch: "
            f"plan expects {resolved_plan.has_attention_sink_bias}, got {has_attention_sink_bias}"
        )
    if window_size is not None:
        window_size_left, window_size_right = _normalize_window_size(window_size)
        if (
            window_size_left != resolved_plan.window_size_left
            or window_size_right != resolved_plan.window_size_right
        ):
            raise ValueError(
                "window_size mismatch: "
                f"plan has {(resolved_plan.window_size_left, resolved_plan.window_size_right)}, "
                f"got {(window_size_left, window_size_right)}"
            )
    _validate_attention_inputs_against_plan(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        plan=resolved_plan,
    )
    if output is None or lse is None:
        raise TypeError("attention binding is missing output or lse")
    else:
        _validate_attention_output_lse(output=output, lse=lse, plan=resolved_plan)
    _, _, _, head_dim = _seq_dims(q_shape)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if attention_sink_bias is None:
        attention_sink_bias = _attention_sink_placeholder(resolved_plan.device_index)

    resolved_plan.compiled(
        make_ptr(
            resolved_plan.cutlass_dtype,
            q.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            resolved_plan.cutlass_dtype,
            k.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            resolved_plan.cutlass_dtype,
            v.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            resolved_plan.cutlass_dtype,
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            lse.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Float32,
            attention_sink_bias.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        float(softmax_scale),
        current_cuda_stream(),
    )
    return output, lse


def sm12x_varlen_attention_forward(
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    plan: VarlenAttentionPlan | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool | None = None,
    window_size: int | tuple[int, int] | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    binding: VarlenAttentionBinding | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute packed varlen contiguous attention using cu_seqlens metadata."""
    output: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q", q),
                ("k", k),
                ("v", v),
                ("cu_seqlens_q", cu_seqlens_q),
                ("cu_seqlens_k", cu_seqlens_k),
                ("plan", plan),
                ("max_seqlen_q", max_seqlen_q),
                ("max_seqlen_k", max_seqlen_k),
                ("softmax_scale", softmax_scale),
                ("causal", causal),
                ("attention_sink_bias", attention_sink_bias),
            )
            if value is not None
        ]
        if window_size is not None:
            extras.append("window_size")
        if extras:
            raise ValueError(
                "varlen attention binding owns runtime tensors, outputs, plan, and options; "
                f"do not also pass {', '.join(extras)}"
            )
        q = binding.q
        k = binding.k
        v = binding.v
        cu_seqlens_q = binding.cu_seqlens_q
        cu_seqlens_k = binding.cu_seqlens_k
        output = binding.output
        lse = binding.lse
        plan = binding.plan
        max_seqlen_q = binding.max_seqlen_q
        max_seqlen_k = binding.max_seqlen_k
        softmax_scale = binding.softmax_scale
        causal = binding.causal
        window_size = binding.window_size
        attention_sink_bias = binding.attention_sink_bias
    else:
        raise TypeError("sm12x_varlen_attention_forward requires binding")
    if q is None or k is None or v is None or cu_seqlens_q is None:
        raise TypeError("varlen attention binding is missing q, k, v, or cu_seqlens_q")
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    (
        q_shape,
        k_shape,
        v_shape,
        cu_seqlens_q_shape,
        cu_seqlens_k_shape,
        device,
        dtype,
    ) = _validate_varlen_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    attention_sink_bias = _prepare_attention_sink_bias(
        attention_sink_bias,
        q_shape=q_shape,
        device=device,
    )
    has_attention_sink_bias = attention_sink_bias is not None
    if plan is None:
        raise TypeError("varlen attention binding is missing plan")
    resolved_plan = plan
    if has_attention_sink_bias != resolved_plan.has_attention_sink_bias:
        raise ValueError(
            "attention_sink_bias mismatch: "
            f"plan expects {resolved_plan.has_attention_sink_bias}, got {has_attention_sink_bias}"
        )
    if max_seqlen_q is not None and int(max_seqlen_q) != resolved_plan.max_seqlen_q:
        raise ValueError(
            f"max_seqlen_q mismatch: plan has {resolved_plan.max_seqlen_q}, got {max_seqlen_q}"
        )
    if max_seqlen_k is not None and int(max_seqlen_k) != resolved_plan.max_seqlen_k:
        raise ValueError(
            f"max_seqlen_k mismatch: plan has {resolved_plan.max_seqlen_k}, got {max_seqlen_k}"
        )
    if causal is not None and bool(causal) != resolved_plan.causal:
        raise ValueError(
            f"causal mismatch: plan has {resolved_plan.causal}, got {causal}"
        )
    if window_size is not None:
        window_size_left, window_size_right = _normalize_window_size(window_size)
        if (
            window_size_left != resolved_plan.window_size_left
            or window_size_right != resolved_plan.window_size_right
        ):
            raise ValueError(
                "window_size mismatch: "
                f"plan has {(resolved_plan.window_size_left, resolved_plan.window_size_right)}, "
                f"got {(window_size_left, window_size_right)}"
            )
    _validate_varlen_inputs_against_plan(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        cu_seqlens_q_shape=cu_seqlens_q_shape,
        cu_seqlens_k_shape=cu_seqlens_k_shape,
        device=device,
        dtype=dtype,
        plan=resolved_plan,
    )
    if output is None or lse is None:
        raise TypeError("varlen attention binding is missing output or lse")
    else:
        _validate_attention_output_lse(output=output, lse=lse, plan=resolved_plan)
    _, _, head_dim = q_shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if attention_sink_bias is None:
        attention_sink_bias = _attention_sink_placeholder(resolved_plan.device_index)

    resolved_plan.compiled(
        make_ptr(
            resolved_plan.cutlass_dtype,
            q.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            resolved_plan.cutlass_dtype,
            k.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            resolved_plan.cutlass_dtype,
            v.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            resolved_plan.cutlass_dtype,
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            lse.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32,
            cu_seqlens_q.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32,
            cu_seqlens_k.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Float32,
            attention_sink_bias.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        float(softmax_scale),
        current_cuda_stream(),
    )
    return output, lse


__all__ = [
    "AttentionBinding",
    "AttentionPlan",
    "AttentionPlanKey",
    "AttentionScratchPlan",
    "VarlenAttentionBinding",
    "VarlenAttentionPlan",
    "VarlenAttentionPlanKey",
    "VarlenAttentionScratchPlan",
    "sm12x_attention_forward",
    "sm12x_varlen_attention_forward",
    "clear_attention_caches",
    "create_attention_plan",
    "create_varlen_attention_plan",
    "plan_attention_scratch",
    "plan_varlen_attention_scratch",
]
