"""CuTeDSL mHC residual helpers for DeepSeek-style residual mixing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from b12x.preparation.types import Plan, plan_from_handle, require_prepared
from ._tuning import MhcConfig, MhcQuery

MHC_MULT = 4
MHC_MIXES = (2 + MHC_MULT) * MHC_MULT
MHC_PARTIALS = 1 + MHC_MIXES
MHC_DEFAULT_SPLIT_K = 64
MHC_DEFAULT_BLOCK_K = 256
MHC_DEFAULT_BLOCK_H = 512
MHC_SOURCE_TILE_H = 128
MHC_GRAM_BLOCK_H = 1024
MHC_SUPPORTED_HIDDEN_SIZES = (4096, 5120, 7168)
MHC_SUPPORTED_RMS_EPS = (1.0e-20, 1.0e-6, 1.0e-5)
_COLLAPSE_SUPPORTED_HIDDEN_SIZES = (4096, 5120, 7168)


def _run_collapse_impl(
    state: torch.Tensor, pre_mix: torch.Tensor | None, *, out: torch.Tensor, _state: "_MhcState",
) -> torch.Tensor:
    """Contract BF16 ``[T,4,H]`` using FP32 weights, or a uniform stream mean.

    Accumulation is FP32 with one BF16 output rounding. ``out`` must be a
    disjoint contiguous caller-owned ``[T,H]`` tensor. Warm either mixing mode
    before graph capture; live rows never select a compiled specialization.
    """
    _state.require("collapse", state)
    if (pre_mix is not None) != _state.query.collapse_weighted:
        raise ValueError("collapse weights differ from prepared operand presence")
    if state.ndim != 3 or state.shape[1] != MHC_MULT:
        raise ValueError("state must have shape [T, 4, H]")
    tokens, _, hidden = state.shape
    if hidden not in _COLLAPSE_SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            f"hidden size must be one of {_COLLAPSE_SUPPORTED_HIDDEN_SIZES}"
        )
    if state.device.type != "cuda":
        raise ValueError("state must be a CUDA tensor")
    for name, tensor, shape, dtype in (
        ("state", state, (tokens, MHC_MULT, hidden), torch.bfloat16),
        ("out", out, (tokens, hidden), torch.bfloat16),
    ):
        _validate_optional_view(
            tensor, shape=shape, dtype=dtype, device=state.device, name=name,
        )
    if pre_mix is not None:
        _validate_optional_view(
            pre_mix, shape=(tokens, MHC_MULT), dtype=torch.float32,
            device=state.device, name="pre_mix",
        )
    _state.launchers["collapse"](state=state, pre_mix=pre_mix, out=out)
    return out


def _required_mhc_split_k(hidden_size: int, block_k: int) -> int:
    total_k = MHC_MULT * int(hidden_size)
    block_k = int(block_k)
    if block_k <= 0 or total_k % block_k != 0:
        return -1
    return total_k // block_k


def _supports_fused_mhc_gram(
    *,
    hidden_size: int,
    split_k: int,
    block_k: int,
    block_h: int,
) -> bool:
    hidden_size = int(hidden_size)
    split_k = int(split_k)
    block_k = int(block_k)
    block_h = int(block_h)
    required_split_k = _required_mhc_split_k(hidden_size, block_k)
    return (
        hidden_size in MHC_SUPPORTED_HIDDEN_SIZES
        and block_k == MHC_DEFAULT_BLOCK_K
        and block_h == MHC_DEFAULT_BLOCK_H
        and hidden_size % MHC_SOURCE_TILE_H == 0
        and hidden_size % MHC_GRAM_BLOCK_H == 0
        and required_split_k > 0
        and split_k == required_split_k
    )


def _supports_mhc_post_hidden(hidden_size: int) -> bool:
    return int(hidden_size) in (4096, 5120, 7168)


@dataclass(frozen=True, kw_only=True)
class B12XMHCBinding:
    state: "_MhcState"
    partials: torch.Tensor | None = None
    y: torch.Tensor | None = None
    post_buffer: torch.Tensor | None = None
    comb_buffer: torch.Tensor | None = None
    out: torch.Tensor | None = None
    pre_out: torch.Tensor | None = None
    expected_m: int | None = None
    plan: Plan | None = None

    def pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        pre_mix: torch.Tensor | None = None,
        pre_out: torch.Tensor | None = None,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return b12x_mhc_pre(
            residual, fn, hc_scale, hc_base, rms_eps=rms_eps, hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters, pre_mix=pre_mix, pre_out=pre_out,
            norm_weight=norm_weight, norm_eps=norm_eps, binding=self,
        )

    def post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        prev_post: torch.Tensor,
        prev_comb: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        pre_mix: torch.Tensor | None = None,
        pre_out: torch.Tensor | None = None,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        fn_bf16: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return b12x_mhc_post_pre(
            x, residual, prev_post, prev_comb, fn, hc_scale, hc_base,
            rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters,
            pre_mix=pre_mix, pre_out=pre_out, norm_weight=norm_weight,
            norm_eps=norm_eps, fn_bf16=fn_bf16, binding=self,
        )


@dataclass(frozen=True, kw_only=True)
class B12XMHCScratchCaps:
    device: torch.device | str
    max_tokens: int
    hidden_size: int
    dtype: torch.dtype = torch.bfloat16
    split_k: int | None = None

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        if self.split_k is None:
            object.__setattr__(self, "split_k", _required_mhc_split_k(self.hidden_size, MHC_DEFAULT_BLOCK_K))
        for name in ("max_tokens", "hidden_size", "split_k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"mHC scratch currently supports torch.bfloat16 outputs, got {self.dtype}"
            )


@dataclass(frozen=True)
class _MHCScratchLayout:
    nbytes: int
    partials_offset_bytes: int


@dataclass(frozen=True)
class _MhcState:
    caps: B12XMHCScratchCaps
    layout: _MHCScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    config: MhcConfig
    query: MhcQuery
    launchers: Mapping[str, Callable]

    def require(self, operation, tensor, *, norm_weight=None, fn_bf16=None,
                rms_eps=None, hc_eps=None, sinkhorn_iters=None, norm_eps=None):
        query = self.query
        if operation != query.operation:
            raise ValueError(f"execution prepares {query.operation}, not {operation}")
        if tensor.device != self.caps.device:
            raise ValueError("MHC tensor and prepared device differ")
        if not 0 <= tensor.shape[0] <= query.max_tokens or tensor.shape[-1] != query.hidden_size:
            raise ValueError("MHC tensor exceeds its planned token capacity or differs in hidden size")
        if operation == "pre" and (tensor.ndim == 3) != query.expanded_residual:
            raise ValueError("MHC residual layout differs from its prepared invocation")
        if operation in ("pre", "post_pre"):
            if (norm_weight is not None) != query.has_norm_weight:
                raise ValueError("MHC normalization operand presence changed")
            if norm_weight is not None and str(norm_weight.dtype).removeprefix("torch.") != query.norm_weight_dtype:
                raise ValueError("MHC normalization dtype changed")
            if operation == "post_pre" and (fn_bf16 is not None) != query.has_fn_bf16:
                raise ValueError("MHC BF16 projection operand presence changed")
            if (rms_eps, hc_eps, sinkhorn_iters, norm_eps) != (
                query.rms_eps, query.hc_eps, query.sinkhorn_iters, query.norm_eps,
            ):
                raise ValueError("MHC numerical recipe differs from prepared invocation")
    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def _partials_from_scratch(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="mHC",
        )
        max_tokens = int(self.caps.max_tokens)
        split_k = int(self.caps.split_k)
        partials, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=self.layout.partials_offset_bytes,
            shape=(max_tokens, split_k, MHC_PARTIALS),
            dtype=torch.float32,
        )
        return partials

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        tokens: int | None = None,
        expected_m: int | None = None,
        y: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pre_out: torch.Tensor | None = None,
        plan: Plan | None = None,
    ) -> B12XMHCBinding:
        live_tokens = int(self.caps.max_tokens) if tokens is None else int(tokens)
        if not 0 <= live_tokens <= self.query.max_tokens:
            raise ValueError("MHC binding exceeds its planned token capacity")
        if self.query.output_mode != "provided" or not self._scratch_specs:
            raise ValueError("this MHC operation does not declare bound scratch")
        expected_m = _canonicalize_mhc_expected_m(
            expected_m,
            min_tokens=live_tokens,
            max_tokens=int(self.caps.max_tokens),
        )
        partials = self._partials_from_scratch(scratch=scratch)[:live_tokens]
        _validate_optional_view(
            pre_out,
            shape=(live_tokens, MHC_MULT),
            dtype=torch.float32,
            device=self.caps.device,
            name="mHC pre_out",
        )
        _validate_mhc_binding_views(
            partials=partials,
            y=y,
            post=post,
            comb=comb,
            out=out,
            tokens=live_tokens,
            hidden_size=int(self.caps.hidden_size),
            split_k=int(self.caps.split_k),
            dtype=self.caps.dtype,
            device=self.caps.device,
        )
        return B12XMHCBinding(
            state=self,
            plan=plan,
            partials=partials,
            y=y,
            post_buffer=post,
            comb_buffer=comb,
            out=out,
            pre_out=pre_out,
            expected_m=expected_m,
        )


def _validate_optional_view(
    tensor: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if tensor is None:
        return
    if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"{name} must have shape {shape}, dtype {dtype}, and device {device}; "
            f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )
    _require_contiguous(tensor, name=name)


def _canonicalize_mhc_expected_m(
    expected_m: int | None,
    *,
    min_tokens: int = 0,
    max_tokens: int | None = None,
) -> int | None:
    if expected_m is None:
        return None
    expected = int(expected_m)
    if expected <= 0:
        raise ValueError(f"expected_m must be positive when provided, got {expected}")
    if expected < int(min_tokens):
        raise ValueError(
            f"expected_m={expected} is smaller than live tokens={int(min_tokens)}"
        )
    if max_tokens is not None and expected > int(max_tokens):
        raise ValueError(
            f"expected_m={expected} exceeds MHC scratch capacity {int(max_tokens)}"
        )
    return expected




def _validate_mhc_binding_views(
    *,
    partials: torch.Tensor | None,
    y: torch.Tensor | None,
    post: torch.Tensor | None,
    comb: torch.Tensor | None,
    out: torch.Tensor | None,
    tokens: int,
    hidden_size: int,
    split_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if partials is not None:
        _validate_optional_view(
            partials,
            shape=(tokens, split_k, MHC_PARTIALS),
            dtype=torch.float32,
            device=device,
            name="mHC partials",
        )
    _validate_optional_view(
        y,
        shape=(tokens, hidden_size),
        dtype=dtype,
        device=device,
        name="mHC y",
    )
    _validate_optional_view(
        post,
        shape=(tokens, MHC_MULT),
        dtype=torch.float32,
        device=device,
        name="mHC post",
    )
    _validate_optional_view(
        comb,
        shape=(tokens, MHC_MULT, MHC_MULT),
        dtype=torch.float32,
        device=device,
        name="mHC comb",
    )
    _validate_optional_view(
        out,
        shape=(tokens, MHC_MULT, hidden_size),
        dtype=dtype,
        device=device,
        name="mHC out",
    )


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _slice_capacity_view(
    tensor: torch.Tensor | None,
    *,
    tokens: int,
    tail_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    expected = (tokens, *tail_shape)
    if tuple(tensor.shape) == expected:
        return tensor
    if (
        tensor.ndim == len(expected)
        and int(tensor.shape[0]) >= tokens
        and tuple(tensor.shape[1:]) == tail_shape
        and tensor.dtype == dtype
        and tensor.device == device
    ):
        return tensor[:tokens]
    raise ValueError(
        f"{name} must have shape {expected} or capacity >= {tokens} with tail "
        f"{tail_shape}, dtype {dtype}, and device {device}; got "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    )


def _layout_mhc_scratch(caps: B12XMHCScratchCaps) -> _MHCScratchLayout:
    cursor = 0

    def reserve(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[int, int]:
        nonlocal cursor
        offset = align_up(cursor, max(SCRATCH_ALIGN_BYTES, dtype_nbytes(dtype)))
        cursor = offset + _shape_numel(shape) * dtype_nbytes(dtype)
        return offset, cursor

    partials_offset_bytes, _ = reserve(
        (int(caps.max_tokens), int(caps.split_k), MHC_PARTIALS),
        torch.float32,
    )
    return _MHCScratchLayout(
        nbytes=cursor,
        partials_offset_bytes=partials_offset_bytes,
    )




def _require_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_pre_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    expanded: bool = False,
) -> tuple[int, int, int]:
    if residual.device.type != "cuda":
        raise ValueError("residual must be a CUDA tensor")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be torch.bfloat16, got {residual.dtype}")
    if residual.ndim != (3 if expanded else 2) or (expanded and residual.shape[1] != MHC_MULT):
        raise ValueError(
            f"residual must match its declared broadcast or four-stream layout, got {tuple(residual.shape)}"
        )
    tokens, hidden_size = int(residual.shape[0]), int(residual.shape[-1])
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if fn.dtype != torch.float32:
        raise ValueError(f"fn must be torch.float32, got {fn.dtype}")
    fn_width = MHC_MULT * hidden_size if expanded else hidden_size
    if fn.shape != (MHC_MIXES, fn_width):
        raise ValueError(
            f"fn must have shape {(MHC_MIXES, fn_width)}, got {tuple(fn.shape)}"
        )
    if hc_scale.dtype != torch.float32 or tuple(hc_scale.shape) != (3,):
        raise ValueError(
            f"hc_scale must be float32 shape [3], got {hc_scale.dtype} {tuple(hc_scale.shape)}"
        )
    if hc_base.dtype != torch.float32 or tuple(hc_base.shape) != (MHC_MIXES,):
        raise ValueError(
            f"hc_base must be float32 shape [{MHC_MIXES}], got {hc_base.dtype} {tuple(hc_base.shape)}"
        )
    if (
        fn.device != residual.device
        or hc_scale.device != residual.device
        or hc_base.device != residual.device
    ):
        raise ValueError("fn, hc_scale, and hc_base must be on the residual device")
    _require_contiguous(residual, name="residual")
    _require_contiguous(fn, name="fn")
    _require_contiguous(hc_scale, name="hc_scale")
    _require_contiguous(hc_base, name="hc_base")
    return tokens, hidden_size, MHC_MULT * hidden_size


def _validate_post_pre_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> tuple[int, int, int]:
    if residual.device.type != "cuda":
        raise ValueError("residual must be a CUDA tensor")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be torch.bfloat16, got {residual.dtype}")
    if residual.ndim != 3:
        raise ValueError(
            f"residual must be rank-3 [tokens, 4, hidden], got {tuple(residual.shape)}"
        )
    tokens, hc_mult, hidden_size = map(int, residual.shape)
    if hc_mult != MHC_MULT:
        raise ValueError(f"residual hc dimension must be {MHC_MULT}, got {hc_mult}")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if fn.dtype != torch.float32:
        raise ValueError(f"fn must be torch.float32, got {fn.dtype}")
    if fn.shape != (MHC_MIXES, MHC_MULT * hidden_size):
        raise ValueError(
            "fn must have shape "
            f"{(MHC_MIXES, MHC_MULT * hidden_size)}, got {tuple(fn.shape)}"
        )
    if hc_scale.dtype != torch.float32 or tuple(hc_scale.shape) != (3,):
        raise ValueError(
            "hc_scale must be float32 shape [3], got "
            f"{hc_scale.dtype} {tuple(hc_scale.shape)}"
        )
    if hc_base.dtype != torch.float32 or tuple(hc_base.shape) != (MHC_MIXES,):
        raise ValueError(
            f"hc_base must be float32 shape [{MHC_MIXES}], got "
            f"{hc_base.dtype} {tuple(hc_base.shape)}"
        )
    if (
        fn.device != residual.device
        or hc_scale.device != residual.device
        or hc_base.device != residual.device
    ):
        raise ValueError("fn, hc_scale, and hc_base must be on the residual device")
    _require_contiguous(residual, name="residual")
    _require_contiguous(fn, name="fn")
    _require_contiguous(hc_scale, name="hc_scale")
    _require_contiguous(hc_base, name="hc_base")
    return tokens, hidden_size, MHC_MULT * hidden_size


def _validate_norm_weight(
    norm_weight: torch.Tensor | None,
    *,
    hidden_size: int,
    device: torch.device,
) -> None:
    if norm_weight is None:
        return
    if norm_weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"norm_weight must be bf16 or fp32, got {norm_weight.dtype}")
    if norm_weight.device != device:
        raise ValueError("norm_weight must be on the residual device")
    if tuple(norm_weight.shape) != (hidden_size,):
        raise ValueError(
            f"norm_weight must have shape {(hidden_size,)}, got {tuple(norm_weight.shape)}"
        )
    _require_contiguous(norm_weight, name="norm_weight")


def _canonicalize_post_mix_input(
    post: torch.Tensor,
    *,
    tokens: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if post.dtype != torch.float32 or post.device != device:
        raise ValueError(
            f"{name} must be float32 on device {device}, got {post.dtype} on {post.device}"
        )
    if tuple(post.shape) == (tokens, MHC_MULT, 1):
        post = post.squeeze(-1)
    elif tuple(post.shape) != (tokens, MHC_MULT):
        raise ValueError(
            f"{name} must have shape {(tokens, MHC_MULT)} or {(tokens, MHC_MULT, 1)}, "
            f"got {tuple(post.shape)}"
        )
    _require_contiguous(post, name=name)
    return post


def _lagged_mix_views(
    pre_mix: torch.Tensor | None,
    pre_out: torch.Tensor | None,
    *,
    tokens: int,
    device: torch.device,
    outputs: tuple[torch.Tensor | None, ...],
    inputs: tuple[torch.Tensor | None, ...],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate the V4.1 ping-pong coefficient pair and every alias boundary."""
    if (pre_mix is None) != (pre_out is None):
        raise ValueError("pre_mix and caller-owned pre_out must be supplied together")
    if pre_mix is None:
        return None, None
    _validate_optional_view(
        pre_mix, shape=(tokens, MHC_MULT), dtype=torch.float32,
        device=device, name="pre_mix",
    )
    pre_out = _slice_capacity_view(
        pre_out, tokens=tokens, tail_shape=(MHC_MULT,), dtype=torch.float32,
        device=device, name="pre_out",
    )
    assert pre_out is not None
    if torch._C._overlaps(pre_mix, pre_out):
        raise ValueError("pre_mix and pre_out must not alias; use ping-pong buffers")
    for tensor in (*outputs, *inputs):
        if tensor is not None and (
            torch._C._overlaps(pre_mix, tensor) or torch._C._overlaps(pre_out, tensor)
        ):
            raise ValueError("lagged MHC coefficient buffers must not alias inputs, outputs, or scratch")
    return pre_mix, pre_out
def _validate_lagged_y_output(y, tensors):
    if any(tensor is not None and torch._C._overlaps(y, tensor) for tensor in tensors):
        raise ValueError("lagged mHC y output must not alias inputs, residual output, or scratch")


def _b12x_mhc_pre_impl(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    residual_out: torch.Tensor | None = None,
    y_out: torch.Tensor | None = None,
    post_out: torch.Tensor | None = None,
    comb_out: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    binding: B12XMHCBinding | None = None,
    pre_mix: torch.Tensor | None = None,
    pre_out: torch.Tensor | None = None,
    _state: _MhcState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _state.require("pre", residual, norm_weight=norm_weight,
                   rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters, norm_eps=norm_eps)
    split_k, block_k, block_h = _state.query.split_k, _state.query.block_k, _state.query.block_h
    partials = None
    if binding is not None:
        if binding.state is not _state:
            raise ValueError("binding and plan have different prepared states")
        extras = [
            name for name, value in (
                ("residual_out", residual_out), ("y_out", y_out),
                ("post_out", post_out), ("comb_out", comb_out), ("pre_out", pre_out),
            ) if value is not None
        ]
        if extras:
            raise ValueError("mHC binding owns scratch and output buffers; do not also pass " + ", ".join(extras))
        partials = binding.partials
        residual_out, y_out = binding.out, binding.y
        post_out, comb_out, pre_out = binding.post_buffer, binding.comb_buffer, binding.pre_out
        split_k = int(binding.state.query.split_k)

    tokens, hidden_size, _ = _validate_pre_inputs(
        residual, fn, hc_scale, hc_base, expanded=_state.query.expanded_residual,
    )
    if (residual.ndim == 3) != _state.query.expanded_residual:
        raise ValueError("MHC pre residual layout differs from preparation")
    _validate_norm_weight(norm_weight, hidden_size=hidden_size, device=residual.device)

    split_k, block_k, block_h = int(split_k), int(block_k), int(block_h)
    sinkhorn_iters = int(sinkhorn_iters)
    if sinkhorn_iters <= 0:
        raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
    if split_k <= 0:
        raise ValueError(f"split_k must be positive, got {split_k}")
    if block_k <= 0:
        raise ValueError(f"block_k must be positive, got {block_k}")
    if block_h <= 0:
        raise ValueError(f"block_h must be positive, got {block_h}")
    pre_mix, pre_out = _lagged_mix_views(
        pre_mix, pre_out, tokens=tokens, device=residual.device,
        outputs=(partials, residual_out, y_out, post_out, comb_out),
        inputs=(residual, fn, hc_scale, hc_base, norm_weight),
    )
    if (pre_mix is not None) != _state.query.lagged_mix:
        raise ValueError("MHC lagged coefficient presence differs from prepared invocation")

    if partials is None:
        partials = torch.empty(
            (tokens, split_k, MHC_PARTIALS),
            dtype=torch.float32,
            device=residual.device,
        )

    if partials is not None:
        partials = _slice_capacity_view(
            partials,
            tokens=tokens,
            tail_shape=(split_k, MHC_PARTIALS),
            dtype=torch.float32,
            device=residual.device,
            name="mHC partials",
        )
        if partials.dtype != torch.float32 or partials.device != residual.device:
            raise ValueError("mHC partials must be float32 on the residual device")
        _require_contiguous(partials, name="mHC partials")

    if residual_out is None:
        residual_out = torch.empty(
            (tokens, MHC_MULT, hidden_size),
            dtype=residual.dtype,
            device=residual.device,
        )
    else:
        residual_out = _slice_capacity_view(
            residual_out,
            tokens=tokens,
            tail_shape=(MHC_MULT, hidden_size),
            dtype=residual.dtype,
            device=residual.device,
            name="residual_out",
        )

    if y_out is None:
        y_out = torch.empty(
            (tokens, hidden_size), dtype=residual.dtype, device=residual.device
        )
    else:
        y_out = _slice_capacity_view(
            y_out,
            tokens=tokens,
            tail_shape=(hidden_size,),
            dtype=residual.dtype,
            device=residual.device,
            name="y_out",
        )
    if post_out is None:
        post_out = torch.empty(
            (tokens, MHC_MULT), dtype=torch.float32, device=residual.device
        )
    else:
        post_out = _slice_capacity_view(
            post_out,
            tokens=tokens,
            tail_shape=(MHC_MULT,),
            dtype=torch.float32,
            device=residual.device,
            name="post_out",
        )
    if comb_out is None:
        comb_out = torch.empty(
            (tokens, MHC_MULT, MHC_MULT), dtype=torch.float32, device=residual.device
        )
    else:
        comb_out = _slice_capacity_view(
            comb_out,
            tokens=tokens,
            tail_shape=(MHC_MULT, MHC_MULT),
            dtype=torch.float32,
            device=residual.device,
            name="comb_out",
        )

    if residual_out.shape != (tokens, MHC_MULT, hidden_size):
        raise ValueError("residual_out must have shape [tokens, 4, hidden_size]")
    if residual_out.dtype != residual.dtype or residual_out.device != residual.device:
        raise ValueError("residual_out must match the residual dtype and device")
    if (
        y_out.shape != (tokens, hidden_size)
        or y_out.dtype != residual.dtype
        or y_out.device != residual.device
    ):
        raise ValueError(
            "y_out must match shape [tokens, hidden_size], residual dtype, and residual device"
        )
    if (
        post_out.shape != (tokens, MHC_MULT)
        or post_out.dtype != torch.float32
        or post_out.device != residual.device
    ):
        raise ValueError(
            "post_out must match shape [tokens, 4], dtype float32, and residual device"
        )
    if (
        comb_out.shape != (tokens, MHC_MULT, MHC_MULT)
        or comb_out.dtype != torch.float32
        or comb_out.device != residual.device
    ):
        raise ValueError(
            "comb_out must match shape [tokens, 4, 4], dtype float32, and residual device"
        )
    _require_contiguous(residual_out, name="residual_out")
    _require_contiguous(y_out, name="y_out")
    _require_contiguous(post_out, name="post_out")
    _require_contiguous(comb_out, name="comb_out")

    if tokens == 0:
        return residual_out, post_out, comb_out, y_out

    if _state.query.lagged_mix:
        assert pre_mix is not None and pre_out is not None
        if _state.config.lagged_prepare:
            _validate_lagged_y_output(y_out, (residual, fn, hc_scale, hc_base, norm_weight, residual_out, partials))
        _state.launchers["partial"](
            residual=residual, fn=fn, partials=partials, out=residual_out,
            pre_mix=pre_mix, y=y_out,
        )
        _state.launchers["finalize"](
            residual=residual_out, partials=partials, scale=hc_scale, bias=hc_base,
            y=y_out, post=post_out, comb=comb_out, norm_weight=norm_weight,
            pre_mix=pre_mix, pre_out=pre_out,
        )
        return residual_out, post_out, comb_out, y_out
    if (
        partials is not None
        and _supports_fused_mhc_gram(
            hidden_size=hidden_size,
            split_k=split_k,
            block_k=block_k,
            block_h=block_h,
        )
        and float(rms_eps) in MHC_SUPPORTED_RMS_EPS
        and float(hc_eps) == 1.0e-6
        and sinkhorn_iters == 20
    ):
        _state.launchers["partial"](
            residual=residual, fn=fn, partials=partials, out=residual_out,
        )
        _state.launchers["finalize"](
            residual=residual_out, partials=partials, scale=hc_scale, bias=hc_base,
            y=y_out, post=post_out, comb=comb_out, norm_weight=norm_weight,
        )
        return residual_out, post_out, comb_out, y_out

    raise ValueError(
        "b12x_mhc_pre is served only by the fused Gram kernel, which "
        "supports the decode config "
        f"(hidden_size divisible by {MHC_GRAM_BLOCK_H}, "
        f"split_k=hc_mult*hidden_size/{MHC_DEFAULT_BLOCK_K}, "
        f"block_k={MHC_DEFAULT_BLOCK_K}, block_h={MHC_DEFAULT_BLOCK_H}, "
        f"rms_eps in {MHC_SUPPORTED_RMS_EPS}, hc_eps=1e-06, "
        "sinkhorn_iters=20); got "
        f"hidden_size={hidden_size}, split_k={split_k}, block_k={block_k}, "
        f"block_h={block_h}, rms_eps={rms_eps}, hc_eps={hc_eps}, "
        f"sinkhorn_iters={sinkhorn_iters}"
    )


def _b12x_mhc_post_pre_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    residual_out: torch.Tensor | None = None,
    y_out: torch.Tensor | None = None,
    post_out: torch.Tensor | None = None,
    comb_out: torch.Tensor | None = None,
    fn_bf16: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    binding: B12XMHCBinding | None = None,
    pre_mix: torch.Tensor | None = None,
    pre_out: torch.Tensor | None = None,
    _state: _MhcState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _state.require(
        "post_pre", residual, norm_weight=norm_weight, fn_bf16=fn_bf16,
        rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters, norm_eps=norm_eps,
    )
    split_k, block_k, block_h = _state.query.split_k, _state.query.block_k, _state.query.block_h
    expected_m = _state.query.max_tokens
    partials = None
    if binding is not None:
        if binding.state is not _state:
            raise ValueError("binding and plan have different prepared states")
        extras = [
            name
            for name, value in (
                ("residual_out", residual_out),
                ("y_out", y_out),
                ("post_out", post_out),
                ("comb_out", comb_out),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "mHC binding owns scratch and output buffers; "
                f"do not also pass {', '.join(extras)}"
            )
        partials = binding.partials
        residual_out = binding.out
        y_out = binding.y
        post_out = binding.post_buffer
        comb_out = binding.comb_buffer
        split_k = int(binding.state.query.split_k)
        pre_out = binding.pre_out
        if (
            expected_m is not None
            and binding.expected_m is not None
            and int(expected_m) != int(binding.expected_m)
        ):
            raise ValueError(
                "expected_m does not match the mHC binding's expected_m: "
                f"{expected_m} vs {binding.expected_m}"
            )
        if expected_m is None:
            expected_m = binding.expected_m

    tokens, hidden_size, _ = _validate_post_pre_inputs(residual, fn, hc_scale, hc_base)
    expected_m = _canonicalize_mhc_expected_m(expected_m, min_tokens=tokens)
    _validate_norm_weight(norm_weight, hidden_size=hidden_size, device=residual.device)
    pre_mix, pre_out = _lagged_mix_views(
        pre_mix, pre_out, tokens=tokens, device=residual.device,
        outputs=(partials, residual_out, y_out, post_out, comb_out),
        inputs=(x, residual, prev_post, prev_comb, fn, hc_scale, hc_base, norm_weight),
    )
    if (pre_mix is not None) != _state.query.lagged_mix:
        raise ValueError("MHC lagged coefficient presence differs from prepared invocation")
    if x.dtype != residual.dtype or x.dtype != torch.bfloat16:
        raise ValueError(
            f"x and residual must both be torch.bfloat16, got {x.dtype} and {residual.dtype}"
        )
    if x.ndim != 2 or tuple(x.shape) != (tokens, hidden_size):
        raise ValueError(
            f"x must have shape {(tokens, hidden_size)}, got {tuple(x.shape)}"
        )
    if x.device != residual.device:
        raise ValueError(
            "x, residual, fn, hc_scale, and hc_base must be on the same device"
        )
    prev_post = _canonicalize_post_mix_input(
        prev_post,
        tokens=tokens,
        device=residual.device,
        name="prev_post",
    )
    if prev_comb.dtype != torch.float32 or tuple(prev_comb.shape) != (
        tokens,
        MHC_MULT,
        MHC_MULT,
    ):
        raise ValueError(
            f"prev_comb must be float32 shape {(tokens, MHC_MULT, MHC_MULT)}, "
            f"got {prev_comb.dtype} {tuple(prev_comb.shape)}"
        )
    if prev_comb.device != residual.device:
        raise ValueError("prev_comb must be on the residual device")
    _require_contiguous(x, name="x")
    _require_contiguous(prev_comb, name="prev_comb")

    split_k = int(split_k)
    block_k = int(block_k)
    block_h = int(block_h)
    sinkhorn_iters = int(sinkhorn_iters)
    if sinkhorn_iters <= 0:
        raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
    if split_k <= 0:
        raise ValueError(f"split_k must be positive, got {split_k}")
    if block_k <= 0:
        raise ValueError(f"block_k must be positive, got {block_k}")
    if block_h <= 0:
        raise ValueError(f"block_h must be positive, got {block_h}")

    if partials is None:
        # The Gram post_pre needs a partials scratch buffer (the launch boundary
        # between the partial pass and the multi-CTA finalize).
        partials = torch.empty(
            (tokens, split_k, MHC_PARTIALS), dtype=torch.float32, device=residual.device
        )
    # The source-tile CuTe post_pre path uses the shared scratch partials as
    # the launch boundary between post+partial reduction and y/post/comb finalize.
    if partials is not None:
        partials = _slice_capacity_view(
            partials,
            tokens=tokens,
            tail_shape=(split_k, MHC_PARTIALS),
            dtype=torch.float32,
            device=residual.device,
            name="mHC partials",
        )
        if partials.dtype != torch.float32 or partials.device != residual.device:
            raise ValueError("mHC partials must be float32 on the residual device")
        _require_contiguous(partials, name="mHC partials")

    if residual_out is None:
        residual_out = torch.empty_like(residual)
    else:
        residual_out = _slice_capacity_view(
            residual_out,
            tokens=tokens,
            tail_shape=(MHC_MULT, hidden_size),
            dtype=residual.dtype,
            device=residual.device,
            name="residual_out",
        )
    if y_out is None:
        y_out = torch.empty(
            (tokens, hidden_size), dtype=residual.dtype, device=residual.device
        )
    else:
        y_out = _slice_capacity_view(
            y_out,
            tokens=tokens,
            tail_shape=(hidden_size,),
            dtype=residual.dtype,
            device=residual.device,
            name="y_out",
        )
    if post_out is None:
        post_out = torch.empty(
            (tokens, MHC_MULT), dtype=torch.float32, device=residual.device
        )
    else:
        post_out = _slice_capacity_view(
            post_out,
            tokens=tokens,
            tail_shape=(MHC_MULT,),
            dtype=torch.float32,
            device=residual.device,
            name="post_out",
        )
    if comb_out is None:
        comb_out = torch.empty(
            (tokens, MHC_MULT, MHC_MULT), dtype=torch.float32, device=residual.device
        )
    else:
        comb_out = _slice_capacity_view(
            comb_out,
            tokens=tokens,
            tail_shape=(MHC_MULT, MHC_MULT),
            dtype=torch.float32,
            device=residual.device,
            name="comb_out",
        )

    if (
        residual_out.shape != residual.shape
        or residual_out.dtype != residual.dtype
        or residual_out.device != residual.device
    ):
        raise ValueError("residual_out must match residual shape, dtype, and device")
    if (
        y_out.shape != (tokens, hidden_size)
        or y_out.dtype != residual.dtype
        or y_out.device != residual.device
    ):
        raise ValueError(
            "y_out must match shape [tokens, hidden_size], residual dtype, and residual device"
        )
    if (
        post_out.shape != (tokens, MHC_MULT)
        or post_out.dtype != torch.float32
        or post_out.device != residual.device
    ):
        raise ValueError(
            "post_out must match shape [tokens, 4], dtype float32, and residual device"
        )
    if (
        comb_out.shape != (tokens, MHC_MULT, MHC_MULT)
        or comb_out.dtype != torch.float32
        or comb_out.device != residual.device
    ):
        raise ValueError(
            "comb_out must match shape [tokens, 4, 4], dtype float32, and residual device"
        )
    _require_contiguous(residual_out, name="residual_out")
    _require_contiguous(y_out, name="y_out")
    _require_contiguous(post_out, name="post_out")
    _require_contiguous(comb_out, name="comb_out")

    if tokens == 0:
        return residual_out, post_out, comb_out, y_out

    if _state.query.lagged_mix:
        assert pre_mix is not None and pre_out is not None
        if _state.config.lagged_prepare:
            _validate_lagged_y_output(y_out, (x, residual, prev_post, prev_comb, fn, hc_scale, hc_base, norm_weight, residual_out, partials))
        _state.launchers["partial"](
            x=x, residual=residual, prev_post=prev_post, prev_comb=prev_comb,
            fn=fn, fn_bf16=fn_bf16, partials=partials, out=residual_out, pre_mix=pre_mix, y=y_out,
        )
        _state.launchers["finalize"](
            residual=residual_out, partials=partials, scale=hc_scale, bias=hc_base,
            y=y_out, post=post_out, comb=comb_out, norm_weight=norm_weight,
            pre_mix=pre_mix, pre_out=pre_out,
        )
        return residual_out, post_out, comb_out, y_out
    if (
        partials is not None
        and _supports_fused_mhc_gram(
            hidden_size=hidden_size, split_k=split_k, block_k=block_k, block_h=block_h,
        )
        and float(rms_eps) in MHC_SUPPORTED_RMS_EPS
        and float(hc_eps) == 1.0e-6
        and sinkhorn_iters == 20
    ):
        _state.launchers["partial"](
            x=x, residual=residual, prev_post=prev_post, prev_comb=prev_comb,
            fn=fn, fn_bf16=fn_bf16, partials=partials, out=residual_out,
        )
        _state.launchers["finalize"](
            residual=residual_out, partials=partials, scale=hc_scale, bias=hc_base,
            y=y_out, post=post_out, comb=comb_out, norm_weight=norm_weight,
        )
        return residual_out, post_out, comb_out, y_out

    raise ValueError(
        "b12x_mhc_post_pre is served only by the fused Gram kernel, which "
        "supports the decode config "
        f"(hidden_size divisible by {MHC_GRAM_BLOCK_H}, "
        f"split_k=hc_mult*hidden_size/{MHC_DEFAULT_BLOCK_K}, "
        f"block_k={MHC_DEFAULT_BLOCK_K}, block_h={MHC_DEFAULT_BLOCK_H}, "
        f"rms_eps in {MHC_SUPPORTED_RMS_EPS}, hc_eps=1e-06, "
        "sinkhorn_iters=20); got "
        f"hidden_size={hidden_size}, split_k={split_k}, block_k={block_k}, "
        f"block_h={block_h}, rms_eps={rms_eps}, hc_eps={hc_eps}, "
        f"sinkhorn_iters={sinkhorn_iters}"
    )


def _b12x_mhc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    _state: _MhcState,
) -> torch.Tensor:
    _state.require("post", residual)
    tokens, hc_mult, hidden_size = residual.shape
    if hc_mult != MHC_MULT:
        raise ValueError(f"residual hc dimension must be {MHC_MULT}, got {hc_mult}")
    if x.dtype != residual.dtype or x.dtype != torch.bfloat16:
        raise ValueError(
            f"x and residual must both be torch.bfloat16, got {x.dtype} and "
            f"{residual.dtype}"
        )
    if x.ndim != 2 or tuple(x.shape) != (tokens, hidden_size):
        raise ValueError(
            f"x must have shape {(tokens, hidden_size)}, got {tuple(x.shape)}"
        )
    if x.device != residual.device:
        raise ValueError("x and residual must be on the same device")
    prev_post = _canonicalize_post_mix_input(
        prev_post,
        tokens=tokens,
        device=residual.device,
        name="prev_post",
    )
    if prev_comb.dtype != torch.float32 or tuple(prev_comb.shape) != (
        tokens,
        MHC_MULT,
        MHC_MULT,
    ):
        raise ValueError(
            f"prev_comb must be float32 shape {(tokens, MHC_MULT, MHC_MULT)}, "
            f"got {prev_comb.dtype} {tuple(prev_comb.shape)}"
        )
    if prev_comb.device != residual.device:
        raise ValueError("prev_comb must be on the residual device")
    _require_contiguous(x, name="x")
    _require_contiguous(residual, name="residual")
    _require_contiguous(prev_comb, name="prev_comb")

    if out is None:
        out = torch.empty_like(residual)

    out = _slice_capacity_view(
        out,
        tokens=tokens,
        tail_shape=(MHC_MULT, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
        name="out",
    )
    if (
        out.shape != residual.shape
        or out.dtype != residual.dtype
        or out.device != residual.device
    ):
        raise ValueError("out must match residual shape, dtype, and device")
    _require_contiguous(out, name="out")

    if tokens == 0:
        return out
    if _supports_mhc_post_hidden(hidden_size):
        _state.launchers["post"](
            x=x, residual=residual, prev_post=prev_post, prev_comb=prev_comb, out=out,
        )
        return out

    raise ValueError(
        "b12x_mhc_post is served only by the post-only mHC kernel, which "
        f"supports hidden_size in {MHC_SUPPORTED_HIDDEN_SIZES}; "
        f"got hidden_size={hidden_size}"
    )


def _prepared_state(plan, *, output_mode):
    state = require_prepared(plan, "norm.mhc")
    if state.query.output_mode != output_mode:
        raise ValueError("MHC output form differs from prepared invocation")
    return state


def _functional_output_metadata(residual):
    tokens, hidden = residual.shape[0], residual.shape[-1]
    return (
        torch.empty((tokens, MHC_MULT, hidden), dtype=residual.dtype, device=residual.device),
        torch.empty((tokens, MHC_MULT), dtype=torch.float32, device=residual.device),
        torch.empty((tokens, MHC_MULT, MHC_MULT), dtype=torch.float32, device=residual.device),
        torch.empty((tokens, hidden), dtype=residual.dtype, device=residual.device),
    )


@torch.library.custom_op("b12x::mhc_pre_planned_functional", mutates_args=())
def _mhc_pre_planned_functional_op(
    residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, norm_weight: torch.Tensor | None,
    rms_eps: float, hc_eps: float, sinkhorn_iters: int, norm_eps: float,
    plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _b12x_mhc_pre_impl(
        residual, fn, hc_scale, hc_base, norm_weight=norm_weight,
        rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters, norm_eps=norm_eps,
        _state=_prepared_state(plan_from_handle(plan_handle), output_mode="functional"),
    )


@_mhc_pre_planned_functional_op.register_fake
def _mhc_pre_planned_functional_fake(
    residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, norm_weight: torch.Tensor | None,
    rms_eps: float, hc_eps: float, sinkhorn_iters: int, norm_eps: float,
    plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _functional_output_metadata(residual)
@torch.library.custom_op(
    "b12x::mhc_pre_lagged_planned_functional", mutates_args=("pre_out",)
)
def _mhc_pre_lagged_planned_functional_op(
    residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, pre_mix: torch.Tensor, pre_out: torch.Tensor,
    norm_weight: torch.Tensor | None, rms_eps: float, hc_eps: float,
    sinkhorn_iters: int, norm_eps: float, plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _b12x_mhc_pre_impl(
        residual, fn, hc_scale, hc_base, pre_mix=pre_mix, pre_out=pre_out,
        norm_weight=norm_weight, rms_eps=rms_eps, hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters, norm_eps=norm_eps,
        _state=_prepared_state(plan_from_handle(plan_handle), output_mode="functional"),
    )


@_mhc_pre_lagged_planned_functional_op.register_fake
def _mhc_pre_lagged_planned_functional_fake(
    residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, pre_mix: torch.Tensor, pre_out: torch.Tensor,
    norm_weight: torch.Tensor | None, rms_eps: float, hc_eps: float,
    sinkhorn_iters: int, norm_eps: float, plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _functional_output_metadata(residual)


@torch.library.custom_op("b12x::mhc_post_pre_planned_functional", mutates_args=())
def _mhc_post_pre_planned_functional_op(
    x: torch.Tensor, residual: torch.Tensor, prev_post: torch.Tensor,
    prev_comb: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, fn_bf16: torch.Tensor | None, norm_weight: torch.Tensor | None,
    rms_eps: float, hc_eps: float, sinkhorn_iters: int, norm_eps: float,
    plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _b12x_mhc_post_pre_impl(
        x, residual, prev_post, prev_comb, fn, hc_scale, hc_base,
        norm_weight=norm_weight, fn_bf16=fn_bf16,
        rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters, norm_eps=norm_eps,
        _state=_prepared_state(plan_from_handle(plan_handle), output_mode="functional"),
    )


@_mhc_post_pre_planned_functional_op.register_fake
def _mhc_post_pre_planned_functional_fake(
    x: torch.Tensor, residual: torch.Tensor, prev_post: torch.Tensor,
    prev_comb: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, fn_bf16: torch.Tensor | None, norm_weight: torch.Tensor | None,
    rms_eps: float, hc_eps: float, sinkhorn_iters: int, norm_eps: float,
    plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _functional_output_metadata(residual)

@torch.library.custom_op(
    "b12x::mhc_post_pre_lagged_planned_functional", mutates_args=("pre_out",)
)
def _mhc_post_pre_lagged_planned_functional_op(
    x: torch.Tensor, residual: torch.Tensor, prev_post: torch.Tensor,
    prev_comb: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, fn_bf16: torch.Tensor | None, pre_mix: torch.Tensor,
    pre_out: torch.Tensor, norm_weight: torch.Tensor | None, rms_eps: float,
    hc_eps: float, sinkhorn_iters: int, norm_eps: float, plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _b12x_mhc_post_pre_impl(
        x, residual, prev_post, prev_comb, fn, hc_scale, hc_base,
        fn_bf16=fn_bf16, pre_mix=pre_mix, pre_out=pre_out,
        norm_weight=norm_weight, rms_eps=rms_eps, hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters, norm_eps=norm_eps,
        _state=_prepared_state(plan_from_handle(plan_handle), output_mode="functional"),
    )


@_mhc_post_pre_lagged_planned_functional_op.register_fake
def _mhc_post_pre_lagged_planned_functional_fake(
    x: torch.Tensor, residual: torch.Tensor, prev_post: torch.Tensor,
    prev_comb: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, fn_bf16: torch.Tensor | None, pre_mix: torch.Tensor,
    pre_out: torch.Tensor, norm_weight: torch.Tensor | None, rms_eps: float,
    hc_eps: float, sinkhorn_iters: int, norm_eps: float, plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _functional_output_metadata(residual)


@torch.library.custom_op("b12x::mhc_post_planned_functional", mutates_args=())
def _mhc_post_planned_functional_op(
    x: torch.Tensor, residual: torch.Tensor, prev_post: torch.Tensor,
    prev_comb: torch.Tensor, plan_handle: int,
) -> torch.Tensor:
    return _b12x_mhc_post_impl(
        x, residual, prev_post, prev_comb,
        _state=_prepared_state(plan_from_handle(plan_handle), output_mode="functional"),
    )


@_mhc_post_planned_functional_op.register_fake
def _mhc_post_planned_functional_fake(
    x: torch.Tensor, residual: torch.Tensor, prev_post: torch.Tensor,
    prev_comb: torch.Tensor, plan_handle: int,
) -> torch.Tensor:
    return torch.empty_like(residual)


@torch.library.custom_op("b12x::mhc_collapse", mutates_args=("out",))
def _mhc_collapse_op(
    state: torch.Tensor, pre_mix: torch.Tensor | None, out: torch.Tensor,
    plan_handle: int,
) -> None:
    _run_collapse_impl(
        state, pre_mix, out=out, _state=_prepared_state(plan_from_handle(plan_handle), output_mode="provided"),
    )


@_mhc_collapse_op.register_fake
def _mhc_collapse_fake(
    state: torch.Tensor, pre_mix: torch.Tensor | None, out: torch.Tensor,
    plan_handle: int,
) -> None:
    del state, pre_mix, out, plan_handle


def _plan_from_binding(plan, binding):
    if binding is not None:
        if plan is not None:
            raise ValueError("pass a plan or a binding, not both")
        plan = binding.plan
    if plan is None:
        raise TypeError("MHC requires a prepared Plan or a prepared binding")
    return plan


def b12x_mhc_pre(
    residual, fn, hc_scale, hc_base, *, rms_eps, hc_eps, sinkhorn_iters,
    plan: Plan | None = None,
    residual_out=None, y_out=None, post_out=None, comb_out=None, pre_mix=None, pre_out=None,
    norm_weight=None, norm_eps=0.0, binding: B12XMHCBinding | None = None,
):
    plan = _plan_from_binding(plan, binding)
    functional = binding is None and all(
        value is None for value in (residual_out, y_out, post_out, comb_out)
    )
    if functional and (pre_mix is None) != (pre_out is None):
        raise ValueError("pre_mix and caller-owned pre_out must be supplied together")
    if functional and pre_mix is not None:
        return torch.ops.b12x.mhc_pre_lagged_planned_functional(
            residual, fn, hc_scale, hc_base, pre_mix, pre_out, norm_weight,
            float(rms_eps), float(hc_eps), int(sinkhorn_iters), float(norm_eps), plan.handle,
        )
    if functional:
        return torch.ops.b12x.mhc_pre_planned_functional(
            residual, fn, hc_scale, hc_base, norm_weight,
            float(rms_eps), float(hc_eps), int(sinkhorn_iters), float(norm_eps), plan.handle,
        )
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned MHC buffers are not supported inside Dynamo")
    return _b12x_mhc_pre_impl(
        residual, fn, hc_scale, hc_base, rms_eps=rms_eps, hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters, pre_mix=pre_mix, pre_out=pre_out,
        norm_weight=norm_weight, norm_eps=norm_eps,
        residual_out=residual_out, y_out=y_out, post_out=post_out, comb_out=comb_out,
        binding=binding, _state=_prepared_state(plan, output_mode="provided"),
    )


def b12x_mhc_post_pre(
    x, residual, prev_post, prev_comb, fn, hc_scale, hc_base, *,
    rms_eps, hc_eps, sinkhorn_iters, plan: Plan | None = None,
    residual_out=None, y_out=None, post_out=None, comb_out=None, pre_mix=None, pre_out=None,
    fn_bf16=None, norm_weight=None, norm_eps=0.0, binding: B12XMHCBinding | None = None,
):
    plan = _plan_from_binding(plan, binding)
    functional = binding is None and all(
        value is None for value in (residual_out, y_out, post_out, comb_out)
    )
    if functional and (pre_mix is None) != (pre_out is None):
        raise ValueError("pre_mix and caller-owned pre_out must be supplied together")
    if functional and pre_mix is not None:
        return torch.ops.b12x.mhc_post_pre_lagged_planned_functional(
            x, residual, prev_post, prev_comb, fn, hc_scale, hc_base, fn_bf16,
            pre_mix, pre_out, norm_weight, float(rms_eps), float(hc_eps),
            int(sinkhorn_iters), float(norm_eps), plan.handle,
        )
    if functional:
        return torch.ops.b12x.mhc_post_pre_planned_functional(
            x, residual, prev_post, prev_comb, fn, hc_scale, hc_base, fn_bf16, norm_weight,
            float(rms_eps), float(hc_eps), int(sinkhorn_iters), float(norm_eps), plan.handle,
        )
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned MHC buffers are not supported inside Dynamo")
    return _b12x_mhc_post_pre_impl(
        x, residual, prev_post, prev_comb, fn, hc_scale, hc_base,
        rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters,
        residual_out=residual_out, y_out=y_out, post_out=post_out, comb_out=comb_out,
        pre_mix=pre_mix, pre_out=pre_out, fn_bf16=fn_bf16,
        norm_weight=norm_weight, norm_eps=norm_eps, binding=binding,
        _state=_prepared_state(plan, output_mode="provided"),
    )


def b12x_mhc_post(x, residual, prev_post, prev_comb, *, plan: Plan, out=None):
    if out is None:
        return torch.ops.b12x.mhc_post_planned_functional(x, residual, prev_post, prev_comb, plan.handle)
    if torch.compiler.is_compiling():
        raise RuntimeError("caller-owned MHC outputs are not supported inside Dynamo")
    return _b12x_mhc_post_impl(
        x, residual, prev_post, prev_comb, out=out,
        _state=_prepared_state(plan, output_mode="provided"),
    )


def run_collapse(state, pre_mix, *, out, plan: Plan):
    torch.ops.b12x.mhc_collapse(state, pre_mix, out, plan.handle)
    return out


__all__ = [
    "B12XMHCBinding", "B12XMHCScratchCaps", "MHC_DEFAULT_BLOCK_H",
    "MHC_DEFAULT_BLOCK_K", "MHC_DEFAULT_SPLIT_K", "MHC_GRAM_BLOCK_H",
    "MHC_MULT", "MHC_MIXES", "MHC_PARTIALS", "MHC_SOURCE_TILE_H",
    "MHC_SUPPORTED_HIDDEN_SIZES", "b12x_mhc_post", "b12x_mhc_pre",
    "b12x_mhc_post_pre", "run_collapse",
]
