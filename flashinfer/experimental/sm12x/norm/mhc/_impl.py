# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/residual.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""CuTeDSL mHC residual helpers for DeepSeek-style residual mixing."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from flashinfer.experimental.sm12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from flashinfer.experimental.sm12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)


MHC_MULT = 4
MHC_MIXES = (2 + MHC_MULT) * MHC_MULT
MHC_PARTIALS = 1 + MHC_MIXES
MHC_DEFAULT_SPLIT_K = 64
MHC_DEFAULT_BLOCK_K = 256
MHC_DEFAULT_BLOCK_H = 512
MHC_SOURCE_TILE_H = 128
MHC_GRAM_BLOCK_H = 1024
MHC_SUPPORTED_HIDDEN_SIZES = (4096, 7168)


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
    return int(hidden_size) in MHC_SUPPORTED_HIDDEN_SIZES


@dataclass(frozen=True, kw_only=True)
class SM12XMHCBinding:
    partials: torch.Tensor | None = None
    y: torch.Tensor | None = None
    post_buffer: torch.Tensor | None = None
    comb_buffer: torch.Tensor | None = None
    out: torch.Tensor | None = None
    split_k: int = MHC_DEFAULT_SPLIT_K
    expected_m: int | None = None

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
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        block_k: int = MHC_DEFAULT_BLOCK_K,
        block_h: int = MHC_DEFAULT_BLOCK_H,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return sm12x_mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            binding=self,
            block_k=block_k,
            block_h=block_h,
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
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        fn_bf16: torch.Tensor | None = None,
        expected_m: int | None = None,
        block_k: int = MHC_DEFAULT_BLOCK_K,
        block_h: int = MHC_DEFAULT_BLOCK_H,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return sm12x_mhc_post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            fn_bf16=fn_bf16,
            expected_m=expected_m,
            binding=self,
            block_k=block_k,
            block_h=block_h,
        )


@dataclass(frozen=True, kw_only=True)
class SM12XMHCScratchCaps:
    device: torch.device | str
    max_tokens: int
    hidden_size: int
    dtype: torch.dtype = torch.bfloat16
    split_k: int = MHC_DEFAULT_SPLIT_K

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "hidden_size", max(int(self.hidden_size), 1))
        object.__setattr__(self, "split_k", max(int(self.split_k), 1))
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"mHC scratch currently supports torch.bfloat16 outputs, got {self.dtype}"
            )


@dataclass(frozen=True)
class _MHCScratchLayout:
    nbytes: int
    partials_offset_bytes: int


@dataclass(frozen=True)
class SM12XMHCScratchPlan:
    caps: SM12XMHCScratchCaps
    layout: _MHCScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

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
    ) -> SM12XMHCBinding:
        live_tokens = int(self.caps.max_tokens) if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > int(self.caps.max_tokens):
            raise ValueError(
                f"tokens={live_tokens} exceeds MHC scratch capacity {self.caps.max_tokens}"
            )
        expected_m = _canonicalize_mhc_expected_m(
            expected_m,
            min_tokens=live_tokens,
            max_tokens=int(self.caps.max_tokens),
        )
        partials = self._partials_from_scratch(scratch=scratch)[:live_tokens]
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
        return SM12XMHCBinding(
            partials=partials,
            y=y,
            post_buffer=post,
            comb_buffer=comb,
            out=out,
            split_k=int(self.caps.split_k),
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


def _use_mhc_prefill_bf16_project(
    *,
    norm_weight: torch.Tensor | None,
    policy_m: int,
    fn_bf16: torch.Tensor | None,
) -> bool:
    prefill_bf16_min_tokens = int(
        os.environ.get("FLASHINFER_EXP_SM12X_MHC_PREFILL_BF16_MIN_TOKENS", "384")
    )
    return (
        norm_weight is not None
        and int(policy_m) >= prefill_bf16_min_tokens
        and fn_bf16 is not None
        and os.environ.get("FLASHINFER_EXP_SM12X_MHC_PREFILL_BF16_MMA", "1") != "0"
    )


def _use_mhc_prefill_tf32_project(
    *,
    norm_weight: torch.Tensor | None,
    policy_m: int,
) -> bool:
    prefill_bf16_min_tokens = os.environ.get(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_BF16_MIN_TOKENS", "384"
    )
    prefill_tf32_min_tokens = int(
        os.environ.get(
            "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_MIN_TOKENS", prefill_bf16_min_tokens
        )
    )
    prefill_tf32_enabled = os.environ.get(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_MMA",
        os.environ.get("FLASHINFER_EXP_SM12X_MHC_PREFILL_BF16_MMA", "1"),
    )
    return (
        norm_weight is not None
        and int(policy_m) >= prefill_tf32_min_tokens
        and prefill_tf32_enabled != "0"
    )


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


def _layout_mhc_scratch(caps: SM12XMHCScratchCaps) -> _MHCScratchLayout:
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


def plan_mhc_scratch(caps: SM12XMHCScratchCaps) -> SM12XMHCScratchPlan:
    layout = _layout_mhc_scratch(caps)
    return SM12XMHCScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "mhc.scratch",
                nbytes=layout.nbytes,
                device=caps.device,
            ),
        ),
    )


def _require_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_pre_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> tuple[int, int, int]:
    if residual.device.type != "cuda":
        raise ValueError("residual must be a CUDA tensor")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be torch.bfloat16, got {residual.dtype}")
    if residual.ndim != 2:
        raise ValueError(
            f"residual must be rank-2 [tokens, hidden], got {tuple(residual.shape)}"
        )
    tokens, hidden_size = map(int, residual.shape)
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if fn.dtype != torch.float32:
        raise ValueError(f"fn must be torch.float32, got {fn.dtype}")
    if fn.shape != (MHC_MIXES, hidden_size):
        raise ValueError(
            f"fn must have shape {(MHC_MIXES, hidden_size)}, got {tuple(fn.shape)}"
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


def _sm12x_mhc_pre_impl(
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
    binding: SM12XMHCBinding | None = None,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    block_k: int = MHC_DEFAULT_BLOCK_K,
    block_h: int = MHC_DEFAULT_BLOCK_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # See sm12x_mhc_post_pre: caller-owned buffers -> mutating ops; no buffers
    # (e.g. torch.compile) -> a single functional op (allocate + return) so the
    # compile graph has zero auto_functionalized mHC nodes. No is_compiling.
    _caller_owned_buffers = (
        binding is not None
        or residual_out is not None
        or y_out is not None
        or post_out is not None
        or comb_out is not None
    )
    partials = None
    if binding is not None:
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
        split_k = int(binding.split_k)

    tokens, hidden_size, _ = _validate_pre_inputs(residual, fn, hc_scale, hc_base)
    _validate_norm_weight(norm_weight, hidden_size=hidden_size, device=residual.device)

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

    if (
        partials is not None
        and _supports_fused_mhc_gram(
            hidden_size=hidden_size,
            split_k=split_k,
            block_k=block_k,
            block_h=block_h,
        )
        and float(rms_eps) == 1.0e-6
        and float(hc_eps) == 1.0e-6
        and sinkhorn_iters == 20
    ):
        from flashinfer.experimental.sm12x.norm.mhc._kernels import (
            run_mhc_finalize_gram,
            run_mhc_pre_functional,
            run_mhc_pre_partial,
        )

        if not _caller_owned_buffers:
            # No caller buffers (e.g. torch.compile): one functional op runs the
            # whole pre (partial + finalize) and returns fresh tensors, so the
            # compile graph carries ZERO auto_functionalized mHC nodes.
            return run_mhc_pre_functional(
                residual=residual,
                fn=fn,
                scale=hc_scale,
                bias=hc_base,
                rms_eps=float(rms_eps),
                hc_eps=float(hc_eps),
                sinkhorn_iters=sinkhorn_iters,
                norm_weight=norm_weight,
                norm_eps=float(norm_eps),
            )

        run_mhc_pre_partial(
            residual=residual,
            fn=fn,
            partials=partials,
            out=residual_out,
            compute_gram=norm_weight is not None,
        )
        run_mhc_finalize_gram(
            residual=residual_out,
            partials=partials,
            scale=hc_scale,
            bias=hc_base,
            y=y_out,
            post=post_out,
            comb=comb_out,
            rms_eps=float(rms_eps),
            hc_eps=float(hc_eps),
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=float(norm_eps),
        )
        return residual_out, post_out, comb_out, y_out

    raise ValueError(
        "sm12x_mhc_pre is served only by the fused Gram kernel, which "
        "supports the decode config "
        f"(hidden_size divisible by {MHC_GRAM_BLOCK_H}, "
        f"split_k=hc_mult*hidden_size/{MHC_DEFAULT_BLOCK_K}, "
        f"block_k={MHC_DEFAULT_BLOCK_K}, block_h={MHC_DEFAULT_BLOCK_H}, "
        "sinkhorn_iters=20); got "
        f"hidden_size={hidden_size}, split_k={split_k}, block_k={block_k}, "
        f"block_h={block_h}, sinkhorn_iters={sinkhorn_iters}"
    )


def _sm12x_mhc_post_pre_impl(
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
    expected_m: int | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    binding: SM12XMHCBinding | None = None,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    block_k: int = MHC_DEFAULT_BLOCK_K,
    block_h: int = MHC_DEFAULT_BLOCK_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Whether the caller owns the output/scratch buffers (binding or
    # explicit out tensors). When it does, the post-pre partial writes them in
    # place (mutating op). When it does not and no expected_m policy is supplied
    # (e.g. the torch.compile path passes nothing), we use the functional alloc
    # op: it returns partials+residual_out with ZERO mutated args, avoiding the
    # auto_functionalized 2-mutated-arg decomposition assertion.
    _caller_owned_buffers = (
        binding is not None
        or residual_out is not None
        or y_out is not None
        or post_out is not None
        or comb_out is not None
    )
    partials = None
    if binding is not None:
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
        split_k = int(binding.split_k)
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
    policy_m = tokens if expected_m is None else expected_m
    _validate_norm_weight(norm_weight, hidden_size=hidden_size, device=residual.device)
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

    if (
        partials is not None
        and _supports_fused_mhc_gram(
            hidden_size=hidden_size,
            split_k=split_k,
            block_k=block_k,
            block_h=block_h,
        )
        and float(rms_eps) == 1.0e-6
        and float(hc_eps) == 1.0e-6
        and sinkhorn_iters == 20
    ):
        # The Gram-trick fused post_pre is THE mHC decode post_pre kernel: one
        # partial pass (POST + the fn@flat reduction + residual_out's 4x4 Gram)
        # feeds a multi-CTA finalize whose RMSNorm uses sum_h y^2 = pre^T G pre
        # (no per-hidden reduction, so no single-CTA bottleneck). Sinkhorn runs
        # the caller's iteration count (the full 20, matching vLLM and the
        # reference; Sinkhorn is ~0.015us/iter so the cost is negligible). The
        # Gram + RMSNorm are skipped when there is no fused norm_weight.
        from flashinfer.experimental.sm12x.norm.mhc._kernels import (
            run_mhc_finalize_gram,
            _selected_post_pre_decode_split_n,
            run_mhc_post_pre_functional,
            run_mhc_post_pre_partial,
            run_mhc_post_pre_prefill_block_m_partial,
            run_mhc_post_pre_prefill_gram,
            run_mhc_post_pre_prefill_partial,
            mhc_prefill_tf32_project_splits,
            run_mhc_prefill_bf16_project,
            run_mhc_prefill_tf32_project,
        )

        if not _caller_owned_buffers and expected_m is None:
            # No caller buffers (e.g. torch.compile): one functional op runs the
            # whole post_pre (partial + finalize) and returns fresh tensors, so
            # the compile graph carries ZERO auto_functionalized mHC nodes.
            return run_mhc_post_pre_functional(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                fn=fn,
                scale=hc_scale,
                bias=hc_base,
                rms_eps=float(rms_eps),
                hc_eps=float(hc_eps),
                sinkhorn_iters=sinkhorn_iters,
                norm_weight=norm_weight,
                norm_eps=float(norm_eps),
            )

        prefill_min_tokens = int(
            os.environ.get("FLASHINFER_EXP_SM12X_MHC_PREFILL_MIN_TOKENS", "96")
        )
        use_prefill_tf32_mma = _use_mhc_prefill_tf32_project(
            norm_weight=norm_weight,
            policy_m=policy_m,
        )
        use_prefill_bf16_mma = (
            _use_mhc_prefill_bf16_project(
                norm_weight=norm_weight,
                policy_m=policy_m,
                fn_bf16=fn_bf16,
            )
            and not use_prefill_tf32_mma
        )
        use_prefill_block_m = (
            not use_prefill_tf32_mma
            and not use_prefill_bf16_mma
            and norm_weight is not None
            and policy_m >= prefill_min_tokens
            and os.environ.get("FLASHINFER_EXP_SM12X_MHC_PREFILL_BLOCK_M", "1") != "0"
        )
        prefill_block_m_size = int(
            os.environ.get("FLASHINFER_EXP_SM12X_MHC_PREFILL_BLOCK_M_SIZE", "2")
        )
        prefill_tile_n_default = 12 if hidden_size == 7168 else 24
        prefill_tile_n = int(
            os.environ.get(
                "FLASHINFER_EXP_SM12X_MHC_PREFILL_TILE_N", str(prefill_tile_n_default)
            )
        )
        use_prefill_compact = (
            not use_prefill_tf32_mma
            and not use_prefill_bf16_mma
            and not use_prefill_block_m
            and norm_weight is not None
            and policy_m >= prefill_min_tokens
            and os.environ.get("FLASHINFER_EXP_SM12X_MHC_PREFILL_COMPACT", "1") != "0"
        )
        decode_source_splits = 0
        if not (
            use_prefill_tf32_mma
            or use_prefill_bf16_mma
            or use_prefill_block_m
            or use_prefill_compact
        ):
            decode_source_splits, _ = _selected_post_pre_decode_split_n(
                num_tokens=tokens,
                hidden_size=hidden_size,
            )
        if use_prefill_tf32_mma:
            run_mhc_post_pre_prefill_gram(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                partials=partials,
                out=residual_out,
            )
            run_mhc_prefill_tf32_project(
                out=residual_out,
                fn=fn,
                partials=partials,
            )
        elif use_prefill_bf16_mma:
            run_mhc_post_pre_prefill_gram(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                partials=partials,
                out=residual_out,
            )
            run_mhc_prefill_bf16_project(
                out=residual_out,
                fn_bf16=fn_bf16,
                partials=partials,
            )
        elif use_prefill_block_m:
            run_mhc_post_pre_prefill_block_m_partial(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                fn=fn,
                partials=partials,
                out=residual_out,
                compute_gram=True,
                block_m=prefill_block_m_size,
                tile_n=prefill_tile_n,
            )
        elif use_prefill_compact:
            run_mhc_post_pre_prefill_partial(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                fn=fn,
                partials=partials,
                out=residual_out,
                compute_gram=True,
            )
        else:
            run_mhc_post_pre_partial(
                x=x,
                residual=residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
                fn=fn,
                partials=partials,
                out=residual_out,
                compute_gram=norm_weight is not None,
            )
        run_mhc_finalize_gram(
            residual=residual_out,
            partials=partials,
            scale=hc_scale,
            bias=hc_base,
            y=y_out,
            post=post_out,
            comb=comb_out,
            rms_eps=float(rms_eps),
            hc_eps=float(hc_eps),
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=float(norm_eps),
            compact_partials=(
                use_prefill_tf32_mma
                or use_prefill_bf16_mma
                or use_prefill_block_m
                or use_prefill_compact
            ),
            compact_projection_splits=(
                mhc_prefill_tf32_project_splits(
                    tokens=tokens,
                    hidden_size=hidden_size,
                )
                if use_prefill_tf32_mma
                else 1
            ),
            active_source_splits=decode_source_splits,
        )
        return residual_out, post_out, comb_out, y_out

    raise ValueError(
        "sm12x_mhc_post_pre is served only by the fused Gram kernel, which "
        "supports the decode config "
        f"(hidden_size divisible by {MHC_GRAM_BLOCK_H}, "
        f"split_k=hc_mult*hidden_size/{MHC_DEFAULT_BLOCK_K}, "
        f"block_k={MHC_DEFAULT_BLOCK_K}, block_h={MHC_DEFAULT_BLOCK_H}, "
        "sinkhorn_iters=20); got "
        f"hidden_size={hidden_size}, split_k={split_k}, block_k={block_k}, "
        f"block_h={block_h}, sinkhorn_iters={sinkhorn_iters}"
    )


def _sm12x_mhc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
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
        # No caller buffer (e.g. torch.compile): the functional op allocates and
        # returns, so the compile graph carries zero auto_functionalized mHC
        # nodes. No is_compiling -- purely caller-intent.
        if not _supports_mhc_post_hidden(hidden_size):
            raise ValueError(
                "sm12x_mhc_post is served only by the post-only mHC kernel, which "
                f"supports hidden_size in {MHC_SUPPORTED_HIDDEN_SIZES}; "
                f"got hidden_size={hidden_size}"
            )
        from flashinfer.experimental.sm12x.norm.mhc._kernels import (
            run_mhc_post_functional,
        )

        return run_mhc_post_functional(
            x=x,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
        )

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
        from flashinfer.experimental.sm12x.norm.mhc._kernels import run_mhc_post

        run_mhc_post(
            x=x,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            out=out,
        )
        return out

    raise ValueError(
        "sm12x_mhc_post is served only by the post-only mHC kernel, which "
        f"supports hidden_size in {MHC_SUPPORTED_HIDDEN_SIZES}; "
        f"got hidden_size={hidden_size}"
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_pre_planned_functional",
    mutates_args=(),
)
def _mhc_pre_planned_functional_op(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
    split_k: int,
    block_k: int,
    block_h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _sm12x_mhc_pre_impl(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight if fuse_norm else None,
        norm_eps=float(norm_eps),
        split_k=int(split_k),
        block_k=int(block_k),
        block_h=int(block_h),
    )


@_mhc_pre_planned_functional_op.register_fake
def _mhc_pre_planned_functional_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
    split_k: int,
    block_k: int,
    block_h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del fn, hc_scale, hc_base, norm_weight
    del rms_eps, hc_eps, sinkhorn_iters, norm_eps, fuse_norm
    del split_k, block_k, block_h
    tokens = residual.shape[0]
    hidden_size = residual.shape[1]
    y = torch.empty(
        (tokens, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    post = torch.empty(
        (tokens, MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    comb = torch.empty(
        (tokens, MHC_MULT, MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    residual_out = torch.empty(
        (tokens, MHC_MULT, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    return residual_out, post, comb, y


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_pre_planned_functional",
    mutates_args=(),
)
def _mhc_post_pre_planned_functional_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    fn_bf16: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    has_fn_bf16: bool,
    expected_m: int,
    has_expected_m: bool,
    fuse_norm: bool,
    split_k: int,
    block_k: int,
    block_h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _sm12x_mhc_post_pre_impl(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        hc_scale,
        hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        fn_bf16=fn_bf16 if has_fn_bf16 else None,
        expected_m=int(expected_m) if has_expected_m else None,
        norm_weight=norm_weight if fuse_norm else None,
        norm_eps=float(norm_eps),
        split_k=int(split_k),
        block_k=int(block_k),
        block_h=int(block_h),
    )


@_mhc_post_pre_planned_functional_op.register_fake
def _mhc_post_pre_planned_functional_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    fn_bf16: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    has_fn_bf16: bool,
    expected_m: int,
    has_expected_m: bool,
    fuse_norm: bool,
    split_k: int,
    block_k: int,
    block_h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del x, prev_post, prev_comb, fn, hc_scale, hc_base, fn_bf16, norm_weight
    del rms_eps, hc_eps, sinkhorn_iters, norm_eps, has_fn_bf16
    del expected_m, has_expected_m, fuse_norm, split_k, block_k, block_h
    tokens = residual.shape[0]
    hidden_size = residual.shape[2]
    residual_out = torch.empty_like(residual)
    y = torch.empty(
        (tokens, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    post = torch.empty(
        (tokens, MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    comb = torch.empty(
        (tokens, MHC_MULT, MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    return residual_out, post, comb, y


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_planned_functional",
    mutates_args=(),
)
def _mhc_post_planned_functional_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
) -> torch.Tensor:
    return _sm12x_mhc_post_impl(
        x,
        residual,
        prev_post,
        prev_comb,
    )


@_mhc_post_planned_functional_op.register_fake
def _mhc_post_planned_functional_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
) -> torch.Tensor:
    del x, prev_post, prev_comb
    return torch.empty_like(residual)


def sm12x_mhc_pre(
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
    binding: SM12XMHCBinding | None = None,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    block_k: int = MHC_DEFAULT_BLOCK_K,
    block_h: int = MHC_DEFAULT_BLOCK_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compiling = torch.compiler.is_compiling()
    no_caller_owned_buffers = (
        binding is None
        and residual_out is None
        and y_out is None
        and post_out is None
        and comb_out is None
    )
    if compiling and no_caller_owned_buffers:
        norm_weight_for_kernel = norm_weight if norm_weight is not None else residual
        return torch.ops.flashinfer_sm12x.mhc_pre_planned_functional(
            residual,
            fn,
            hc_scale,
            hc_base,
            norm_weight_for_kernel,
            float(rms_eps),
            float(hc_eps),
            int(sinkhorn_iters),
            float(norm_eps),
            norm_weight is not None,
            int(split_k),
            int(block_k),
            int(block_h),
        )
    if compiling:
        raise RuntimeError(
            "sm12x_mhc_pre must be opaque to torch.compile; caller-owned mHC "
            "buffers are not supported inside Dynamo."
        )
    return _sm12x_mhc_pre_impl(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        residual_out=residual_out,
        y_out=y_out,
        post_out=post_out,
        comb_out=comb_out,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        binding=binding,
        split_k=split_k,
        block_k=block_k,
        block_h=block_h,
    )


def sm12x_mhc_post_pre(
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
    expected_m: int | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    binding: SM12XMHCBinding | None = None,
    split_k: int = MHC_DEFAULT_SPLIT_K,
    block_k: int = MHC_DEFAULT_BLOCK_K,
    block_h: int = MHC_DEFAULT_BLOCK_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compiling = torch.compiler.is_compiling()
    no_caller_owned_buffers = (
        binding is None
        and residual_out is None
        and y_out is None
        and post_out is None
        and comb_out is None
    )
    if compiling and no_caller_owned_buffers:
        fn_bf16_for_kernel = fn_bf16 if fn_bf16 is not None else fn
        norm_weight_for_kernel = norm_weight if norm_weight is not None else residual
        return torch.ops.flashinfer_sm12x.mhc_post_pre_planned_functional(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            hc_scale,
            hc_base,
            fn_bf16_for_kernel,
            norm_weight_for_kernel,
            float(rms_eps),
            float(hc_eps),
            int(sinkhorn_iters),
            float(norm_eps),
            fn_bf16 is not None,
            0 if expected_m is None else int(expected_m),
            expected_m is not None,
            norm_weight is not None,
            int(split_k),
            int(block_k),
            int(block_h),
        )
    if compiling:
        raise RuntimeError(
            "sm12x_mhc_post_pre must be opaque to torch.compile; caller-owned "
            "mHC buffers are not supported inside Dynamo."
        )
    return _sm12x_mhc_post_pre_impl(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        residual_out=residual_out,
        y_out=y_out,
        post_out=post_out,
        comb_out=comb_out,
        fn_bf16=fn_bf16,
        expected_m=expected_m,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        binding=binding,
        split_k=split_k,
        block_k=block_k,
        block_h=block_h,
    )


def sm12x_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    compiling = torch.compiler.is_compiling()
    if compiling and out is None:
        return torch.ops.flashinfer_sm12x.mhc_post_planned_functional(
            x,
            residual,
            prev_post,
            prev_comb,
        )
    if compiling:
        raise RuntimeError(
            "sm12x_mhc_post must be opaque to torch.compile; caller-owned mHC "
            "outputs are not supported inside Dynamo."
        )
    return _sm12x_mhc_post_impl(
        x,
        residual,
        prev_post,
        prev_comb,
        out=out,
    )


__all__ = [
    "SM12XMHCBinding",
    "SM12XMHCScratchCaps",
    "SM12XMHCScratchPlan",
    "MHC_DEFAULT_BLOCK_H",
    "MHC_DEFAULT_BLOCK_K",
    "MHC_DEFAULT_SPLIT_K",
    "MHC_GRAM_BLOCK_H",
    "MHC_MULT",
    "MHC_MIXES",
    "MHC_PARTIALS",
    "MHC_SOURCE_TILE_H",
    "MHC_SUPPORTED_HIDDEN_SIZES",
    "sm12x_mhc_post",
    "sm12x_mhc_pre",
    "sm12x_mhc_post_pre",
    "plan_mhc_scratch",
]
