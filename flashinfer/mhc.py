"""
Copyright (c) 2026 by FlashInfer team.

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

import functools
import math

import torch

from .api_logging import flashinfer_api
from .jit.mhc import gen_mhc_module
from .trace.templates.mhc import (
    mhc_post_trace,
    mhc_pre_big_fuse_trace,
    mhc_pre_big_fuse_with_prenorm_trace,
)
from .utils import register_custom_op, register_fake_op


_MHC_HC = 4
_MHC_MIX = _MHC_HC * (2 + _MHC_HC)


@functools.cache
def get_mhc_module():
    return gen_mhc_module().build_and_load()


def _check_mhc_post_inputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> tuple[int, int, tuple[int, ...]]:
    if residual.ndim < 3:
        raise ValueError("residual must have shape [..., 4, H]")

    outer_shape = tuple(residual.shape[:-2])
    hc = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc != _MHC_HC:
        raise ValueError(f"residual.shape[-2] / HC must be 4, got {hc}")
    if tuple(x.shape) != outer_shape + (hidden_size,):
        raise ValueError(
            f"x shape must be {outer_shape + (hidden_size,)}, got {tuple(x.shape)}"
        )
    if tuple(post_layer_mix.shape) not in (
        outer_shape + (hc,),
        outer_shape + (hc, 1),
    ):
        raise ValueError(
            "post_layer_mix shape must be [..., 4] or [..., 4, 1], "
            f"got {tuple(post_layer_mix.shape)}"
        )
    if tuple(comb_res_mix.shape) != outer_shape + (hc, hc):
        raise ValueError(
            f"comb_res_mix shape must be {outer_shape + (hc, hc)}, "
            f"got {tuple(comb_res_mix.shape)}"
        )
    return hc, hidden_size, outer_shape


@flashinfer_api(trace=mhc_post_trace)
def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    r"""Apply mHC post mapping for HC=4.

    ``out[..., new_hc, h] = x[..., h] * post_layer_mix[..., new_hc]
    + sum_old residual[..., old_hc, h] * comb_res_mix[..., old_hc, new_hc]``
    """

    hc, hidden_size, _ = _check_mhc_post_inputs(
        x, residual, post_layer_mix, comb_res_mix
    )
    x_flat = x.reshape(-1, hidden_size).contiguous()
    residual_flat = residual.reshape(-1, hc, hidden_size).contiguous()
    post_layer_mix_flat = post_layer_mix.reshape(-1, hc).contiguous()
    comb_res_mix_flat = comb_res_mix.reshape(-1, hc, hc).contiguous()
    out = torch.empty_like(residual_flat)
    _mhc_post_impl(
        out,
        x_flat,
        residual_flat,
        post_layer_mix_flat,
        comb_res_mix_flat,
    )
    return out.reshape_as(residual)


@register_custom_op("flashinfer::mhc_post", mutates_args=("out",))
def _mhc_post_impl(
    out: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> None:
    get_mhc_module().mhc_post(out, x, residual, post_layer_mix, comb_res_mix)


@register_fake_op("flashinfer::mhc_post")
def _mhc_post_impl_fake(
    out: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> None:
    pass


def _check_mhc_pre_common_inputs(
    residual: torch.Tensor,
) -> tuple[int, int, tuple[int, ...]]:
    if residual.ndim < 3:
        raise ValueError("residual must have shape [..., 4, H]")

    outer_shape = tuple(residual.shape[:-2])
    total_tokens = math.prod(outer_shape)
    hc = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc != _MHC_HC:
        raise ValueError(f"residual.shape[-2] / HC must be 4, got {hc}")
    return total_tokens, hidden_size, outer_shape


def _check_positive_eps(name: str, value: float) -> None:
    if not value > 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value}")


@flashinfer_api(trace=mhc_pre_big_fuse_trace)
def mhc_pre_big_fuse(
    dot_mix: torch.Tensor,
    sqrsum: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    k: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    num_splits: int = 1,
    block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Apply mHC pre-map big-fuse using external projection and sqrsum.

    ``dot_mix`` contains raw projection logits laid out as
    ``[pre(4), post(4), comb(16)]``. ``sqrsum`` contains the corresponding
    residual-square sums used for RMS normalization. When ``num_splits > 1``,
    both inputs have a leading split dimension that is reduced inside the CUDA
    kernel.
    """

    if num_splits not in (1, 2, 4, 8, 16):
        raise ValueError("num_splits must be one of {1, 2, 4, 8, 16}")
    _check_positive_eps("rms_eps", rms_eps)
    _check_positive_eps("mhc_pre_eps", mhc_pre_eps)
    _check_positive_eps("mhc_sinkhorn_eps", mhc_sinkhorn_eps)

    total_tokens, hidden_size, outer_shape = _check_mhc_pre_common_inputs(residual)

    if num_splits == 1:
        expected_dot_shape = outer_shape + (_MHC_MIX,)
        expected_sq_shape = outer_shape
        if tuple(dot_mix.shape) != expected_dot_shape:
            raise ValueError(
                f"dot_mix shape must be {expected_dot_shape}, got {tuple(dot_mix.shape)}"
            )
        if tuple(sqrsum.shape) != expected_sq_shape:
            raise ValueError(
                f"sqrsum shape must be {expected_sq_shape}, got {tuple(sqrsum.shape)}"
            )
        dot_mix_flat = dot_mix.reshape(total_tokens, _MHC_MIX).contiguous()
        sqrsum_flat = sqrsum.reshape(total_tokens).contiguous()
    else:
        expected_dot_shape = (num_splits,) + outer_shape + (_MHC_MIX,)
        expected_sq_shape = (num_splits,) + outer_shape
        if tuple(dot_mix.shape) != expected_dot_shape:
            raise ValueError(
                f"dot_mix shape must be {expected_dot_shape}, got {tuple(dot_mix.shape)}"
            )
        if tuple(sqrsum.shape) != expected_sq_shape:
            raise ValueError(
                f"sqrsum shape must be {expected_sq_shape}, got {tuple(sqrsum.shape)}"
            )
        dot_mix_flat = dot_mix.reshape(num_splits, total_tokens, _MHC_MIX).contiguous()
        sqrsum_flat = sqrsum.reshape(num_splits, total_tokens).contiguous()

    residual_flat = residual.reshape(total_tokens, _MHC_HC, hidden_size).contiguous()
    mhc_scale = mhc_scale.contiguous()
    mhc_base = mhc_base.contiguous()
    post_mix = torch.empty(
        (total_tokens, _MHC_HC), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (total_tokens, _MHC_HC, _MHC_HC), dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        (total_tokens, hidden_size), dtype=torch.bfloat16, device=residual.device
    )

    _mhc_pre_big_fuse_impl(
        post_mix,
        comb_mix,
        layer_input,
        dot_mix_flat,
        sqrsum_flat,
        residual_flat,
        mhc_scale,
        mhc_base,
        k,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        num_splits,
        block_size,
    )

    return (
        post_mix.reshape(outer_shape + (_MHC_HC,)).unsqueeze(-1),
        comb_mix.reshape(outer_shape + (_MHC_HC, _MHC_HC)),
        layer_input.reshape(outer_shape + (hidden_size,)),
    )


@flashinfer_api(trace=mhc_pre_big_fuse_with_prenorm_trace)
def mhc_pre_big_fuse_with_prenorm(
    dot_mix: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Apply mHC pre-map big-fuse and compute RMS sqrsum from ``residual``.

    This matches the Agentic ``mhc_pre_finalize`` boundary when no precomputed
    ``sqrsum`` is supplied. ``dot_mix`` may be shaped ``[..., 24]`` or
    ``[1, ..., 24]``.
    """

    _check_positive_eps("rms_eps", rms_eps)
    _check_positive_eps("mhc_pre_eps", mhc_pre_eps)
    _check_positive_eps("mhc_sinkhorn_eps", mhc_sinkhorn_eps)

    total_tokens, hidden_size, outer_shape = _check_mhc_pre_common_inputs(residual)

    expected_dot_shape = outer_shape + (_MHC_MIX,)
    split_dot_shape = (1,) + expected_dot_shape
    dot_mix_shape = tuple(dot_mix.shape)
    if dot_mix_shape in (expected_dot_shape, split_dot_shape):
        dot_mix_2d = dot_mix.reshape(total_tokens, _MHC_MIX).contiguous()
    else:
        raise ValueError(
            f"dot_mix shape must be {expected_dot_shape} or {split_dot_shape}, "
            f"got {dot_mix_shape}"
        )

    residual_flat = residual.reshape(total_tokens, _MHC_HC, hidden_size).contiguous()
    mhc_scale = mhc_scale.contiguous()
    mhc_base = mhc_base.contiguous()
    post_mix = torch.empty(
        (total_tokens, _MHC_HC), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (total_tokens, _MHC_HC, _MHC_HC), dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        (total_tokens, hidden_size), dtype=torch.bfloat16, device=residual.device
    )

    _mhc_pre_big_fuse_with_prenorm_impl(
        post_mix,
        comb_mix,
        layer_input,
        dot_mix_2d,
        residual_flat,
        mhc_scale,
        mhc_base,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        block_size,
    )

    return (
        post_mix.reshape(outer_shape + (_MHC_HC,)).unsqueeze(-1),
        comb_mix.reshape(outer_shape + (_MHC_HC, _MHC_HC)),
        layer_input.reshape(outer_shape + (hidden_size,)),
    )


@register_custom_op(
    "flashinfer::mhc_pre_big_fuse",
    mutates_args=("post_mix", "comb_mix", "layer_input"),
)
def _mhc_pre_big_fuse_impl(
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    dot_mix: torch.Tensor,
    sqrsum: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    k: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    num_splits: int,
    block_size: int,
) -> None:
    get_mhc_module().mhc_pre_big_fuse(
        post_mix,
        comb_mix,
        layer_input,
        dot_mix,
        sqrsum,
        residual,
        mhc_scale,
        mhc_base,
        k,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        num_splits,
        block_size,
    )


@register_fake_op("flashinfer::mhc_pre_big_fuse")
def _mhc_pre_big_fuse_impl_fake(
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    dot_mix: torch.Tensor,
    sqrsum: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    k: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    num_splits: int,
    block_size: int,
) -> None:
    pass


@register_custom_op(
    "flashinfer::mhc_pre_big_fuse_with_prenorm",
    mutates_args=("post_mix", "comb_mix", "layer_input"),
)
def _mhc_pre_big_fuse_with_prenorm_impl(
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    dot_mix: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    block_size: int,
) -> None:
    get_mhc_module().mhc_pre_big_fuse_with_prenorm(
        post_mix,
        comb_mix,
        layer_input,
        dot_mix,
        residual,
        mhc_scale,
        mhc_base,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        block_size,
    )


@register_fake_op("flashinfer::mhc_pre_big_fuse_with_prenorm")
def _mhc_pre_big_fuse_with_prenorm_impl_fake(
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    dot_mix: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    block_size: int,
) -> None:
    pass
