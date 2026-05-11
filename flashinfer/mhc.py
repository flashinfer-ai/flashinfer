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

import torch

from .jit.mhc import gen_mhc_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_mhc_module():
    return gen_mhc_module().build_and_load()


def _check_mhc_post_inputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> tuple[int, int, tuple[int, ...]]:
    if (
        not x.is_cuda
        or not residual.is_cuda
        or not post_layer_mix.is_cuda
        or not comb_res_mix.is_cuda
    ):
        raise ValueError("all mhc_post inputs must be CUDA tensors")
    if (
        x.device != residual.device
        or x.device != post_layer_mix.device
        or x.device != comb_res_mix.device
    ):
        raise ValueError("all mhc_post inputs must be on the same device")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be torch.bfloat16, got {x.dtype}")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be torch.bfloat16, got {residual.dtype}")
    if post_layer_mix.dtype != torch.float32:
        raise ValueError(
            f"post_layer_mix must be torch.float32, got {post_layer_mix.dtype}"
        )
    if comb_res_mix.dtype != torch.float32:
        raise ValueError(
            f"comb_res_mix must be torch.float32, got {comb_res_mix.dtype}"
        )
    if residual.ndim < 3:
        raise ValueError("residual must have shape [..., 4, H]")

    outer_shape = tuple(residual.shape[:-2])
    hc = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if hc != 4:
        raise ValueError(f"residual.shape[-2] / HC must be 4, got {hc}")
    if hidden_size <= 0:
        raise ValueError("hidden size must be positive")
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
