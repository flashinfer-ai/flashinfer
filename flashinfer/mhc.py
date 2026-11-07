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
from typing import Optional

import torch

from .api_logging import flashinfer_api, flashinfer_experimental_api
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

    Parameters
    ----------
    x : torch.Tensor
        Layer output tensor, shape ``[..., H]`` (matches ``residual``'s
        outer dimensions and ``H = residual.shape[-1]``).
    residual : torch.Tensor
        Multi-head residual tensor, shape ``[..., HC=4, H]``. The
        ``HC=4`` axis must equal 4 (mHC is currently hard-wired to 4 sub-heads).
    post_layer_mix : torch.Tensor
        Layer-output mixing weights, shape ``[..., HC=4]`` or
        ``[..., HC=4, 1]`` (both accepted). Provides the per-new-head
        coefficient applied to ``x`` (the ``post_layer_mix`` factor in the
        formula above). See mHC paper / reference implementation for the
        exact training-time derivation.
    comb_res_mix : torch.Tensor
        Residual combination matrix, shape ``[..., HC=4, HC=4]``. Each
        ``[old_hc, new_hc]`` entry is the coefficient mixing
        ``residual[..., old_hc, :]`` into the new head ``new_hc`` (the
        ``comb_res_mix`` factor above). See mHC paper / reference
        implementation for the training-time derivation.

    Returns
    -------
    torch.Tensor
        Output tensor with the same shape as ``residual``
        (``[..., HC=4, H]``).
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

    Parameters
    ----------
    dot_mix : torch.Tensor
        Raw projection logits, shape ``[..., 24]`` when ``num_splits=1`` or
        ``[num_splits, ..., 24]`` when ``num_splits > 1``. The trailing
        ``24 = 4 (pre) + 4 (post) + 16 (comb)`` slots are partitioned and
        re-shaped to per-head ``pre / post / comb`` factors inside the kernel.
        See mHC paper / reference implementation for the projection
        derivation.
    sqrsum : torch.Tensor
        Pre-computed per-token residual square-sum used for RMS
        normalization. Shape ``[...]`` matching ``dot_mix`` outer dims, or
        ``[num_splits, ...]`` when ``num_splits > 1`` (the kernel reduces
        the split axis internally). dtype is implementation-defined.
    residual : torch.Tensor
        Multi-head residual tensor, shape ``[..., HC=4, H]`` (bfloat16).
        Same layout as :func:`mhc_post`'s ``residual``.
    mhc_scale : torch.Tensor
        mHC scaling tensor consumed by the Sinkhorn / mix step. Shape and
        dtype are implementation-defined; see mHC paper / reference
        implementation for the exact semantics.
    mhc_base : torch.Tensor
        mHC bias / base tensor consumed by the Sinkhorn / mix step. Shape
        and dtype are implementation-defined; see mHC paper / reference
        implementation for the exact semantics.
    k : int
        mHC algorithm parameter ``k`` passed to the CUDA kernel. See mHC
        paper / reference implementation for the precise semantics.
    rms_eps : float
        RMSNorm epsilon for numerical stability. Default ``1e-6``. Must be
        strictly positive.
    mhc_pre_eps : float
        Numerical-stability epsilon used in the mHC pre-map step.
        Default ``1e-6``. Must be strictly positive.
    mhc_sinkhorn_eps : float
        Numerical-stability epsilon used in the Sinkhorn iteration step.
        Default ``1e-6``. Must be strictly positive.
    mhc_post_mult_value : float
        Post-mix multiplicative factor applied to the mHC post outputs.
        Default ``1.0``.
    sinkhorn_repeat : int
        Number of Sinkhorn iterations. Default ``20``.
    num_splits : int
        Split factor along the leading dimension. Must be one of
        ``{1, 2, 4, 8, 16}``. When > 1, ``dot_mix`` and ``sqrsum`` carry a
        leading split axis that is reduced inside the kernel. Default
        ``1``.
    block_size : int
        CUDA block-size hint forwarded to the kernel (``0`` selects the
        kernel's default). Default ``0``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(post_mix, comb_mix, layer_input)`` tensors with shapes
        ``[..., HC=4, 1]``, ``[..., HC=4, HC=4]`` and ``[..., H]``
        respectively. See mHC paper / reference implementation for how
        downstream layers consume them.
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

    Parameters
    ----------
    dot_mix : torch.Tensor
        Raw projection logits, shape ``[..., 24]`` or ``[1, ..., 24]``. The
        trailing ``24 = 4 (pre) + 4 (post) + 16 (comb)`` slots are
        partitioned and re-shaped to per-head factors inside the kernel.
        See mHC paper / reference implementation for the projection
        derivation.
    residual : torch.Tensor
        Multi-head residual tensor, shape ``[..., HC=4, H]`` (bfloat16).
        ``sqrsum`` is computed from this tensor inside the kernel; same
        layout as :func:`mhc_post`'s ``residual``.
    mhc_scale : torch.Tensor
        mHC scaling tensor consumed by the Sinkhorn / mix step. Shape and
        dtype are implementation-defined; see mHC paper / reference
        implementation for the exact semantics.
    mhc_base : torch.Tensor
        mHC bias / base tensor consumed by the Sinkhorn / mix step. Shape
        and dtype are implementation-defined; see mHC paper / reference
        implementation for the exact semantics.
    rms_eps : float
        RMSNorm epsilon for numerical stability. Default ``1e-6``. Must be
        strictly positive.
    mhc_pre_eps : float
        Numerical-stability epsilon used in the mHC pre-map step.
        Default ``1e-6``. Must be strictly positive.
    mhc_sinkhorn_eps : float
        Numerical-stability epsilon used in the Sinkhorn iteration step.
        Default ``1e-6``. Must be strictly positive.
    mhc_post_mult_value : float
        Post-mix multiplicative factor applied to the mHC post outputs.
        Default ``1.0``.
    sinkhorn_repeat : int
        Number of Sinkhorn iterations. Default ``20``.
    block_size : int
        CUDA block-size hint forwarded to the kernel (``0`` selects the
        kernel's default). Default ``0``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(post_mix, comb_mix, layer_input)`` tensors with shapes
        ``[..., HC=4, 1]``, ``[..., HC=4, HC=4]`` and ``[..., H]``
        respectively. See mHC paper / reference implementation for how
        downstream layers consume them.
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


# ---------------------------------------------------------------------------
# Experimental: fused mHC post -> pre (+ input RMSNorm)
# ---------------------------------------------------------------------------
#
# Thin entry points. The implementation (support checks, JIT module, kernels and the
# composable fallback) lives in flashinfer/experimental/mhc_fused_post_pre/ and is
# imported on first call only.


@flashinfer_experimental_api
def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-5,
    out: Optional[tuple] = None,
    fallback: bool = True,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    r"""Fused mHC post map followed by the next block's pre map and input RMSNorm (HC=4).

    ``residual_cur[..., j, :] = bf16(x * post_layer_mix[..., j]
    + sum_i residual[..., i, :] * comb_res_mix[..., i, j])``; the 24 mixes are the
    projection of the flattened BF16 ``residual_cur`` onto ``fn`` scaled by
    ``rsqrt(mean(residual_cur^2) + rms_eps)`` and by ``hc_scale``, plus ``hc_base``;
    ``post_mix_cur = sigmoid(post logits) * hc_post_mult_value``; ``comb_mix_cur`` is
    ``sinkhorn_repeat`` Sinkhorn iterations of ``softmax(comb logits) +
    hc_sinkhorn_eps``; ``layer_input_cur`` is the RMSNorm (``norm_weight``,
    ``norm_eps``, statistic from the pre-rounding FP32 sum) of the BF16 pre-mix
    weighted sum of the ``residual_cur`` streams with
    ``pre_mix = sigmoid(pre logits) + hc_pre_eps``.

    On SM103 with hidden size 4096, the default constants
    (``rms_eps=1e-5, hc_pre_eps=hc_sinkhorn_eps=1e-6, hc_post_mult_value=2,
    sinkhorn_repeat=20, norm_eps=1e-5``) and 33 <= tokens <= 16384, one specialized
    kernel launch computes all four outputs (TF32 projection with FP32 accumulation,
    as the unfused TF32 pre-norm GEMM path). Every other call takes the fallback.

    Calling this experimental API is the opt-in: no environment variable is involved,
    and it takes part in no automatic backend selection. See
    ``flashinfer/experimental/mhc_fused_post_pre/README.md``.

    Parameters
    ----------
    x : torch.Tensor
        Layer output, shape ``[..., H]``, bfloat16.
    residual : torch.Tensor
        Multi-stream residual, shape ``[..., 4, H]``, bfloat16.
    post_layer_mix : torch.Tensor
        Shape ``[..., 4]`` or ``[..., 4, 1]``, float32.
    comb_res_mix : torch.Tensor
        Shape ``[..., 4, 4]``, float32 (``[old_hc, new_hc]``).
    fn : torch.Tensor
        Projection weight, shape ``[24, 4 * H]``, float32.
    hc_scale : torch.Tensor
        Shape ``[3]``, float32 (pre / post / comb logit scales).
    hc_base : torch.Tensor
        Shape ``[24]``, float32 (logit biases).
    norm_weight : torch.Tensor
        RMSNorm weight, shape ``[H]``, bfloat16.
    rms_eps : float
        Epsilon of the RMS scaling of the projected mixes.
    hc_pre_eps : float
        Added to the sigmoid pre-mix weights.
    hc_sinkhorn_eps : float
        Epsilon of the softmax and of every Sinkhorn normalization.
    hc_post_mult_value : float
        Scale of the sigmoid post-mix weights.
    sinkhorn_repeat : int
        Number of Sinkhorn iterations.
    norm_eps : float
        Epsilon of the input RMSNorm.
    out : tuple, optional
        Preallocated ``(residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)``;
        views of ``[M, 4, H]``, ``[M, 4]``, ``[M, 4, 4]`` and ``[M, H]`` are accepted.
    fallback : bool
        ``True`` (default): calls outside the specialized surface run the composable
        PyTorch implementation. ``False``: such calls compute nothing and return
        ``None``, so a framework can keep its own implementation for them.

    Returns
    -------
    tuple or None
        ``(residual_cur [..., 4, H] bf16, post_mix_cur [..., 4, 1] fp32,
        comb_mix_cur [..., 4, 4] fp32, layer_input_cur [..., H] bf16)``; ``None`` only
        when ``fallback=False`` and the specialized kernel did not serve the call.

    Note
    ----
    The specialized kernel is built, loaded and prepared on the first call outside
    CUDA graph capture, never under capture; call :func:`mhc_fused_post_pre_warmup`
    before capturing graphs whose first call may be this one.
    """
    from .experimental.mhc_fused_post_pre import run

    return run(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        out=out,
        fallback=fallback,
    )


@flashinfer_experimental_api
def mhc_fused_post_pre_supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-5,
    out: Optional[tuple] = None,
) -> bool:
    r"""Return ``True`` when :func:`mhc_fused_post_pre` would run the specialized kernel
    for exactly these arguments (nothing is launched).

    Capture-safe: under CUDA graph capture it never compiles or prepares a device, so an
    unprepared device reports ``False`` (see :func:`mhc_fused_post_pre_warmup`). It never
    allocates: without ``out`` only the inputs are checked (the outputs the op would
    allocate always pass the output checks); with ``out`` those tensors are checked too.

    Parameters
    ----------
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, norm_weight
        As in :func:`mhc_fused_post_pre`.
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps
        As in :func:`mhc_fused_post_pre`.
    out : tuple, optional
        As in :func:`mhc_fused_post_pre`.

    Returns
    -------
    bool
        Whether the call would be served by the specialized kernel.
    """
    from .experimental.mhc_fused_post_pre import supported

    return supported(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        out=out,
    )


@flashinfer_experimental_api
def mhc_fused_post_pre_warmup(
    device: Optional[torch.device] = None,
    hidden_size: int = 4096,
    hc_mult: int = 4,
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-5,
) -> bool:
    r"""Build, load and prepare the specialized :func:`mhc_fused_post_pre` kernels for
    ``device`` (default: the current CUDA device) ahead of CUDA graph capture.

    :func:`mhc_fused_post_pre` never compiles or prepares a device while the current
    stream is capturing: an unprepared call made under capture is served by the
    fallback (or declined with ``fallback=False``) and that path is what the graph
    records. Callers that may issue their first call inside a capture (e.g. a decode
    graph for a token count no eager warmup ran) call this once beforehand.

    Parameters
    ----------
    device : torch.device, optional
        The CUDA device to prepare; defaults to the current device.
    hidden_size, hc_mult
        Geometry of the calls to prepare for.
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps
        Constants of the calls to prepare for, as in :func:`mhc_fused_post_pre`.

    Returns
    -------
    bool
        ``True`` when calls with this geometry and these constants can dispatch the
        specialized kernel on ``device`` -- also under capture. ``False`` without
        building anything when the geometry / constants / architecture are outside the
        specialized surface, or when called under capture.
    """
    from .experimental.mhc_fused_post_pre import warmup

    return warmup(
        device,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
    )


@flashinfer_experimental_api
def mhc_pre_big_fuse_with_norm(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    post_mix: Optional[torch.Tensor] = None,
    comb_mix: Optional[torch.Tensor] = None,
    layer_input: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""mHC pre-map big-fuse from split-K partials, with the output RMSNorm fused in.

    ``rms = rsqrt(sum_s gemm_out_sqrsum[s] / (4 H) + rms_eps)``,
    ``mixes = (sum_s gemm_out_mul[s]) * rms``; ``pre = sigmoid(mixes[0:4] * hc_scale[0]
    + hc_base[0:4]) + hc_pre_eps``, ``post_mix = sigmoid(mixes[4:8] * hc_scale[1] +
    hc_base[4:8]) * hc_post_mult_value``, ``comb_mix`` = ``sinkhorn_repeat``-iteration
    Sinkhorn of ``softmax(mixes[8:24].view(4, 4) * hc_scale[2] + hc_base[8:24])``;
    ``ol = sum_j pre[j] * residual[:, j]`` (fp32), ``layer_input = bf16(float(bf16(ol)) *
    rsqrt(mean(ol^2) + norm_eps) * norm_weight)``.

    On SM100 / SM103, ``H = 4096`` decode calls (``1 <= T <= 32``, ``NS`` in ``{8, 64}``,
    ``rms_eps = norm_eps = 1e-5``, ``hc_pre_eps = hc_sinkhorn_eps = 1e-6``,
    ``hc_post_mult_value = 2``, ``sinkhorn_repeat = 20``, dense 16-byte aligned tensors) run a
    thread-block-cluster kernel; every other call runs a composable fp32 PyTorch
    implementation of the same contract.

    Parameters
    ----------
    gemm_out_mul : torch.Tensor
        fp32 ``[NS, T, 24]`` split-K partial projection logits.
    gemm_out_sqrsum : torch.Tensor
        fp32 ``[NS, T]`` split-K partial sums of squares of the residual.
    hc_scale, hc_base : torch.Tensor
        fp32 ``[3]`` and ``[24]`` logit scales / biases.
    residual : torch.Tensor
        bf16 ``[T, 4, H]`` residual streams.
    norm_weight : torch.Tensor
        bf16 ``[H]`` RMSNorm weight applied to the collapsed stream.
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps
        Model constants (no defaults: they are part of the numerical contract).
    post_mix, comb_mix, layer_input : Optional[torch.Tensor]
        Optional destination buffers (fp32 ``[T, 4]``, fp32 ``[T, 16]`` or ``[T, 4, 4]``,
        bf16 ``[T, H]``); allocated when omitted.
    enable_pdl : Optional[bool]
        Launch the cluster kernel with programmatic dependent launch. ``None`` (default)
        launches without it, which is the faster choice for an isolated call; a caller whose
        producer kernel triggers its dependents early may pass ``True``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(post_mix, comb_mix, layer_input)``.

    Notes
    -----
    :func:`mhc_pre_big_fuse_with_norm_supported` reports whether the cluster kernel serves a
    call, and :func:`mhc_pre_big_fuse_with_norm_warmup` loads it ahead of CUDA-graph capture
    (it is never built during capture). ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` forces the
    composable implementation at call time.
    """
    if residual.dim() != 3:
        raise ValueError("residual must have shape [T, 4, H]")
    if residual.shape[1] != _MHC_HC:
        raise ValueError(f"residual.shape[1] / HC must be 4, got {residual.shape[1]}")

    from .experimental.mhc_pre_big_fuse_with_norm import run

    return run(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        post_mix=post_mix,
        comb_mix=comb_mix,
        layer_input=layer_input,
        enable_pdl=enable_pdl,
    )


@flashinfer_experimental_api
def mhc_pre_big_fuse_with_norm_supported(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    post_mix: Optional[torch.Tensor] = None,
    comb_mix: Optional[torch.Tensor] = None,
    layer_input: Optional[torch.Tensor] = None,
) -> bool:
    r"""Return True when :func:`mhc_pre_big_fuse_with_norm` runs its cluster kernel for
    exactly these arguments.

    The decision is made at call time: the ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``
    switch, the supported shapes and model constants, dtypes, strides, 16-byte alignment,
    the device architecture and -- under CUDA-graph capture -- whether the kernel module is
    already loaded. Callers that have their own fast path for the op should keep it whenever
    this returns False.
    """
    from .experimental.mhc_pre_big_fuse_with_norm import supported

    return supported(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        post_mix=post_mix,
        comb_mix=comb_mix,
        layer_input=layer_input,
    )


@flashinfer_experimental_api
def mhc_pre_big_fuse_with_norm_warmup(device: Optional[torch.device] = None) -> bool:
    r"""Load the cluster kernel of :func:`mhc_pre_big_fuse_with_norm` outside CUDA-graph
    capture.

    Returns True when the kernel is usable on ``device`` (default: the current CUDA device).
    """
    from .experimental.mhc_pre_big_fuse_with_norm import warmup

    return warmup(device)
