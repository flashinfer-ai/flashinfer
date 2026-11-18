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

The per-token NVFP4 GEMM of this backend as the last autotuning candidate of
``mm_fp4(backend="auto")``: the check of the operands it serves and its
``fp4_gemm`` runner. Core reaches this module through a deferred import only,
on the automatic path of ``mm_fp4`` while the candidate is listed; the kernels
are reached through a deferred import of :mod:`.cake_backend`.
"""

from __future__ import annotations

import functools
import math
from typing import Any, List, Tuple

import torch

from ...autotuner import OptimizationProfile, TunableRunner
from ...utils import get_compute_capability


@functools.cache
def _cake_backend():
    """The ``cake_backend`` module, imported once on first use rather than by a
    function-level import on every call."""
    from . import cake_backend

    return cake_backend


@functools.lru_cache(maxsize=4096)
def _cake_fp4_gemm_layout_serves(
    out_dtype: torch.dtype,
    a_meta: Tuple[Any, ...],
    b_meta: Tuple[Any, ...],
    a_descale_meta: Tuple[Any, ...],
    b_descale_meta: Tuple[Any, ...],
    alpha_meta: Tuple[Any, ...],
    out_meta: Tuple[Any, ...],
) -> bool:
    """The checks of :func:`cake_fp4_gemm_serves` that read no data pointer,
    on the ``(shape, stride, dtype, device)`` of each tensor operand; memoized,
    since a model calls few distinct layouts."""
    if out_dtype not in (torch.bfloat16, torch.float16):
        return False
    a_shape, a_stride, a_dtype, device = a_meta
    b_shape, b_stride, b_dtype, b_device = b_meta
    sfa_shape, sfa_stride, sfa_dtype, sfa_device = a_descale_meta
    sfb_shape, sfb_stride, sfb_dtype, sfb_device = b_descale_meta
    alpha_shape, alpha_dtype, alpha_device = alpha_meta
    out_shape, out_stride, out_dtype_actual, out_device = out_meta
    if len(a_shape) != 2 or len(b_shape) != 2:
        return False
    m, kh = (int(v) for v in a_shape)
    n = int(b_shape[1])
    # Every generated GEMM program stores the output through a TMA box at least
    # 32 columns wide, and its binding refuses a box wider than the output
    # (cake_backend._tma_boxes_fit): N in {8, 16, 24} raises at the launch.
    if m <= 0 or n < 32 or n % 8 != 0 or (2 * kh) % 256 != 0:
        return False
    if (
        a_dtype.itemsize != 1
        or b_dtype.itemsize != 1
        or not _is_contiguous_layout(a_shape, a_stride)
    ):
        return False
    if b_shape[0] != kh or tuple(b_stride) != (1, kh):
        return False
    if alpha_dtype != torch.float32 or tuple(alpha_shape) != (m,):
        return False
    if (
        out_dtype_actual != out_dtype
        or tuple(out_shape) != (m, n)
        or not _is_contiguous_layout(out_shape, out_stride)
    ):
        return False
    if len(sfb_shape) != 2:
        return False
    if device.type != "cuda" or any(
        d != device
        for d in (b_device, sfa_device, sfb_device, alpha_device, out_device)
    ):
        return False
    if get_compute_capability(device) not in ((10, 0), (10, 3)):
        return False
    cake_backend = _cake_backend()
    cols = cake_backend.padded_sf_cols(2 * kh)
    # b_descale is read through its transpose: (rows, cols) = b_descale.T.
    for dtype, numel, contiguous, rows in (
        (
            sfa_dtype,
            math.prod(sfa_shape),
            _is_contiguous_layout(sfa_shape, sfa_stride),
            m,
        ),
        (
            sfb_dtype,
            math.prod(sfb_shape),
            _is_contiguous_layout(sfb_shape[::-1], sfb_stride[::-1]),
            n,
        ),
    ):
        if dtype not in (torch.uint8, torch.float8_e4m3fn) or not contiguous:
            return False
        if numel != cake_backend.padded_rows(rows) * cols:
            return False
    return cake_backend.generated_program_available(device)


def _is_contiguous_layout(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> bool:
    """``torch.Tensor.is_contiguous`` from a shape and stride: dimensions of
    size 1 may have any stride, and an empty tensor is contiguous."""
    if 0 in shape:
        return True
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride), strict=True):
        if size != 1 and step != expected:
            return False
        expected *= size
    return True


def cake_fp4_gemm_serves(
    inputs: List[Any], use_8x4_sf_layout: bool, enable_pdl: bool
) -> bool:
    """Whether ``mm_fp4(backend="cake")`` runs the ``fp4_gemm`` operands
    ``inputs`` without raising.

    The checks of ``cake_mm_fp4_requirement`` plus those ``mm_fp4_per_token``
    and the generated bindings add (scale tensor sizes and layouts, ``N >= 32``,
    16-byte aligned operands, a registered program for the device), as a bool.
    None of them depends on M beyond M >= 1, so a call is served iff every M
    of its tuning bucket is. The automatic cake candidate of ``mm_fp4`` is
    offered only where this holds. The layout checks are memoized by
    :func:`_cake_fp4_gemm_layout_serves`; the data pointers are read per call.
    """
    a, b, a_descale, b_descale, alpha, out_dtype, out, block_size, use_nvfp4, _ = inputs
    if use_8x4_sf_layout or not enable_pdl or not use_nvfp4 or block_size != 16:
        return False
    if not isinstance(alpha, torch.Tensor):
        return False
    if not _cake_fp4_gemm_layout_serves(
        out_dtype,
        (a.shape, a.stride(), a.dtype, a.device),
        (b.shape, b.stride(), b.dtype, b.device),
        (a_descale.shape, a_descale.stride(), a_descale.dtype, a_descale.device),
        (b_descale.shape, b_descale.stride(), b_descale.dtype, b_descale.device),
        (alpha.shape, alpha.dtype, alpha.device),
        (out.shape, out.stride(), out.dtype, out.device),
    ):
        return False
    # The bindings build TMA descriptors on these addresses and raise unless
    # each is 16-byte aligned.
    return (
        a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
        and a_descale.data_ptr() % 16 == 0
        and b_descale.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
    )


@functools.cache
def cake_gemm_fp4_runner(use_8x4_sf_layout: bool, enable_pdl: bool):
    """The cake per-token NVFP4 GEMM as an ``fp4_gemm`` candidate of
    ``mm_fp4(backend="auto")``: one configuration (the default tactic of
    ``mm_fp4_per_token``, chosen by shape), valid only for the operands
    :func:`cake_fp4_gemm_serves` accepts."""

    class CakeFp4GemmRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            # The programs differ by output dtype; tune each dtype apart.
            return (str(inputs[5]),)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            if cake_fp4_gemm_serves(inputs, use_8x4_sf_layout, enable_pdl):
                return [-1]
            return []

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, a_descale, b_descale, alpha, _, out, _, _, _ = inputs
            return _cake_backend().mm_fp4_per_token(
                a, b, a_descale, b_descale, alpha, out
            )

    return CakeFp4GemmRunner()
