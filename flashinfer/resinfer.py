# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Experimental ResInfer quantization and folded GEMM APIs.

Ported from https://github.com/ShawRong/resinfer at
51b906836a69fa43915040314df57bc9f23030ad. See the backend README for lineage.

Import public entry points from this module. Each call explicitly opts into
an experimental API and lazily imports its backend. Implementation modules
under ``flashinfer.experimental.resinfer_*`` are internal. These APIs remain
experimental and may change without compatibility guarantees.
"""

import torch

from .api_logging import flashinfer_experimental_api
from .utils import get_compute_capability

__all__ = [
    "nvfp4_quantize_residue",
    "mm_nvfp4_residue",
]


def _check_tensor(tensor, name, device, dtype, alignment=16):
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tensor.dtype not in dtype:
        raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")
    if not tensor.is_contiguous() or tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


def _check_sm100(tensor):
    if not tensor.is_cuda:
        raise ValueError("ResInfer requires CUDA tensors")
    if get_compute_capability(tensor.device) != (10, 0):
        raise ValueError("ResInfer currently supports SM100 only")


@flashinfer_experimental_api
def nvfp4_quantize_residue(input: torch.Tensor, global_scale: torch.Tensor):
    """Quantize each activation row and its full-channel residue on SM100.

    ``input`` is contiguous FP16/BF16 ``[M, K]``, with positive M and K,
    K divisible by 16. ``global_scale`` contains one or two positive finite
    FP32 quantization multipliers on the same device: base, optionally residue.
    Scale values and finite input values are caller preconditions (not checked
    with a synchronizing device-to-host copy).

    Returns uint8 packed E2M1 ``[2*M, K/2]`` in adjacent base/residue row order
    and a flat uint8 E4M3 scale buffer in the 128-row, 4-scale swizzle layout.
    Padding scale entries are zero. The residue is computed from the quantized
    and reconstructed base, following ResInfer's production pack16 algorithm.
    """
    _check_sm100(input)
    _check_tensor(input, "input", input.device, (torch.float16, torch.bfloat16), 32)
    _check_tensor(global_scale, "global_scale", input.device, (torch.float32,), 4)
    if input.ndim != 2 or min(input.shape) <= 0 or input.shape[1] % 16:
        raise ValueError("input must be [M, K], M > 0, K > 0 and K divisible by 16")
    if global_scale.numel() not in (1, 2):
        raise ValueError("global_scale must contain one or two elements")
    from .experimental.resinfer_mext.quantization import quantize

    with torch.cuda.device(input.device):
        return quantize(input, global_scale)


@flashinfer_experimental_api
def mm_nvfp4_residue(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Multiply ResInfer row-pair activations by shared weights and fold in FP32.

    ``act`` is packed uint8 ``[2*M, K/2]`` from ``nvfp4_quantize_residue``;
    ``weight`` is contiguous packed uint8 ``[N, K/2]``. Both scale buffers use
    the 128-row, 4-scale swizzle (E4M3 bytes). K is divisible by 64, N by 8.
    ``alpha`` is one or two FP32 reconstruction multipliers: reciprocals of
    the product of activation and weight quantization global multipliers.
    Two values apply independently to the base and residue accumulators.
    The output is ``[M, N]`` in FP16 or BF16. Weight quantization is external.
    Warm up outside CUDA graph capture to compile the selected kernel.
    """
    _check_sm100(act)
    for tensor, name, alignment in (
        (act, "act", 32),
        (weight, "weight", 32),
        (act_scale, "act_scale", 16),
        (weight_scale, "weight_scale", 16),
    ):
        _check_tensor(tensor, name, act.device, (torch.uint8,), alignment)
    _check_tensor(alpha, "alpha", act.device, (torch.float32,), 4)
    if act.ndim != 2 or weight.ndim != 2:
        raise ValueError("act and weight must be matrices")
    m2, kp = act.shape
    n, wk = weight.shape
    if min(m2, kp, n) <= 0 or m2 % 2 or kp != wk or kp % 32 or n % 8:
        raise ValueError(
            "require positive dimensions, paired rows, matching K divisible by 64, N divisible by 8"
        )
    for scale, rows, name in (
        (act_scale, m2, "act_scale"),
        (weight_scale, n, "weight_scale"),
    ):
        expected = ((rows + 127) // 128) * ((2 * kp + 63) // 64) * 512
        if scale.numel() != expected:
            raise ValueError(f"{name} must contain {expected} bytes")
    if alpha.numel() not in (1, 2) or out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("alpha must have one/two elements; output must be FP16/BF16")
    from .experimental.resinfer_mext.host import mext_fold_gemm

    with torch.cuda.device(act.device):
        return mext_fold_gemm(
            weight, act, weight_scale, act_scale, alpha, out_dtype=out_dtype
        )
