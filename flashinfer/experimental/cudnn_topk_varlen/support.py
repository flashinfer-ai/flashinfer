# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight eligibility checks; no frontend or kernel imports at module load."""

import torch

from ...api_logging import experimental_auto_backends_allowed
from ...utils import experimental_backend, supported_compute_capability

# Deliberately enumerated: eligibility is narrower than the explicit backend's
# contract. Changes require integrated warm/cold measurements on that device.
CUDNN_TOPK_AUTO_SHAPES = {
    103: frozenset({(512, 16384), (512, 32768), (512, 131072)}),
    107: frozenset({(512, 16384), (512, 32768), (256, 131072)}),
}


def _tensor_layout_ok(tensor, dtype, shape, device):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tensor.is_contiguous()
        and not tensor.is_neg()
        and not tensor.is_conj()
        and tensor.data_ptr() % 16 == 0
    )


@experimental_backend
@supported_compute_capability([103, 107])
def check_cudnn_top_k_varlen(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    compress_ratio=1,
    next_n=1,
    return_values=False,
    out_indices=None,
    out_values=None,
    backend="auto",
    load_balance=True,
    workspace=None,
):
    # The decorator also enforces this gate. Check it before the optional
    # dependency probe so ordinary auto calls never import cuDNN for this route.
    if backend == "auto" and not experimental_auto_backends_allowed():
        return False
    if return_values or pre_idx is not None:
        return False
    if any(type(x) is not int for x in (top_k, next_n, compress_ratio)):
        return False
    if top_k not in (512, 1024, 2048) or not 1 <= next_n <= 512:
        return False
    if not 1 <= compress_ratio <= 2**31 - 1:
        return False
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        return False
    rows, cols = logits.shape
    if not 1 <= rows <= 512 or not 1 <= cols <= 262144 or rows % next_n:
        return False
    if not _tensor_layout_ok(logits, torch.bfloat16, (rows, cols), logits.device):
        return False
    if not _tensor_layout_ok(seq_lens, torch.int32, (rows // next_n,), logits.device):
        return False
    if out_indices is not None:
        # The public API normalizes flat outputs to [rows, K] before dispatch.
        if not isinstance(out_indices, torch.Tensor):
            return False
        shape = tuple(out_indices.shape)
        if shape not in ((rows, top_k), (rows * top_k,)):
            return False
        if not _tensor_layout_ok(out_indices, torch.int32, shape, logits.device):
            return False
    major, minor = torch.cuda.get_device_capability(logits.device)
    cc = major * 10 + minor
    if cc not in CUDNN_TOPK_AUTO_SHAPES:
        return False
    if backend == "auto" and (
        (rows, cols) not in CUDNN_TOPK_AUTO_SHAPES[cc]
        or (top_k, next_n, compress_ratio) != (512, 1, 1)
    ):
        return False
    # Deferred feature probe only: never builds a plan or imports a kernel.
    from .backend import frontend_api

    return frontend_api(cc) is not None
