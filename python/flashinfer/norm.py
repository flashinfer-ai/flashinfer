"""
Copyright (c) 2024 by FlashInfer team.

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

import torch

from .jit import load_cuda_ops, FLASHINFER_CSRC_DIR

_norm_module = None


def get_norm_module():
    global _norm_module
    if _norm_module is None:
        _norm_module = load_cuda_ops(
            "norm",
            [
                FLASHINFER_CSRC_DIR / "norm.cu",
                FLASHINFER_CSRC_DIR / "flashinfer_norm_ops.cu",
            ],
        )
    return _norm_module


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    get_norm_module().rmsnorm(out, input, weight, eps)
    return out


def fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
):
    r"""Fused add root mean square normalization.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    """
    get_norm_module().fused_add_rmsnorm(input, residual, weight, eps)


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor = None,
):
    r"""Gemma Root mean square normalization.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    get_norm_module().gemma_rmsnorm(out, input, weight, eps)
    return out


def gemma_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
):
    r"""Gemma Fused add root mean square normalization.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    """
    get_norm_module().gemma_fused_add_rmsnorm(input, residual, weight, eps)
