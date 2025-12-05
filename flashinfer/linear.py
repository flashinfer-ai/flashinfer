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

"""
Simple linear/GEMM operations for FlashInfer.

This module provides basic linear algebra operations that can be used alongside
other FlashInfer kernels for complete LLM inference pipelines.

The implementations use PyTorch's optimized backends (cuBLAS) under the hood.
"""

from typing import Optional
import torch
import torch.nn.functional as F

from .utils import register_custom_op, register_fake_op


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Linear transformation without bias.

    Computes ``output = input @ weight.T``

    This is equivalent to ``torch.nn.functional.linear(input, weight, bias=None)``.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(..., in_features)``.
    weight : torch.Tensor
        Weight matrix of shape ``(out_features, in_features)``.
    out : Optional[torch.Tensor]
        Optional output tensor. If provided, the result will be written to this tensor.
        Shape must be ``(..., out_features)``.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(..., out_features)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size, seq_len = 2, 128
    >>> in_features, out_features = 4096, 4096
    >>> x = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.bfloat16)
    >>> w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    >>> y = flashinfer.linear(x, w)
    >>> y.shape
    torch.Size([2, 128, 4096])
    """
    if out is not None:
        # Use addmm for inplace-like behavior when output is provided
        # Reshape for batch matmul if needed
        input_shape = input.shape
        if input.ndim > 2:
            input_2d = input.view(-1, input.shape[-1])
            result = torch.mm(input_2d, weight.t())
            out.copy_(result.view(input_shape[:-1] + (weight.shape[0],)))
        else:
            torch.mm(input, weight.t(), out=out)
        return out
    else:
        return F.linear(input, weight, bias=None)


def linear_with_bias(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Linear transformation with bias.

    Computes ``output = input @ weight.T + bias``

    This is equivalent to ``torch.nn.functional.linear(input, weight, bias)``.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(..., in_features)``.
    weight : torch.Tensor
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.Tensor
        Bias vector of shape ``(out_features,)``.
    out : Optional[torch.Tensor]
        Optional output tensor. If provided, the result will be written to this tensor.
        Shape must be ``(..., out_features)``.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(..., out_features)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size, seq_len = 2, 128
    >>> in_features, out_features = 4096, 4096
    >>> x = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.bfloat16)
    >>> w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    >>> b = torch.randn(out_features, device="cuda", dtype=torch.bfloat16)
    >>> y = flashinfer.linear_with_bias(x, w, b)
    >>> y.shape
    torch.Size([2, 128, 4096])
    """
    if out is not None:
        result = F.linear(input, weight, bias)
        out.copy_(result)
        return out
    else:
        return F.linear(input, weight, bias)


def bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Batched matrix multiplication.

    Computes ``output = input @ mat2``

    Parameters
    ----------
    input : torch.Tensor
        First batch of matrices, shape ``(batch, n, m)``.
    mat2 : torch.Tensor
        Second batch of matrices, shape ``(batch, m, p)``.
    out : Optional[torch.Tensor]
        Optional output tensor of shape ``(batch, n, p)``.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(batch, n, p)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 32
    >>> n, m, p = 128, 256, 512
    >>> a = torch.randn(batch_size, n, m, device="cuda", dtype=torch.bfloat16)
    >>> b = torch.randn(batch_size, m, p, device="cuda", dtype=torch.bfloat16)
    >>> c = flashinfer.bmm(a, b)
    >>> c.shape
    torch.Size([32, 128, 512])
    """
    if out is not None:
        torch.bmm(input, mat2, out=out)
        return out
    else:
        return torch.bmm(input, mat2)


def matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""General matrix multiplication with broadcasting.

    Computes ``output = input @ other`` with NumPy-style broadcasting.

    Parameters
    ----------
    input : torch.Tensor
        First tensor.
    other : torch.Tensor
        Second tensor.
    out : Optional[torch.Tensor]
        Optional output tensor.

    Returns
    -------
    torch.Tensor
        Output tensor.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> a = torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16)
    >>> b = torch.randn(4, 5, device="cuda", dtype=torch.bfloat16)
    >>> c = flashinfer.matmul(a, b)
    >>> c.shape
    torch.Size([2, 3, 5])
    """
    if out is not None:
        torch.matmul(input, other, out=out)
        return out
    else:
        return torch.matmul(input, other)


def embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    r"""Embedding lookup.

    Retrieves embeddings from the weight matrix based on input indices.

    Parameters
    ----------
    input : torch.Tensor
        Tensor of indices, shape ``(...)``.
    weight : torch.Tensor
        Embedding weight matrix, shape ``(num_embeddings, embedding_dim)``.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(..., embedding_dim)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> vocab_size, embed_dim = 32000, 4096
    >>> weight = torch.randn(vocab_size, embed_dim, device="cuda", dtype=torch.bfloat16)
    >>> indices = torch.tensor([1, 2, 3, 4], device="cuda")
    >>> embeddings = flashinfer.embedding(indices, weight)
    >>> embeddings.shape
    torch.Size([4, 4096])
    """
    return F.embedding(input, weight)


# Register custom ops for torch.compile compatibility

@register_custom_op("flashinfer::linear", mutates_args=("out",))
def _linear_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return linear(input, weight, out)


@register_fake_op("flashinfer::linear")
def _linear_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (weight.shape[0],)
    return input.new_empty(out_shape)


@register_custom_op("flashinfer::linear_with_bias", mutates_args=("out",))
def _linear_with_bias_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return linear_with_bias(input, weight, bias, out)


@register_fake_op("flashinfer::linear_with_bias")
def _linear_with_bias_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (weight.shape[0],)
    return input.new_empty(out_shape)


@register_custom_op("flashinfer::bmm", mutates_args=("out",))
def _bmm_op(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return bmm(input, mat2, out)


@register_fake_op("flashinfer::bmm")
def _bmm_fake(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out_shape = (input.shape[0], input.shape[1], mat2.shape[2])
    return input.new_empty(out_shape)


@register_custom_op("flashinfer::embedding", mutates_args=())
def _embedding_op(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return embedding(input, weight)


@register_fake_op("flashinfer::embedding")
def _embedding_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    out_shape = input.shape + (weight.shape[1],)
    return weight.new_empty(out_shape)

