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

try:
    from . import _kernels
except ImportError as e:
    import os
    import logging

    if os.environ.get("BUILD_DOC", "0") == "1":
        _kernels = None
        logging.warning("Kernels are not loaded in documentation build mode.")
    else:
        raise e


def sampling_from_probs(probs: torch.Tensor, uniform_samples: torch.Tensor):
    r"""Fused GPU kernel for category sampling from probabilities.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    uniform_samples: torch.Tensor
        The uniform samples used as needle for sampling, shape ``(batch_size,)``.
        Expected to be uniformly distributed in ``[0, 1)``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape (batch_size,).

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> uniform_samples = torch.rand(batch_size).to(0)
    >>> samples = flashinfer.sampling.sampling_from_probs(norm_prob, uniform_samples)
    >>> samples
    tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)

    Notes
    -----
    This function expects float32 inputs, and the output is int32.
    """
    return _kernels.sampling_from_probs(probs, uniform_samples)


def top_p_sampling_from_probs(
    probs: torch.Tensor, uniform_samples: torch.Tensor, top_p: float
):
    r"""Fused GPU kernel for top-p sampling (nucleus sampling) from probabilities,
    this operator implements GPU-based rejection sampling without explicit sorting.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    uniform_samples: torch.Tensor
        The uniform samples used as needle for sampling, shape ``(max_top_p_rounds, batch_size,)``,
        where the first dimension is the maximum number of rounds for rejection sampling.
        Expected to be uniformly distributed in ``[0, 1)``.
    top_p: float
        The threshold for top-p sampling.

    Returns
    -------
    (samples, success): Tuple[torch.Tensor, torch.Tensor]
        samples: torch.Tensor
            Sampled categories, shape ``(batch_size,)``.
        success: torch.Tensor
            Whether the sampling is successful within ``max_top_p_rounds`` rounds,
            shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> max_top_p_rounds = 3
    >>> top_p = 0.5
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> uniform_samples = torch.rand(max_top_p_rounds, batch_size).to(0)
    >>> samples, success = flashinfer.sampling.top_p_sampling_from_probs(norm_prob, uniform_samples, top_p)
    >>> samples
    tensor([1, 2, 0, 4], device='cuda:0', dtype=torch.int32)
    >>> success
    tensor([True, True, True, True], device='cuda:0')

    Notes
    -----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_top_p_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.
    """
    return _kernels.top_p_sampling_from_probs(probs, uniform_samples, top_p)


def top_k_sampling_from_probs(
    probs: torch.Tensor, uniform_samples: torch.Tensor, top_k: int
):
    r"""Fused GPU kernel for top-k sampling from probabilities,
    this operator implements GPU-based rejection sampling without explicit sorting.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    uniform_samples: torch.Tensor
        The uniform samples used as needle for sampling, shape ``(max_top_k_rounds, batch_size,)``,
        where the first dimension is the maximum number of rounds for rejection sampling.
        Expected to be uniformly distributed in ``[0, 1)``.
    top_k: int
        The k in "top-k".

    Returns
    -------
    (samples, success): Tuple[torch.Tensor, torch.Tensor]
        samples: torch.Tensor
            Sampled categories, shape ``(batch_size,)``.
        success: torch.Tensor
            Whether the sampling is successful within ``max_top_k_rounds`` rounds,
            shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> max_top_k_rounds = 3
    >>> top_k = 1
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> uniform_samples = torch.rand(max_top_k_rounds, batch_size).to(0)
    >>> samples, success = flashinfer.sampling.top_k_sampling_from_probs(norm_prob, uniform_samples, top_k)
    >>> samples
    tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)
    >>> success
    tensor([True, True, True, True], device='cuda:0')

    Notes
    -----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_top_k_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.
    """
    return _kernels.top_k_sampling_from_probs(probs, uniform_samples, top_k)
