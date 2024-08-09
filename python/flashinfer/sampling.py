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
from typing import Tuple, Union

# mypy: disable-error-code="attr-defined"
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


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def sampling_from_probs(
    probs: torch.Tensor, uniform_samples: torch.Tensor, deterministic: bool = True
) -> torch.Tensor:
    r"""Fused GPU kernel for category sampling from probabilities.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    uniform_samples: torch.Tensor
        The uniform samples used as needle for sampling, shape ``(batch_size,)``.
        Expected to be uniformly distributed in ``[0, 1)``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.

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
    return _kernels.sampling_from_probs(probs, uniform_samples, deterministic)


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.

    Returns
    -------
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
    return _kernels.top_p_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_p), deterministic
    )


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.

    Returns
    -------
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
    return _kernels.top_k_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_k), deterministic
    )


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused GPU kernel for `min_p sampling <https://arxiv.org/abs/2407.01082>`_ from probabilities,

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
    min_p: torch.Tensor
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for min-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.

    Returns
    -------
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
    <torch._C.Generator object at 0x7f8b3db06df0>
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> max_rounds = 3
    >>> min_p = torch.full((batch_size,), 0.05).to(0)
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> uniform_samples = torch.rand(max_rounds, batch_size).to(0)
    >>> samples, success = flashinfer.sampling.min_p_sampling_from_probs(norm_prob, uniform_samples, min_p)
    >>> samples
    tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)
    >>> success
    tensor([True, True, True, True], device='cuda:0')

    Notes
    -----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.
    """
    return _kernels.min_p_sampling_from_probs(
        probs, uniform_samples, _to_tensor_scalar_tuple(min_p), deterministic
    )


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused GPU kernel for top-k and top-p sampling from probabilities,

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
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    filter_apply_order: str
        The order of applying top-k and top-p sampling, should be either ``"top_k_first"`` or ``"joint"``.
        If ``"top_k_first"``, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.

    Returns
    -------
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
    >>> max_rounds = 3
    >>> top_p = torch.full((batch_size,), 0.2).to(0)
    >>> top_k = torch.full((batch_size,), 2).to(0)
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> uniform_samples = torch.rand(max_rounds, batch_size).to(0)
    >>> samples, success = flashinfer.sampling.top_k_top_p_sampling_from_probs(norm_prob, uniform_samples, top_k, top_p)
    >>> samples
    tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)
    >>> success
    tensor([True, True, True, True], device='cuda:0')

    Notes
    -----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.
    """
    if filter_apply_order == "top_k_first":
        masked_logits = top_k_mask_logits(probs, top_k)
        probs = torch.softmax(masked_logits, dim=-1)
        return top_p_sampling_from_probs(probs, uniform_samples, top_p, deterministic)
    elif filter_apply_order == "joint":
        return _kernels.top_k_top_p_sampling_from_probs(
            probs,
            uniform_samples,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def top_p_renorm_prob(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float], eps: float = 1e-5
) -> torch.Tensor:
    r"""Fused GPU kernel for renormalizing probabilities by top-p thresholding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-p threshold for for
        re-normalizing probabilities, should be in ``(0, 1)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We mask out the probabilities less than `threshold` where the cumulative sum
        of ``probs[probs >= threshold]`` is `top_p`, and renormalize the probabilities.
    eps: float
        The epsilon value for numerical stability.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    This combination of ``top_p_renorm_prob`` and ``sampling_from_probs`` should be equivalent to
    ``top_p_sampling_from_probs``.
    """
    return _kernels.top_p_renorm_prob(probs, _to_tensor_scalar_tuple(top_p), eps)


def top_k_renorm_prob(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int], eps: float = 1e-5
) -> torch.Tensor:
    r"""Fused GPU kernel for renormalizing probabilities by top-k thresholding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for for
        for re-normalizing probabilities, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k probabilities, set the rest to zero, and renormalize the probabilities.
    eps: float
        The epsilon value for numerical stability.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Note
    ----
    This combination of ``top_k_renorm_prob`` and ``sampling_from_probs`` should be equivalent to
    ``top_k_sampling_from_probs``.
    """
    return _kernels.top_k_renorm_prob(probs, _to_tensor_scalar_tuple(top_k), eps)


def top_k_mask_logits(
    logits: torch.Tensor, top_k: Union[torch.Tensor, int], eps: float = 1e-5
) -> torch.Tensor:
    r"""Fused GPU kernel for masking logits by top-k thresholding.

    Parameters
    ----------
    logits: torch.Tensor
        Logits before softmax, shape ``(batch_size, num_classes)``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for for
        for masking logits, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k logits, set the rest to negative infinity.
    eps: float
        The epsilon value for numerical stability.

    Returns
    -------
    masked_logits: torch.Tensor
        Masked logits, shape ``(batch_size, num_classes)``.

    Note
    ----
    The combination of ``top_k_mask_logits`` and ``softmax`` should be equivalent to ``top_k_renorm_prob``.
    """
    return _kernels.top_k_mask_logits(logits, _to_tensor_scalar_tuple(top_k), eps)


def chain_speculative_sampling(
    draft_probs,
    draft_token_ids,
    uniform_samples,
    target_probs,
    deterministic: bool = True,
) -> torch.Tensor:
    r"""Fused-GPU kernel for speculative sampling for sequence generation (proposed in
    paper `Accelerating Large Language Model Decoding with Speculative Sampling <https://arxiv.org/pdf/2302.01318>`_),
    where the draft model generates a sequence(chain) of tokens for each request.

    Parameters
    ----------
    draft_probs: torch.Tensor
        The probability over vocabulary generated by draft model.
        Shape: ``(batch_size, num_speculate_tokens, vocab_size)``
    draft_token_ids: torch.Tensor
        The draft model's generated token indices.
        Shape: ``(batch_size, num_specutate_tokens)``
    uniform_samples: torch.Tensor
        The uniform samples used as needle for sampling, shape ``(batch_size, num_speculate_tokens + 1)``.
        Expected to be uniformly distributed in ``[0, 1)``.
    target_probs: torch.Tensor
        The probability over vocabulary generated by target model.
        Compared to input :attr:`draft_probs`, the target model's probability has an additional
        slot at the end because the target model will generate one more token than the draft model.
        Shape: ``(batch_size, num_speculate_tokens + 1, vocab_size)``
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.

    Returns
    -------
    output_token_ids: torch.Tensor
        The output token indices verified by the target model, rejected samples are
        padded with ``-1``.
        Compared to input :attr:`draft_token_ids`, the output tensor has an additional
        token index at the end for the final token, if all previous tokens are accepted,
        another "bonus" token will be sampled from the target model's probability.
        Shape: (batch_size, num_specutate_tokens + 1)
    """
    return _kernels.chain_speculative_sampling(
        draft_probs, draft_token_ids, uniform_samples, target_probs, deterministic
    )
