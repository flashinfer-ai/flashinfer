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

from types import SimpleNamespace
from typing import Optional, Tuple, Union

import torch

from .jit import FLASHINFER_CSRC_DIR, has_prebuilt_ops, load_cuda_ops
from .utils import get_cuda_stream, register_custom_op, register_fake_op

_sampling_module = None


def get_sampling_module():
    global _sampling_module
    if _sampling_module is None:
        if has_prebuilt_ops:
            from . import _kernels

            module = _kernels
        else:
            module = load_cuda_ops(
                "sampling",
                [
                    FLASHINFER_CSRC_DIR / "sampling.cu",
                    FLASHINFER_CSRC_DIR / "renorm.cu",
                    FLASHINFER_CSRC_DIR / "flashinfer_sampling_ops.cu",
                ],
            )

        # torch library for sampling_from_probs

        @register_custom_op("flashinfer::sampling_from_probs", mutates_args=())
        def sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            deterministic: bool,
        ) -> torch.Tensor:
            with probs.device as device:
                probs = probs.float()
                uniform_samples = uniform_samples.float()
                samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
                module.sampling_from_probs(
                    probs,
                    uniform_samples,
                    samples,
                    deterministic,
                    get_cuda_stream(device),
                )
                return samples

        @register_fake_op("flashinfer::sampling_from_probs")
        def _fake_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            deterministic: bool,
        ) -> torch.Tensor:
            return torch.empty(probs.size(0), dtype=torch.int32, device=probs.device)

        # torch library for top_p_sampling_from_probs

        @register_custom_op("flashinfer::top_p_sampling_from_probs", mutates_args=())
        def top_p_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with probs.device as device:
                probs = probs.float()
                uniform_samples = uniform_samples.float()
                maybe_top_p_arr = (
                    maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
                )
                samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
                success = torch.empty(probs.size(0), dtype=torch.bool, device=device)
                module.top_p_sampling_from_probs(
                    probs,
                    uniform_samples,
                    samples,
                    success,
                    maybe_top_p_arr,
                    top_p_val,
                    deterministic,
                    get_cuda_stream(device),
                )
                return samples, success

        @register_fake_op("flashinfer::top_p_sampling_from_probs")
        def _fake_top_p_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            sample = torch.empty(probs.size(0), dtype=torch.int32, device=probs.device)
            success = torch.empty(probs.size(0), dtype=torch.bool, device=probs.device)
            return sample, success

        # torch library for top_k_sampling_from_probs

        @register_custom_op("flashinfer::top_k_sampling_from_probs", mutates_args=())
        def top_k_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            deterministic: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with probs.device as device:
                probs = probs.float()
                uniform_samples = uniform_samples.float()
                maybe_top_k_arr = (
                    maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
                )
                samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
                success = torch.empty(probs.size(0), dtype=torch.bool, device=device)
                module.top_k_sampling_from_probs(
                    probs,
                    uniform_samples,
                    samples,
                    success,
                    maybe_top_k_arr,
                    top_k_val,
                    deterministic,
                    get_cuda_stream(device),
                )
                return samples, success

        @register_fake_op("flashinfer::top_k_sampling_from_probs")
        def _fake_top_k_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            deterministic: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            sample = torch.empty(probs.size(0), dtype=torch.int32, device=probs.device)
            success = torch.empty(probs.size(0), dtype=torch.bool, device=probs.device)
            return sample, success

        # torch library for min_p_sampling_from_probs

        @register_custom_op("flashinfer::min_p_sampling_from_probs", mutates_args=())
        def min_p_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            maybe_min_p_arr: Optional[torch.Tensor],
            min_p_val: float,
            deterministic: bool,
        ) -> torch.Tensor:
            with probs.device as device:
                probs = probs.float()
                uniform_samples = uniform_samples.float()
                maybe_min_p_arr = (
                    maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
                )
                samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
                module.min_p_sampling_from_probs(
                    probs,
                    uniform_samples,
                    samples,
                    maybe_min_p_arr,
                    min_p_val,
                    deterministic,
                    get_cuda_stream(device),
                )
                return samples

        # torch library for top_k_top_p_sampling_from_probs

        @register_custom_op(
            "flashinfer::top_k_top_p_sampling_from_probs", mutates_args=()
        )
        def top_k_top_p_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with probs.device as device:
                probs = probs.float()
                uniform_samples = uniform_samples.float()
                maybe_top_k_arr = (
                    maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
                )
                maybe_top_p_arr = (
                    maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
                )
                samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
                success = torch.empty(probs.size(0), dtype=torch.bool, device=device)
                module.top_k_top_p_sampling_from_probs(
                    probs,
                    uniform_samples,
                    samples,
                    success,
                    maybe_top_k_arr,
                    top_k_val,
                    maybe_top_p_arr,
                    top_p_val,
                    deterministic,
                    get_cuda_stream(device),
                )
                return samples, success

        @register_fake_op("flashinfer::top_k_top_p_sampling_from_probs")
        def _fake_top_k_top_p_sampling_from_probs(
            probs: torch.Tensor,
            uniform_samples: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            sample = torch.empty(probs.size(0), dtype=torch.int32, device=probs.device)
            success = torch.empty(probs.size(0), dtype=torch.bool, device=probs.device)
            return sample, success

        # torch library for top_p_renorm_probs

        @register_custom_op("flashinfer::top_p_renorm_probs", mutates_args=())
        def top_p_renorm_probs(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            with probs.device as device:
                probs = probs.float()
                maybe_top_p_arr = (
                    maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
                )
                renorm_probs = torch.empty_like(probs)
                module.top_p_renorm_probs(
                    probs,
                    renorm_probs,
                    maybe_top_p_arr,
                    top_p_val,
                    get_cuda_stream(device),
                )
                return renorm_probs

        @register_fake_op("flashinfer::top_p_renorm_probs")
        def _fake_top_p_renorm_probs(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            return torch.empty_like(probs)

        # torch library for top_k_renorm_probs

        @register_custom_op("flashinfer::top_k_renorm_probs", mutates_args=())
        def top_k_renorm_probs(
            probs: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            with probs.device as device:
                probs = probs.float()
                maybe_top_k_arr = (
                    maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
                )
                renorm_probs = torch.empty_like(probs)
                module.top_k_renorm_probs(
                    probs,
                    renorm_probs,
                    maybe_top_k_arr,
                    top_k_val,
                    get_cuda_stream(device),
                )
                return renorm_probs

        @register_fake_op("flashinfer::top_k_renorm_probs")
        def _fake_top_k_renorm_probs(
            probs: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            return torch.empty_like(probs)

        # torch library for top_k_mask_logits

        @register_custom_op("flashinfer::top_k_mask_logits", mutates_args=())
        def top_k_mask_logits(
            logits: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            with logits.device as device:
                logits = logits.float()
                maybe_top_k_arr = (
                    maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
                )
                mask_logits = torch.empty_like(logits)
                module.top_k_mask_logits(
                    logits,
                    mask_logits,
                    maybe_top_k_arr,
                    top_k_val,
                    get_cuda_stream(device),
                )
                return mask_logits

        @register_fake_op("flashinfer::top_k_mask_logits")
        def _fake_top_k_mask_logits(
            logits: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            return torch.empty_like(logits)

        # torch library for chain_speculative_sampling

        @register_custom_op(
            "flashinfer::chain_speculative_sampling",
            mutates_args=("output_accepted_token_num", "output_emitted_token_num"),
        )
        def chain_speculative_sampling(
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            uniform_samples: torch.Tensor,
            target_probs: torch.Tensor,
            output_accepted_token_num: torch.Tensor,
            output_emitted_token_num: torch.Tensor,
            deterministic: bool,
        ) -> torch.Tensor:
            with draft_probs.device as device:
                draft_probs = draft_probs.float()
                draft_token_ids = draft_token_ids.int()
                uniform_samples = uniform_samples.float()
                target_probs = target_probs.float()
                output_accepted_token_num = output_accepted_token_num.int()
                output_emitted_token_num = output_emitted_token_num.int()
                b, n = draft_token_ids.shape
                output_token_ids = torch.empty(
                    (b, n + 1), dtype=torch.int32, device=device
                )
                module.chain_speculative_sampling(
                    draft_probs,
                    draft_token_ids,
                    uniform_samples,
                    target_probs,
                    output_token_ids,
                    output_accepted_token_num,
                    output_emitted_token_num,
                    deterministic,
                    get_cuda_stream(device),
                )
                return output_token_ids

        @register_fake_op("flashinfer::chain_speculative_sampling")
        def _fake_chain_speculative_sampling(
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            uniform_samples: torch.Tensor,
            target_probs: torch.Tensor,
            output_accepted_token_num: torch.Tensor,
            output_emitted_token_num: torch.Tensor,
            deterministic: bool,
        ) -> torch.Tensor:
            b, n = draft_token_ids.shape
            device = draft_token_ids.device
            return torch.empty((b, n + 1), dtype=torch.int32, device=device)

        # Register the module
        _sampling_module = SimpleNamespace(
            sampling_from_probs=sampling_from_probs,
            top_p_sampling_from_probs=top_p_sampling_from_probs,
            top_k_sampling_from_probs=top_k_sampling_from_probs,
            min_p_sampling_from_probs=min_p_sampling_from_probs,
            top_k_top_p_sampling_from_probs=top_k_top_p_sampling_from_probs,
            top_p_renorm_probs=top_p_renorm_probs,
            top_k_renorm_probs=top_k_renorm_probs,
            top_k_mask_logits=top_k_mask_logits,
            chain_speculative_sampling=chain_speculative_sampling,
        )

    return _sampling_module


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    deterministic: bool = True,
    check_nan: bool = False,
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
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

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

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().sampling_from_probs(
        probs, uniform_samples, deterministic
    )


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    deterministic: bool = True,
    check_nan: bool = False,
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
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

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

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_top_p_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_k_sampling_from_probs
    top_p_renorm_probs
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().top_p_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_p), deterministic
    )


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    deterministic: bool = True,
    check_nan: bool = False,
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
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

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

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_top_k_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_p_sampling_from_probs
    top_k_renorm_probs
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().top_k_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_k), deterministic
    )


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    deterministic: bool = True,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for `min_p sampling <https://arxiv.org/abs/2407.01082>`_ from probabilities,

    this operator implements GPU-based rejection sampling without explicit sorting.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    uniform_samples: torch.Tensor
        The uniform samples used as needle for sampling, shape ``(batch_size,)``,
        Expected to be uniformly distributed in ``[0, 1)``.
    min_p: torch.Tensor
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for min-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    <torch._C.Generator object at 0x7f8b3db06df0>
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> min_p = torch.full((batch_size,), 0.05).to(0)
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> uniform_samples = torch.rand(batch_size).to(0)
    >>> samples = flashinfer.sampling.min_p_sampling_from_probs(norm_prob, uniform_samples, min_p)
    >>> samples
    tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.
    """
    # NOTE(Zihao): for backward compatibility (https://github.com/flashinfer-ai/flashinfer/pull/713)
    if uniform_samples.dim() == 2:
        # Take the first row (round) of uniform_samples
        uniform_samples = uniform_samples[0]

    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().min_p_sampling_from_probs(
        probs, uniform_samples, *_to_tensor_scalar_tuple(min_p), deterministic
    )


def top_k_top_p_sampling_from_logits(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused GPU kernel for top-k and top-p sampling from pre-softmax logits,

    this operator implements GPU-based rejection sampling without explicit sorting.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    logits: torch.Tensor
        Pre-softmax logits, shape ``(batch_size, num_classes)``.
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
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

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
    >>> top_p = 0.5
    >>> top_k = 3
    >>> logits = torch.rand(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> uniform_samples = torch.rand(max_rounds, batch_size).to(0)
    >>> samples, success = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, uniform_samples, top_k, top_p)
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32
    >>> success
    tensor([True, True, True, True], device='cuda:0')
    >>> probs = torch.softmax(logits, dim=-1)
    >>> probs
    tensor([[0.4788, 0.3085, 0.1716, 0.0085, 0.0327],
        [0.2358, 0.1787, 0.4307, 0.1145, 0.0404],
        [0.1358, 0.2831, 0.1764, 0.2318, 0.1729],
        [0.3613, 0.1122, 0.1029, 0.3415, 0.0821]], device='cuda:0')
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32)
    >>> success
    tensor([True, True, True, True], device='cuda:0')

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_k_mask_logits
    top_p_sampling_from_probs
    """
    if filter_apply_order == "top_k_first":
        masked_logits = top_k_mask_logits(probs, top_k)
        probs = torch.softmax(masked_logits, dim=-1)
        return top_p_sampling_from_probs(
            probs, uniform_samples, top_p, deterministic, check_nan=check_nan
        )
    elif filter_apply_order == "joint":
        probs = torch.softmax(probs, dim=-1)
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return get_sampling_module().top_k_top_p_sampling_from_probs(
            probs,
            uniform_samples,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
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
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

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

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    We encourage users to set ``max_rounds`` to a reasonable value, e.g., 32. The actual
    implementation usually use much fewer rounds for rejection sampling because of early stopping.

    See Also
    --------
    top_k_sampling_from_probs
    top_p_sampling_from_probs
    top_k_renorm_probs
    top_p_renorm_probs
    top_k_mask_logits
    """
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs, uniform_samples, top_p, deterministic, check_nan=check_nan
        )
    elif filter_apply_order == "joint":
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return get_sampling_module().top_k_top_p_sampling_from_probs(
            probs,
            uniform_samples,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
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

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.3
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> renormed_probs = flashinfer.sampling.top_p_renorm_probs(prob, top_p)
    >>> renormed_probs
    tensor([[0.0000, 0.4882, 0.0000, 0.5118, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
            [0.5181, 0.0000, 0.4819, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000, 0.0000]], device='cuda:0')

    Note
    ----
    This combination of ``top_p_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_p_sampling_from_probs``.

    See Also
    --------
    top_p_sampling_from_probs
    sampling_from_probs
    top_k_renorm_probs
    """
    return get_sampling_module().top_p_renorm_probs(
        probs, *_to_tensor_scalar_tuple(top_p)
    )


top_p_renorm_prob = top_p_renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
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

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 3
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> renormed_probs = flashinfer.sampling.top_k_renorm_probs(prob, top_k)
    >>> renormed_probs
    tensor([[0.3201, 0.3319, 0.0000, 0.3480, 0.0000],
            [0.2573, 0.0000, 0.3398, 0.4028, 0.0000],
            [0.3672, 0.0000, 0.3416, 0.0000, 0.2912],
            [0.0000, 0.4243, 0.2750, 0.0000, 0.3007]], device='cuda:0')

    Note
    ----
    This combination of ``top_k_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_k_sampling_from_probs``.

    See Also
    --------
    top_k_sampling_from_probs
    sampling_from_probs
    top_p_renorm_probs
    """
    return get_sampling_module().top_k_renorm_probs(
        probs, *_to_tensor_scalar_tuple(top_k)
    )


top_k_renorm_prob = top_k_renorm_probs


def top_k_mask_logits(
    logits: torch.Tensor, top_k: Union[torch.Tensor, int]
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

    Returns
    -------
    masked_logits: torch.Tensor
        Masked logits, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 3
    >>> logits = torch.randn(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> masked_logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
    >>> masked_logits
    tensor([[ 1.9269,  1.4873,  0.9007,    -inf,    -inf],
            [ 1.0783,  0.8008,  1.6806,    -inf,    -inf],
            [   -inf,  0.2415, -0.2316,  0.0418,    -inf],
            [ 0.8599, -0.3097,    -inf,  0.8034,    -inf]], device='cuda:0')

    Note
    ----
    The combination of ``top_k_mask_logits`` and ``softmax`` should be equivalent to ``top_k_renorm_probs``.

    See Also
    --------
    top_k_renorm_probs
    """
    return get_sampling_module().top_k_mask_logits(
        logits, *_to_tensor_scalar_tuple(top_k)
    )


def chain_speculative_sampling(
    draft_probs,
    draft_token_ids,
    uniform_samples,
    target_probs,
    maybe_output_accepted_token_num: Optional[torch.Tensor] = None,
    maybe_output_emitted_token_num: Optional[torch.Tensor] = None,
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
    maybe_output_accepted_token_num: Optional[torch.Tensor]
        The number of tokens that can be accepted if each token is considered independently for each request.
        This metric does not consider the fact that rejection sampling will stop at the first token that does not
        satisfy the probability requirement r < p/q.
        It only evaluates the alignment of draft model and target model.
        Shape: ``(batch_size)``
        If specified, the number of accepted token number will be added to this tensor inplace. Default is ``None``.
    maybe_output_emitted_token_num: Optional[torch.Tensor]
        The number of tokens that are finally emitted/generated for each request.
        Shape: ``(batch_size)``
        If specified, the number of emitted token number will be added to this tensor inplace. Default is ``None``.
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
    output_accepted_token_num: torch.Tensor
        The number of tokens that can be accepted if each token is considered independently for each request.
        This metric does not consider the fact that rejection sampling will stop at the first token that does not
        satisfy the probability requirement r < p/q.
        It only evaluates the alignment of draft model and target model.
        Shape: ``(batch_size)``
    output_emitted_token_num: torch.Tensor
        The number of tokens that are finally emitted/generated for each request.
        Shape: ``(batch_size)``

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 1
    >>> num_speculate_tokens = 2
    >>> vocab_size = 4
    >>> draft_probs = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1]]]).to(0)
    >>> # token 2 was sampled from draft model for the first token, and
    >>> # token 1 was sampled from draft model for the second token
    >>> draft_token_ids = torch.tensor([[2, 1]], dtype=torch.int32).to(0)
    >>> # uniform samples for rejection sampling
    >>> uniform_samples = torch.rand(batch_size, num_speculate_tokens + 1).to(0)
    tensor([[0.8823, 0.9150, 0.3829], device='cuda:0')
    >>> target_probs = torch.tensor([[[0.0, 0.1, 0.6, 0.3], [1.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.1, 0.1]]]).to(0)
    >>> output_token_ids, output_accepted_token_num, output_accepted_token_num =\
    ...     flashinfer.sampling.chain_speculative_sampling(
    ...         draft_probs, draft_token_ids, uniform_samples, target_probs)
    >>> # the first token is accepted, the second token is rejected and sampled from the difference
    >>> # between the target model and the draft model, the third token is padded with -1
    >>> output_token_ids
    tensor([[ 2,  0, -1]], device='cuda:0', dtype=torch.int32)
    >>> output_accepted_token_num
    tensor([1], device='cuda:0')
    >>> output_emitted_token_num
    tensor([1], device='cuda:0')
    """
    b = draft_probs.size(0)
    dev = draft_probs.device
    if maybe_output_accepted_token_num is None:
        output_accepted_token_num = torch.zeros(b, dtype=torch.int32, device=dev)
    else:
        output_accepted_token_num = maybe_output_accepted_token_num
    if maybe_output_emitted_token_num is None:
        output_emitted_token_num = torch.zeros(b, dtype=torch.int32, device=dev)
    else:
        output_emitted_token_num = maybe_output_emitted_token_num
    output_token_ids = get_sampling_module().chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        uniform_samples,
        target_probs,
        output_accepted_token_num,
        output_emitted_token_num,
        deterministic,
    )
    return output_token_ids, output_accepted_token_num, output_emitted_token_num
