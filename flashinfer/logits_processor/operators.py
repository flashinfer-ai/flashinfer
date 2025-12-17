"""
Copyright (c) 2025 by FlashInfer team.

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

from typing import Any, Optional, Tuple, Union

import torch

from flashinfer.sampling import get_sampling_module
from flashinfer.utils import _get_cache_buf, device_support_pdl

from .op import ParameterizedOp
from .types import TaggedTensor, TensorType


def _to_tensor_scalar_tuple(
    x: Union[torch.Tensor, float, int],
) -> Tuple[Optional[torch.Tensor], Union[float, int]]:
    if isinstance(x, torch.Tensor):
        return (x, 0 if x.dtype == torch.int32 else 0.0)
    else:
        return (None, x)


class TemperatureOp(ParameterizedOp):
    """
    Temperature scaling operator.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.LOGITS`

    Parameters
    ----------
    temperature : float or torch.Tensor
        Temperature value for scaling.
    """

    IN = TensorType.LOGITS
    OUT = TensorType.LOGITS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        temperature = self._get_param("temperature", kwargs, required=True)
        maybe_temperature_arr, temperature_val = _to_tensor_scalar_tuple(temperature)
        if maybe_temperature_arr is None and (
            not isinstance(temperature_val, float) or temperature_val <= 0
        ):
            raise ValueError("Temperature must be positive float or a tensor array")

        if maybe_temperature_arr is not None:
            temperature = maybe_temperature_arr
        else:
            temperature = temperature_val

        scaled_logits = tensor.data / temperature

        return TaggedTensor(scaled_logits, output_type)


class SoftmaxOp(ParameterizedOp):
    """
    Softmax operator.

    Converts logits to probabilities using softmax function.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    enable_pdl: bool, optional
        Whether to enable PDL for the fused kernel.
    """

    IN = TensorType.LOGITS
    OUT = TensorType.PROBS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        enable_pdl = self.default_params.get("enable_pdl", None)
        if enable_pdl is None:
            enable_pdl = device_support_pdl(tensor.data.device)

        probs = torch.softmax(tensor.data, dim=-1)
        return TaggedTensor(probs, output_type)


class ProbsTopKOp(ParameterizedOp):
    """
    Top-k filtering operator for probabilities.

    Keeps top-k probabilities, zeros out others, and renormalizes.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    top_k : int or torch.Tensor
        Number of top tokens to keep.

    See Also
    --------
    :meth:`~flashinfer.sampling.top_k_renorm_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.PROBS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        top_k = self._get_param("top_k", kwargs, required=True)
        maybe_top_k_arr, top_k_val = _to_tensor_scalar_tuple(top_k)

        if maybe_top_k_arr is None and (
            not isinstance(top_k_val, int) or top_k_val <= 0
        ):
            raise ValueError("top_k must be a positive integer or a tensor array")

        # Allocate row_states buffer for multi-CTA kernel (1MB is enough for any GPU)
        row_states_buffer = _get_cache_buf(
            f"top_k_renorm_probs_row_states_{tensor.data.device}",
            1024 * 1024,
            tensor.data.device,
            zero_init=True,
        )
        renorm_probs = get_sampling_module().top_k_renorm_probs(
            tensor.data, maybe_top_k_arr, top_k_val, row_states_buffer
        )

        return TaggedTensor(renorm_probs, output_type)


class LogitsTopKOp(ParameterizedOp):
    """
    Top-k filtering operator for logits.

    Masks rejected logits to -inf.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.LOGITS`

    Parameters
    ----------
    top_k : int or torch.Tensor
        Number of top tokens to keep.

    See Also
    --------
    :class:`~flashinfer.sampling.top_k_mask_logits`
    """

    IN = TensorType.LOGITS
    OUT = TensorType.LOGITS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        top_k = self._get_param("top_k", kwargs, required=True)
        maybe_top_k_arr, top_k_val = _to_tensor_scalar_tuple(top_k)

        if maybe_top_k_arr is None and (
            not isinstance(top_k_val, int) or top_k_val <= 0
        ):
            raise ValueError("top_k must be a positive integer or a tensor array")

        # Allocate row_states buffer for multi-CTA kernel (1MB is enough for any GPU)
        row_states_buffer = _get_cache_buf(
            f"top_k_mask_logits_row_states_{tensor.data.device}",
            1024 * 1024,
            tensor.data.device,
            zero_init=True,
        )
        masked_logits = get_sampling_module().top_k_mask_logits(
            tensor.data, maybe_top_k_arr, top_k_val, row_states_buffer
        )
        return TaggedTensor(masked_logits, output_type)


class TopPOp(ParameterizedOp):
    """
    Top-p (nucleus) filtering operator.

    Keeps tokens with cumulative probability up to threshold p, zeros out others, and renormalizes.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    top_p : float or torch.Tensor
        Cumulative probability threshold in (0, 1].

    See Also
    --------
    :meth:`~flashinfer.sampling.top_p_renorm_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.PROBS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        top_p = self._get_param("top_p", kwargs, required=True)
        maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)

        if maybe_top_p_arr is None and not (0 < top_p_val <= 1):
            raise ValueError("top_p must be float in (0, 1] or a tensor array")

        renorm_probs = get_sampling_module().top_p_renorm_probs(
            tensor.data, maybe_top_p_arr, top_p_val
        )

        return TaggedTensor(renorm_probs, output_type)


class MinPOp(ParameterizedOp):
    """
    Min-p filtering operator.

    Keeps tokens with probability at least p times the maximum probability, zeros out others, and renormalizes.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    min_p : float or torch.Tensor
        Minimum probability threshold as ratio of max probability.

    See Also
    --------
    :meth:`~flashinfer.sampling.min_p_renorm_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.PROBS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        min_p = self._get_param("min_p", kwargs, required=True)
        maybe_min_p_arr, min_p_val = _to_tensor_scalar_tuple(min_p)

        if maybe_min_p_arr is None and not (0 < min_p_val <= 1):
            raise ValueError("min_p must be float in (0, 1] or a tensor array")

        if maybe_min_p_arr is not None:
            min_p_mask = tensor.data >= (
                maybe_min_p_arr.unsqueeze(-1) * tensor.data.max(dim=-1, keepdim=True)[0]
            )
        else:
            min_p_mask = tensor.data >= (
                min_p_val * tensor.data.max(dim=-1, keepdim=True)[0]
            )

        masked_probs = tensor.data.clone()
        masked_probs[~min_p_mask] = 0
        probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)

        return TaggedTensor(probs, output_type)


class ProbsSampleOp(ParameterizedOp):
    """
    Sampling operator for probabilities.

    Samples token indices from probability distribution using inverse transform sampling.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.INDICES`

    Parameters
    ----------
    deterministic : bool, optional
        Whether to use deterministic kernel implementation.
    indices : torch.Tensor, optional
        Indices for batched sampling.
    generator : torch.Generator, optional
        Random number generator.

    See Also
    --------
    :meth:`~flashinfer.sampling.sampling_from_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.INDICES

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().sampling_from_probs(
            tensor.data, indices, deterministic, generator
        )

        return TaggedTensor(samples, output_type)


class LogitsSampleOp(ParameterizedOp):
    """
    Sampling operator for logits.

    Samples token indices from logits using Gumbel-max trick.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.INDICES`

    Parameters
    ----------
    deterministic : bool, optional
        Whether to use deterministic kernel implementation.
    indices : torch.Tensor, optional
        Indices for batched sampling.
    generator : torch.Generator, optional
        Random number generator.

    See Also
    --------
    :meth:`~flashinfer.sampling.sampling_from_logits`
    """

    IN = TensorType.LOGITS
    OUT = TensorType.INDICES

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().sampling_from_logits(
            tensor.data, indices, deterministic, generator
        )

        return TaggedTensor(samples, output_type)


# Fused operators
class FusedTemperatureSoftmaxOp(ParameterizedOp):
    """
    Fused temperature scaling and softmax operator.

    :attr:`TensorType.LOGITS` -> :attr:`TensorType.PROBS`

    Parameters
    ----------
    enable_pdl: bool, optional
        Whether to enable PDL for the fused kernel.
    temperature : float or torch.Tensor
        Temperature value for scaling.

    See Also
    --------
    :meth:`~flashinfer.sampling.softmax`
    """

    IN = TensorType.LOGITS
    OUT = TensorType.PROBS

    def __init__(self, enable_pdl: Optional[bool] = None, **default_params: Any):
        super().__init__(enable_pdl=enable_pdl, **default_params)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        temperature = self._get_param("temperature", kwargs, required=True)
        maybe_temperature_arr, temperature_val = _to_tensor_scalar_tuple(temperature)
        if maybe_temperature_arr is None and (
            not isinstance(temperature_val, float) or temperature_val <= 0
        ):
            raise ValueError("Temperature must be positive float or a tensor array")

        workspace_buffer = _get_cache_buf(
            "softmax_workspace", 1024 * 1024, tensor.data.device
        )

        enable_pdl = self.default_params.get("enable_pdl", None)
        if enable_pdl is None:
            enable_pdl = device_support_pdl(tensor.data.device)

        probs = get_sampling_module().softmax(
            workspace_buffer,
            tensor.data,
            maybe_temperature_arr,
            temperature_val,
            enable_pdl,
        )

        return TaggedTensor(probs, output_type)


class FusedProbsTopKSampleOp(ParameterizedOp):
    """
    Fused top-k filtering and sampling operator for probabilities.

    Use rejection sampling to directly sample from the top-k probabilities.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.INDICES`

    Parameters
    ----------
    deterministic : bool, optional
        Whether to use deterministic kernel implementation.
    top_k : int or torch.Tensor
        Number of top tokens to keep.
    indices : torch.Tensor, optional
        Indices for batched sampling.
    generator : torch.Generator, optional
        Random number generator.

    See Also
    --------
    :meth:`~flashinfer.sampling.top_k_sampling_from_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.INDICES

    def __init__(self, deterministic: bool = True, **default_params: Any):
        super().__init__(deterministic=deterministic, **default_params)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        top_k = self._get_param("top_k", kwargs, required=True)
        maybe_top_k_arr, top_k_val = _to_tensor_scalar_tuple(top_k)

        if maybe_top_k_arr is None and (
            not isinstance(top_k_val, int) or top_k_val <= 0
        ):
            raise ValueError("top_k must be a positive integer or a tensor array")

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().top_k_sampling_from_probs(
            tensor.data, indices, maybe_top_k_arr, top_k_val, deterministic, generator
        )

        return TaggedTensor(samples, output_type)


class FusedProbsTopPSampleOp(ParameterizedOp):
    """
    Fused top-p filtering and sampling operator for probabilities.

    Use rejection sampling to directly sample from the top-p probabilities.

    :attr:`TensorType.PROBS` -> :attr:`TensorType.INDICES`

    Parameters
    ----------
    deterministic : bool, optional
        Whether to use deterministic kernel implementation.
    top_p : float or torch.Tensor
        Cumulative probability threshold.
    indices : torch.Tensor, optional
        Indices for batched sampling.
    generator : torch.Generator, optional
        Random number generator.

    See Also
    --------
    :meth:`~flashinfer.sampling.top_p_sampling_from_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.INDICES

    def __init__(self, deterministic: bool = True, **default_params: Any):
        super().__init__(deterministic=deterministic, **default_params)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        top_p = self._get_param("top_p", kwargs, required=True)
        maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)

        if maybe_top_p_arr is None and not (0 < top_p_val <= 1):
            raise ValueError("top_p must be float in (0, 1] or a tensor array")

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().top_p_sampling_from_probs(
            tensor.data, indices, maybe_top_p_arr, top_p_val, deterministic, generator
        )

        return TaggedTensor(samples, output_type)


class FusedProbsMinPSampleOp(ParameterizedOp):
    """
    Fused min-p filtering and sampling operator for probabilities.

    Use rejection sampling to directly sample from the min-p probabilities.

    PROBS → INDICES

    Parameters
    ----------
    deterministic : bool, optional
        Whether to use deterministic kernel implementation.
    min_p : float or torch.Tensor
        Minimum probability threshold.
    indices : torch.Tensor, optional
        Indices for batched sampling.
    generator : torch.Generator, optional
        Random number generator.

    See Also
    --------
    :meth:`~flashinfer.sampling.min_p_sampling_from_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.INDICES

    def __init__(self, deterministic: bool = True, **default_params: Any):
        super().__init__(deterministic=deterministic, **default_params)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        min_p = self._get_param("min_p", kwargs, required=True)
        maybe_min_p_arr, min_p_val = _to_tensor_scalar_tuple(min_p)

        if maybe_min_p_arr is None and not (0 < min_p_val <= 1):
            raise ValueError("min_p must be float in (0, 1] or a tensor array")

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().min_p_sampling_from_probs(
            tensor.data, indices, maybe_min_p_arr, min_p_val, deterministic, generator
        )

        return TaggedTensor(samples, output_type)


class FusedProbsTopKTopPSampleOp(ParameterizedOp):
    """
    Fused top-k, top-p filtering and sampling operator for probabilities.

    Use rejection sampling to directly sample from the probabilities, top-k and top-p filtering are applied jointly (rather than applying first -> renormalize -> second).

    :attr:`TensorType.PROBS` -> :attr:`TensorType.INDICES`

    Parameters
    ----------
    deterministic : bool, optional
        Whether to use deterministic kernel implementation.
    top_k : int or torch.Tensor
        Number of top tokens to keep.
    top_p : float or torch.Tensor
        Cumulative probability threshold.
    indices : torch.Tensor, optional
        Indices for batched sampling.
    generator : torch.Generator, optional
        Random number generator.

    See Also
    --------
    :meth:`~flashinfer.sampling.top_k_top_p_sampling_from_probs`
    """

    IN = TensorType.PROBS
    OUT = TensorType.INDICES

    def __init__(self, deterministic: bool = True, **default_params: Any):
        super().__init__(deterministic=deterministic, **default_params)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        top_k = self._get_param("top_k", kwargs, required=True)
        maybe_top_k_arr, top_k_val = _to_tensor_scalar_tuple(top_k)

        top_p = self._get_param("top_p", kwargs, required=True)
        maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)

        if maybe_top_k_arr is None and (
            not isinstance(top_k_val, int) or top_k_val <= 0
        ):
            raise ValueError("top_k must be a positive integer or a tensor array")

        if maybe_top_p_arr is None and not (0 < top_p_val <= 1):
            raise ValueError("top_p must be float in (0, 1] or a tensor array")

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().top_k_top_p_sampling_from_probs(
            tensor.data,
            indices,
            maybe_top_k_arr,
            top_k_val,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            generator,
        )

        return TaggedTensor(samples, output_type)
