from typing import Any, Optional, Tuple, Union

import torch

from flashinfer.sampling import get_sampling_module

from .op import Op, ParameterizedOp
from .types import TaggedTensor, TensorType


def _to_tensor_scalar_tuple(
    x: Union[torch.Tensor, float, int]
) -> Tuple[Optional[torch.Tensor], Union[float, int]]:
    if isinstance(x, torch.Tensor):
        return (x, 0 if x.dtype == torch.int32 else 0.0)
    else:
        return (None, x)


class TemperatureOp(ParameterizedOp):
    """Temperature scaling: Logits -> Logits"""

    IN = TensorType.LOGITS
    OUT = TensorType.LOGITS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        temperature = self._get_param("temperature", kwargs)
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        scaled_logits = tensor.data / temperature

        return TaggedTensor(scaled_logits, output_type)


class SoftmaxOp(Op):
    """Softmax: Logits -> Probs"""

    IN = TensorType.LOGITS
    OUT = TensorType.PROBS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        probs = torch.softmax(tensor.data, dim=-1)

        return TaggedTensor(probs, output_type)


class ProbsTopKOp(ParameterizedOp):
    """TopK Renorm Probs: Probs -> Probs"""

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

        renorm_probs = get_sampling_module().top_k_renorm_probs(
            tensor.data, maybe_top_k_arr, top_k_val
        )

        return TaggedTensor(renorm_probs, output_type)


class LogitsTopKOp(ParameterizedOp):
    """TopK Mask Logits: Logits -> Logits"""

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

        masked_logits = get_sampling_module().top_k_mask_logits(
            tensor.data, maybe_top_k_arr, top_k_val
        )
        return TaggedTensor(masked_logits, output_type)


class TopPOp(ParameterizedOp):
    """TopP: Probs -> Probs"""

    IN = TensorType.PROBS
    OUT = TensorType.PROBS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        top_p = self._get_param("top_p", kwargs, required=True)
        maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)

        if maybe_top_p_arr is None and not (
            0 < top_p_val <= 1 and isinstance(top_p_val, float)
        ):
            raise ValueError("top_p must be float in (0, 1] or a tensor array")

        renorm_probs = get_sampling_module().top_p_renorm_probs(
            tensor.data, maybe_top_p_arr, top_p_val
        )

        return TaggedTensor(renorm_probs, output_type)


class MinPOp(ParameterizedOp):
    """MinP: Probs -> Probs"""

    IN = TensorType.PROBS
    OUT = TensorType.PROBS

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        min_p = self._get_param("min_p", kwargs, required=True)
        maybe_min_p_arr, min_p_val = _to_tensor_scalar_tuple(min_p)

        if maybe_min_p_arr is None and not (
            0 < min_p_val <= 1 and isinstance(min_p_val, float)
        ):
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
    """Sample: Probs -> Indices"""

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
    """Sample: Logits -> Indices"""

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
class FusedProbsTopKSampleOp(ParameterizedOp):
    """Fused TopK -> Sample: Probs -> Indices"""

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
    """Fused TopP -> Sample: Probs -> Indices"""

    IN = TensorType.PROBS
    OUT = TensorType.INDICES

    def __init__(self, deterministic: bool = True, **default_params: Any):
        super().__init__(deterministic=deterministic, **default_params)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        top_p = self._get_param("top_p", kwargs, required=True)
        maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)

        if maybe_top_p_arr is None and not (
            0 < top_p_val <= 1 and isinstance(top_p_val, float)
        ):
            raise ValueError("top_p must be float in (0, 1] or a tensor array")

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().top_p_sampling_from_probs(
            tensor.data, indices, maybe_top_p_arr, top_p_val, deterministic, generator
        )

        return TaggedTensor(samples, output_type)


class FusedProbsMinPSampleOp(ParameterizedOp):
    """Fused MinP -> Sample: Probs -> Indices"""

    IN = TensorType.PROBS
    OUT = TensorType.INDICES

    def __init__(self, deterministic: bool = True, **default_params: Any):
        super().__init__(deterministic=deterministic, **default_params)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_type = self._validate_input_type(tensor)

        deterministic = self.default_params.get("deterministic", True)

        min_p = self._get_param("min_p", kwargs, required=True)
        maybe_min_p_arr, min_p_val = _to_tensor_scalar_tuple(min_p)

        if maybe_min_p_arr is None and not (
            0 < min_p_val <= 1 and isinstance(min_p_val, float)
        ):
            raise ValueError("min_p must be float in (0, 1] or a tensor array")

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().min_p_sampling_from_probs(
            tensor.data, indices, maybe_min_p_arr, min_p_val, deterministic, generator
        )

        return TaggedTensor(samples, output_type)


class FusedProbsTopKTopPSampleOp(ParameterizedOp):
    """Fused TopK -> TopP -> Sample: Probs -> Indices"""

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

        if maybe_top_p_arr is None and not (
            0 < top_p_val <= 1 and isinstance(top_p_val, float)
        ):
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
