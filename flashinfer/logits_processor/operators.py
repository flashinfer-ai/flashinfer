import torch
from typing import Any

from .types import Sort, TaggedTensor
from .op import Op, ParameterizedOp

from flashinfer.sampling import get_sampling_module


class Temperature(ParameterizedOp):
    """Temperature scaling: Logits → Logits"""
    
    IN = [Sort.LOGITS]
    OUT = [Sort.LOGITS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        temperature = self._get_param("temperature", kwargs)
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        scaled_logits = tensor.data / temperature
        
        return TaggedTensor(scaled_logits, output_sort)


class Softmax(Op):
    """Softmax: Logits → Probabilities"""
    
    IN = [Sort.LOGITS]
    OUT = [Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        probs = torch.softmax(tensor.data, dim=-1)
        
        return TaggedTensor(probs, output_sort)


class TopK(ParameterizedOp):
    """TopK: Probabilities → Probabilities"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        maybe_top_k_arr = self._get_param("maybe_top_k_arr", kwargs)
        top_k_val = self._get_param("top_k_val", kwargs)
        if (not isinstance(top_k_val, int) or top_k_val <= 0) and maybe_top_k_arr is None:
            raise ValueError("top_k must be a positive integer or maybe_top_k_arr must be provided")
                
        renorm_probs = get_sampling_module().top_k_renorm_probs(tensor.data, maybe_top_k_arr, top_k_val)
        
        return TaggedTensor(renorm_probs, output_sort)


class TopP(ParameterizedOp):
    """TopP: Probabilities → Probabilities"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        maybe_top_p_arr = self._get_param("maybe_top_p_arr", kwargs)
        top_p_val = self._get_param("top_p_val", kwargs)
        if not (0 < top_p_val <= 1) and maybe_top_p_arr is None:
            raise ValueError("top_p must be in (0, 1] or maybe_top_p_arr must be provided")
                
        renorm_probs = get_sampling_module().top_p_renorm_probs(tensor.data, maybe_top_p_arr, top_p_val)
        
        return TaggedTensor(renorm_probs, output_sort)


class MinP(ParameterizedOp):
    """MinP: Probabilities → Probabilities"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        min_p = self._get_param("min_p", kwargs)
        if not (0 < min_p <= 1):
            raise ValueError("min_p must be in (0, 1]")
        
        min_p_mask = tensor.data >= (min_p * tensor.data.max(dim=-1, keepdim=True)[0])
        tensor.data[min_p_mask] = 0
        probs = tensor.data / tensor.data.sum(dim=-1, keepdim=True)
        
        return TaggedTensor(probs, output_sort)

class MaskLogits(ParameterizedOp):
    """MaskLogits: Logits | Probabilities → Logits"""
    
    IN = [Sort.LOGITS, Sort.PROBS]
    OUT = [Sort.LOGITS, Sort.LOGITS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)

        # TODO(shanli): implement Mask Op. This class is kept temporarily for fusing SoftmaxTopKMaskLogits
        pass


class Sampling(ParameterizedOp):
    """Sampling: Probabilities → Indices"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        deterministic = self._get_param("deterministic", kwargs, required=False)
        if deterministic is None:
            deterministic = self.default_params.get("deterministic", True)

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)
        
        samples = get_sampling_module().sampling_from_probs(
            tensor.data, indices, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort)

class FusedSoftmaxSampling(ParameterizedOp):
    """Fused Softmax → Sampling: Logits → Indices
    Note that this operator is actually softmax-free,
    """
    
    IN = [Sort.LOGITS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        deterministic = (self._get_param("deterministic", kwargs, required=False) or
                         self.default_params.get("deterministic", True))
        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)
        
        samples = get_sampling_module().sampling_from_logits(
            tensor.data, indices, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort)

class FusedTopKSampling(ParameterizedOp):
    """Fused TopK → Sampling: Probabilities → Indices"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        deterministic = (self._get_param("deterministic", kwargs, required=False) or
                    self.default_params.get("deterministic", True))
        maybe_top_k_arr = self._get_param("maybe_top_k_arr", kwargs)
        top_k_val = self._get_param("top_k_val", kwargs)
        
        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().top_k_sampling_from_probs(
            tensor.data, indices, maybe_top_k_arr, top_k_val, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort)


class FusedTopPSampling(ParameterizedOp):
    """Fused TopP → Sampling: Probabilities → Indices"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)

        deterministic = (self._get_param("deterministic", kwargs, required=False) or
                         self.default_params.get("deterministic", True))
        maybe_top_p_arr = self._get_param("maybe_top_p_arr", kwargs)
        top_p_val = self._get_param("top_p_val", kwargs)
        
        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)
        
        samples = get_sampling_module().top_p_sampling_from_probs(
            tensor.data, indices, maybe_top_p_arr, top_p_val, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort) 
    
class FusedMinPSampling(ParameterizedOp):
    """Fused MinP → Sampling: Probabilities → Indices"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)
        
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)

        deterministic = (self._get_param("deterministic", kwargs, required=False) or
                         self.default_params.get("deterministic", True))
        maybe_min_p_arr = self._get_param("maybe_min_p_arr", kwargs)
        min_p_val = self._get_param("min_p_val", kwargs)

        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)
        
        samples = get_sampling_module().min_p_sampling_from_probs(
            tensor.data, indices, maybe_min_p_arr, min_p_val, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort)
    
class FusedJointTopKTopPSampling(ParameterizedOp):
    """Fused Joint TopK → TopP → Sampling: Probabilities → Indices"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)

        deterministic = (self._get_param("deterministic", kwargs, required=False) or
                         self.default_params.get("deterministic", True))
        maybe_top_k_arr = self._get_param("maybe_top_k_arr", kwargs)
        top_k_val = self._get_param("top_k_val", kwargs)
        maybe_top_p_arr = self._get_param("maybe_top_p_arr", kwargs)
        top_p_val = self._get_param("top_p_val", kwargs)
        
        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().top_k_top_p_sampling_from_probs(
            tensor.data, indices, maybe_top_k_arr, top_k_val, maybe_top_p_arr, top_p_val, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort)

class FusedSoftmaxTopKMaskLogits(ParameterizedOp):
    """Fused TopK → MaskLogits: Logits → Logits"""
    
    IN = [Sort.LOGITS]
    OUT = [Sort.LOGITS]
    
    def __init__(self):
        super().__init__()

    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        maybe_top_k_arr = self._get_param("maybe_top_k_arr", kwargs)
        top_k_val = self._get_param("top_k_val", kwargs)
        
        mask_logits = get_sampling_module().top_k_mask_logits(
            tensor.data, maybe_top_k_arr, top_k_val
        )
        
        return TaggedTensor(mask_logits, output_sort)