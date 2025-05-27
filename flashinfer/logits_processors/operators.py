import torch
from typing import Any

from .types import Sort, TaggedTensor
from .base import Op, ParameterizedOp

from flashinfer.sampling import get_sampling_module


class Temperature(ParameterizedOp):
    """Temperature scaling operator: Logits → Logits"""
    
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
    """Softmax operator: Logits → Probabilities"""
    
    IN = [Sort.LOGITS]
    OUT = [Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        probs = torch.softmax(tensor.data, dim=-1)
        
        return TaggedTensor(probs, output_sort)


class TopK(ParameterizedOp):
    """TopK operator: Logits → Logits | Probabilities → Probabilities"""
    
    IN = [Sort.LOGITS, Sort.PROBS]
    OUT = [Sort.LOGITS, Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        top_k = self._get_param("top_k", kwargs)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
                
        if tensor.sort == Sort.LOGITS:
            masked_logits = get_sampling_module().top_k_mask_logits(
                tensor.data, None, top_k
            )
            return TaggedTensor(masked_logits, output_sort)
        else:  # Sort.PROBS
            renorm_probs = get_sampling_module().top_k_renorm_probs(
                tensor.data, None, top_k
            )
            return TaggedTensor(renorm_probs, output_sort)


class TopP(ParameterizedOp):
    """TopP operator: Probabilities → Probabilities"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        top_p = self._get_param("top_p", kwargs)
        if not (0 < top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
                
        renorm_probs = get_sampling_module().top_p_renorm_probs(
            tensor.data, None, top_p
        )
        
        return TaggedTensor(renorm_probs, output_sort)


class MinP(ParameterizedOp):
    """MinP operator: Logits → Logits | Probabilities → Probabilities"""
    
    IN = [Sort.LOGITS, Sort.PROBS]
    OUT = [Sort.LOGITS, Sort.PROBS]
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        min_p = self._get_param("min_p", kwargs)
        if not (0 < min_p <= 1):
            raise ValueError("min_p must be in (0, 1]")
        
        if tensor.sort == Sort.LOGITS:
            masked_logits = get_sampling_module().min_p_mask_logits(
                tensor.data, None, min_p
            )
            return TaggedTensor(masked_logits, output_sort)
        else:  # Sort.PROBS
            renorm_probs = get_sampling_module().min_p_renorm_probs(
                tensor.data, None, min_p
            )
            return TaggedTensor(renorm_probs, output_sort)


class Sampling(ParameterizedOp):
    """Sampling operator: Logits | Probabilities → Indices"""
    
    IN = [Sort.LOGITS, Sort.PROBS]
    OUT = [Sort.INDICES, Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        deterministic = self._get_param("deterministic", kwargs, required=False)
        if deterministic is None:
            deterministic = self.default_params.get("deterministic", True)
        
        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)
        
        if tensor.sort == Sort.LOGITS:
            samples = get_sampling_module().sampling_from_logits(
                tensor.data, indices, deterministic, generator
            )
        else:  # Sort.PROBS
            samples = get_sampling_module().sampling_from_probs(
                tensor.data, indices, deterministic, generator
            )
        
        return TaggedTensor(samples, output_sort)


class FusedTopKSamplingWithProbs(ParameterizedOp):
    """Fused TopK + Sampling from probabilities: Probabilities → Indices"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        top_k = self._get_param("top_k", kwargs)
        deterministic = self._get_param("deterministic", kwargs, required=False)
        if deterministic is None:
            deterministic = self.default_params.get("deterministic", True)
        
        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)

        samples = get_sampling_module().top_k_sampling_from_probs(
            tensor.data, indices, None, top_k, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort)


class FusedTopPSamplingWithProbs(ParameterizedOp):
    """Fused TopP + Sampling from probabilities: Probabilities → Indices"""
    
    IN = [Sort.PROBS]
    OUT = [Sort.INDICES]
    
    def __init__(self, deterministic: bool = True):
        super().__init__(deterministic=deterministic)
    
    def __call__(self, tensor: TaggedTensor, **kwargs: Any) -> TaggedTensor:
        output_sort = self._validate_input_sort(tensor)
        
        top_p = self._get_param("top_p", kwargs)
        deterministic = self._get_param("deterministic", kwargs, required=False)
        if deterministic is None:
            deterministic = self.default_params.get("deterministic", True)
        
        indices = self._get_param("indices", kwargs, required=False)
        generator = self._get_param("generator", kwargs, required=False)
        
        samples = get_sampling_module().top_p_sampling_from_probs(
            tensor.data, indices, None, top_p, deterministic, generator
        )
        
        return TaggedTensor(samples, output_sort) 