from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch

from flashinfer.sampling import _to_tensor_scalar_tuple, get_sampling_module


class ProcessorType(Enum):
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    MIN_P = "min_p"
    SOFTMAX = "softmax"
    SAMPLING = "sampling"


class BaseProcessor(ABC):
    def __init__(self, processor_type: ProcessorType):
        self.processor_type = processor_type
    
    @abstractmethod
    def validate_inputs(self, batch_size: int, vocab_size: int) -> bool:
        pass

    @abstractmethod
    def __call__(self, tensor: torch.Tensor, is_logits: bool, **kwargs) -> Tuple[torch.Tensor, bool]:
        pass

class Temperature(BaseProcessor):
    def __init__(self, temperature: Union[float, torch.Tensor]):
        super().__init__(ProcessorType.TEMPERATURE)
        self.temperature = temperature
    
    def validate_inputs(self, batch_size: int, vocab_size: int) -> bool:
        if isinstance(self.temperature, torch.Tensor):
            return self.temperature.size(0) == batch_size
        return self.temperature > 0
    
    def __call__(self, tensor: torch.Tensor, is_logits: bool, **kwargs) -> Tuple[torch.Tensor, bool]:
        if not is_logits:
            raise ValueError("Temperature can only be applied to logits")
        
        if isinstance(self.temperature, torch.Tensor):
            temp = self.temperature.view(-1, 1)  # (batch_size, 1)
            result = tensor / temp
        else:
            result = tensor / self.temperature
            
        return result, True

class TopK(BaseProcessor):
    def __init__(self, k: Union[int, torch.Tensor]):
        super().__init__(ProcessorType.TOP_K)
        self.k = k
    
    def validate_inputs(self, batch_size: int, vocab_size: int) -> bool:
        if isinstance(self.k, torch.Tensor):
            return self.k.size(0) == batch_size and torch.all(self.k <= vocab_size)
        return 0 < self.k <= vocab_size

    def __call__(self, tensor: torch.Tensor, is_logits: bool, **kwargs) -> Tuple[torch.Tensor, bool]:
        if is_logits:
            masked_logits = get_sampling_module().top_k_mask_logits(
                tensor, *_to_tensor_scalar_tuple(self.k)
            )
            return masked_logits, True
        else:    
            renorm_probs = get_sampling_module().top_k_renorm_probs(
                tensor, *_to_tensor_scalar_tuple(self.k)
            )
            return renorm_probs, False


class TopP(BaseProcessor):
    def __init__(self, p: Union[float, torch.Tensor]):
        super().__init__(ProcessorType.TOP_P)
        self.p = p
    
    def validate_inputs(self, batch_size: int, vocab_size: int) -> bool:
        if isinstance(self.p, torch.Tensor):
            return (self.p.size(0) == batch_size and 
                   torch.all(self.p > 0) and torch.all(self.p <= 1))
        return 0 < self.p <= 1

    def __call__(self, tensor: torch.Tensor, is_logits: bool, **kwargs) -> Tuple[torch.Tensor, bool]:
        if is_logits:   
            masked_logits = get_sampling_module().top_p_mask_logits(
                tensor, *_to_tensor_scalar_tuple(self.p)
            )
            return masked_logits, True
        else:
            renorm_probs = get_sampling_module().top_p_renorm_probs(
                tensor, *_to_tensor_scalar_tuple(self.p)
            )
            return renorm_probs, False

class MinP(BaseProcessor):
    def __init__(self, min_p: Union[float, torch.Tensor]):
        super().__init__(ProcessorType.MIN_P)
        self.min_p = min_p
    
    def validate_inputs(self, batch_size: int, vocab_size: int) -> bool:
        if isinstance(self.min_p, torch.Tensor):
            return (self.min_p.size(0) == batch_size and 
                   torch.all(self.min_p > 0) and torch.all(self.min_p <= 1))
        return 0 < self.min_p <= 1

    def __call__(self, tensor: torch.Tensor, is_logits: bool, **kwargs) -> Tuple[torch.Tensor, bool]:
        if is_logits:
            masked_logits = get_sampling_module().min_p_mask_logits(
                tensor, *_to_tensor_scalar_tuple(self.min_p)
            )
            return masked_logits, True
        else:
            renorm_probs = get_sampling_module().min_p_renorm_probs(
                tensor, *_to_tensor_scalar_tuple(self.min_p)
            )
            return renorm_probs, False


class Softmax(BaseProcessor):
    def __init__(self):
        super().__init__(ProcessorType.SOFTMAX)
    
    def validate_inputs(self, batch_size: int, vocab_size: int) -> bool:
        return True
    
    def __call__(self, tensor: torch.Tensor, is_logits: bool, **kwargs) -> Tuple[torch.Tensor, bool]:
        if is_logits:
            return torch.softmax(tensor, dim=-1), True
        else:
            return tensor, False


class Sampling(BaseProcessor):
    def __init__(self, deterministic: bool = True):
        super().__init__(ProcessorType.SAMPLING)
        self.deterministic = deterministic
    
    def validate_inputs(self, batch_size: int, vocab_size: int) -> bool:
        return True
    
    def __call__(self, tensor: torch.Tensor, is_logits: bool, **kwargs) -> Tuple[torch.Tensor, bool]:
        if is_logits:
            samples = get_sampling_module().sampling_from_logits(
                tensor,
                kwargs.get('indices'),
                self.deterministic,
                kwargs.get('generator')
            )
        else:
            samples = get_sampling_module().sampling_from_probs(
                tensor,
                kwargs.get('indices'),
                self.deterministic,
                kwargs.get('generator')
            )
        
        return samples, False
    
class SamplingPlan:    
    def __init__(self, processors: List[BaseProcessor], batch_size: int, vocab_size: int):
        self.processors = processors
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self._validate_plan()
    
    def _validate_plan(self):
        # we can verify the list of logits processors, and maybe search for how to generate fused kernel
        for processor in self.processors:
            if not processor.validate_inputs(self.batch_size, self.vocab_size):
                raise ValueError(f"Invalid processor configuration: {processor}")
        
        # Check that Sampling is last if present
        sampling_indices = [i for i, p in enumerate(self.processors) 
                          if p.processor_type == ProcessorType.SAMPLING]
        if sampling_indices and sampling_indices[-1] != len(self.processors) - 1:
            raise ValueError("Sampling processor must be last in the pipeline")

class LogitsProcessor:
    def __init__(self, processors: List[BaseProcessor]):
        if not processors:
            raise ValueError("Processor list cannot be empty")
            
        self.processors = processors
        self._plan_cache = {}
        
    def plan(self, batch_size: int, vocab_size: int) -> SamplingPlan:
        cache_key = (batch_size, vocab_size, tuple(id(p) for p in self.processors))
        
        if cache_key not in self._plan_cache:
            plan = SamplingPlan(self.processors, batch_size, vocab_size)
            self._plan_cache[cache_key] = plan
            
        return self._plan_cache[cache_key]
    
    def __call__(self, 
                 logits: torch.Tensor,
                 indices: Optional[torch.Tensor] = None,
                 generator: Optional[torch.Generator] = None,
                 plan: Optional[SamplingPlan] = None) -> torch.Tensor:        
        batch_size, vocab_size = logits.shape
        
        if plan is None:
            plan = self.plan(batch_size, vocab_size)
        
        kwargs = {
            "indices": indices,
            "generator": generator
        }

        current_tensor = logits
        current_is_logits = True
        for processor in plan.processors:
            if processor.processor_type in ProcessorType:
                current_tensor, current_is_logits = processor(
                    current_tensor, current_is_logits, **kwargs
                )
            else:
                raise ValueError(f"Unknown processor type: {type(processor)}")
            
        return current_tensor
    

def demo_run_logits_processors():
    logits = torch.randn(1, 18, device='cuda')
    print(logits)
    
    processor = LogitsProcessor([
        Temperature(1.2),
        TopK(3),
        TopP(0.9),
        Softmax(),
        Sampling(deterministic=True)
    ])
    
    samples = processor(logits)
    print(samples)

if __name__ == "__main__":
    demo_run_logits_processors()