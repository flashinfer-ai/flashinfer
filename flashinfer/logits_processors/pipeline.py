from typing import List, Union, Any
import torch

from .op import Op
from .types import TaggedTensor
from .compiler import compile_pipeline
from .validators import CompileError


class Pipe:
    def __init__(self, ops: List[Op], compile: bool = True):
        if not ops:
            raise ValueError("Pipeline cannot be empty")
        
        self.original_ops = list(ops)
        
        if compile:
            try:
                self.ops = compile_pipeline(ops)
            except CompileError as e:
                raise ValueError(f"Pipeline compilation failed: {e}") from e
        else:
            self.ops = list(ops)
    
    def __call__(self, 
                 x: Union[torch.Tensor, TaggedTensor], 
                 **kwargs: Any) -> torch.Tensor:
        if isinstance(x, TaggedTensor):
            tagged_tensor = x
        else:
            tagged_tensor = TaggedTensor.logits(x)
        
        runtime_kwargs = dict(kwargs)
        
        for i, op in enumerate(self.ops):
            try:
                tagged_tensor = op(tagged_tensor, **runtime_kwargs)
            except Exception as e:
                raise ValueError(
                    f"Error executing operator {i} ({op.__class__.__name__}): {e}"
                ) from e
            
        if runtime_kwargs:
            unused_keys = list(runtime_kwargs.keys())
            print(f"WARN: Unused kwargs: {unused_keys}")
        
        return tagged_tensor.data

    def compile(self, ops: List[Op]) -> None:
        try:
            self.ops = compile_pipeline(ops)
        except CompileError as e:
            raise ValueError(f"Pipeline compilation failed: {e}") from e