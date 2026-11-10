"""Framework-facing collective token planning, including an entirely empty EP group.

Counts must be identical, already-host metadata agreed by every rank in the EP
collective. No GPU-to-host route read or collective is performed here. A rank
with zero local tokens MUST participate if any peer has work. Active plans use
a common padded capacity; callers zero route weights for padded rows and slice
outputs to local_output_tokens. The low-level kernel API remains positive-size.
"""
from dataclasses import dataclass,replace
from typing import Optional,Tuple
from .api import MegaMoEProblemSpec,MegaMoECompileSpec,select_compile_spec

@dataclass(frozen=True)
class CollectiveExecutionPlan:
    problem: MegaMoEProblemSpec
    ep_token_counts: Tuple[int,...]
    compile_spec: Optional[MegaMoECompileSpec]

    @property
    def is_empty(self):return self.compile_spec is None

    @property
    def local_output_tokens(self):return self.ep_token_counts[self.problem.expert_parallel_rank]

    @property
    def requires_collective_launch(self):return not self.is_empty

    def make_empty_output(self,*,device,dtype=None):
        if not self.is_empty:raise ValueError('A locally empty rank must still join nonempty EP work')
        import torch
        return torch.empty((0,self.problem.hidden),device=device,dtype=dtype or torch.bfloat16)

def select_collective_execution_plan(*,problem: MegaMoEProblemSpec,ep_token_counts: Tuple[int,...],**selection_options):
    if type(problem.tokens_per_rank) is not int or problem.tokens_per_rank<0:
        raise ValueError('tokens_per_rank must be a nonnegative integer')
    # Validate all other problem fields even for an empty collective.
    replace(problem,tokens_per_rank=max(1,problem.tokens_per_rank)).validate()
    if not isinstance(ep_token_counts,tuple) or len(ep_token_counts)!=problem.expert_parallel_size:
        raise ValueError('Provide one agreed host token count per EP rank')
    if any(type(x) is not int or x<0 for x in ep_token_counts):
        raise ValueError('EP token counts must be nonnegative integers')
    if selection_options.get('numerical_contract','ordinary_topk') not in ('ordinary_topk','owner_bf16'):
        raise ValueError('Unknown numerical contract')
    maximum=max(ep_token_counts)
    if maximum==0:return CollectiveExecutionPlan(problem,ep_token_counts,None)
    if problem.tokens_per_rank<maximum:
        raise ValueError('Common compile capacity must cover every EP rank')
    spec=select_compile_spec(problem=problem,**selection_options)
    return CollectiveExecutionPlan(problem,ep_token_counts,spec)
