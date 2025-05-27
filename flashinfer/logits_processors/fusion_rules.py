from typing import List, Callable, Tuple, NamedTuple

from .base import Op
from .types import Sort
from .operators import (
    TopK, TopP, Sampling, FusedTopKSamplingWithProbs, 
    FusedTopPSamplingWithProbs
)


class FusionRule(NamedTuple):
    """    
    Attributes:
        pattern: Tuple of operator types to match (e.g., (TopK, Sampling))
        guard: Function that takes the matched operators and returns True if fusion should apply
        build: Function that takes the matched operators and returns a fused operator
        prio: Priority for rule application (higher = tried earlier)
    """
    pattern: Tuple[type, ...]
    guard: Callable[[List[Op]], bool]
    build: Callable[[List[Op]], Op]
    prio: int = 0

def topk_sampling_probs_guard(window: List[Op]) -> bool:
    topk_op = window[0]
    return Sort.PROBS in topk_op.IN


def topp_sampling_guard(window: List[Op]) -> bool:
    return True


def build_topk_sampling_probs(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, 'default_params', {}).get('deterministic', True)
    return FusedTopKSamplingWithProbs(deterministic=deterministic)


def build_topp_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, 'default_params', {}).get('deterministic', True)
    return FusedTopPSamplingWithProbs(deterministic=deterministic)


def get_default_fusion_rules() -> List[FusionRule]:
    return [
        FusionRule(
            pattern=(TopK, Sampling),
            guard=topk_sampling_probs_guard,
            build=build_topk_sampling_probs,
            prio=10,
        ),
        FusionRule(
            pattern=(TopP, Sampling),
            guard=topp_sampling_guard,
            build=build_topp_sampling,
            prio=9,
        ),
    ]