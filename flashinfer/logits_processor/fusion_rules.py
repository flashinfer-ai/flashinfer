from typing import Callable, List, NamedTuple, Tuple

from .op import Op
from .operators import (
    FusedJointTopKTopPSampleProbsOp,
    FusedMinPSampleProbsOp,
    FusedTopKSampleProbsOp,
    FusedTopPSampleProbsOp,
    MinPOp,
    ProbsTopKOp,
    SampleProbsOp,
    TopPOp,
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


def joint_topk_topp_sampleprobs_guard(window: List[Op]) -> bool:
    """
    Logits -> Indices
    Only fuse when joint_topk_topp is True
    """
    topk_op = window[0]
    joint_topk_topp = getattr(topk_op, "default_params", {}).get(
        "joint_topk_topp", False
    )
    return joint_topk_topp


def build_topk_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedTopKSampleProbsOp(deterministic=deterministic)


def build_topp_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedTopPSampleProbsOp(deterministic=deterministic)


def build_minp_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedMinPSampleProbsOp(deterministic=deterministic)


def get_default_fusion_rules() -> List[FusionRule]:
    return [
        FusionRule(
            pattern=(ProbsTopKOp, TopPOp, SampleProbsOp),
            guard=joint_topk_topp_sampleprobs_guard,
            build=lambda window: FusedJointTopKTopPSampleProbsOp(),
            prio=100,
        ),
        FusionRule(
            pattern=(ProbsTopKOp, SampleProbsOp),
            guard=lambda window: True,
            build=build_topk_sampling,
            prio=10,
        ),
        FusionRule(
            pattern=(TopPOp, SampleProbsOp),
            guard=lambda window: True,
            build=build_topp_sampling,
            prio=10,
        ),
        FusionRule(
            pattern=(MinPOp, SampleProbsOp),
            guard=lambda window: True,
            build=build_minp_sampling,
            prio=10,
        ),
    ]
