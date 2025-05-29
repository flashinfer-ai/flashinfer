from typing import Callable, List, NamedTuple, Tuple

from .op import Op
from .operators import (
    FusedJointTopKTopPSampling,
    FusedMinPSampling,
    FusedSoftmaxSampling,
    FusedSoftmaxTopKMaskLogits,
    FusedTopKSampling,
    FusedTopPSampling,
    MaskLogits,
    MinP,
    Sampling,
    Softmax,
    TopK,
    TopP,
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


def softmax_sampling_guard(window: List[Op]) -> bool:
    """
    Logits → Indices
    Allow any input for fusion since softmax only accepted LOGITS.
    """
    return True


def topk_sampling_guard(window: List[Op]) -> bool:
    """
    Probabilities → Indices
    Allow any input for fusion since topk only accepted PROBS.
    """
    return True


def topp_sampling_guard(window: List[Op]) -> bool:
    """
    Probabilities → Indices
    Allow any input for fusion since topp only accepted PROBS.
    """
    return True


def minp_sampling_guard(window: List[Op]) -> bool:
    """
    Probabilities → Indices
    Allow any input for fusion since minp only accepted PROBS.
    """
    return True


def joint_topk_topp_sampling_guard(window: List[Op]) -> bool:
    """
    Logits → Indices
    Only fuse when joint_topk_topp is True
    """
    topk_op = window[0]
    joint_topk_topp = getattr(topk_op, "default_params", {}).get("joint_topk_topp", False)
    return joint_topk_topp


def softmax_topk_masklogits_guard(window: List[Op]) -> bool:
    """
    Logits → Logits
    Allow any input for fusion since softmax only accepted LOGITS.
    """
    return True


def build_softmax_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedSoftmaxSampling(deterministic=deterministic)


def build_topk_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedTopKSampling(deterministic=deterministic)


def build_topp_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedTopPSampling(deterministic=deterministic)


def build_minp_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedMinPSampling(deterministic=deterministic)


def build_softmax_topk_masklogits(window: List[Op]) -> Op:
    return FusedSoftmaxTopKMaskLogits()


def build_joint_topk_topp_sampling(window: List[Op]) -> Op:
    return FusedJointTopKTopPSampling()


def get_default_fusion_rules() -> List[FusionRule]:
    return [
        FusionRule(
            pattern=(TopK, TopP, Sampling),
            guard=joint_topk_topp_sampling_guard,
            build=build_joint_topk_topp_sampling,
            prio=100,
        ),
        FusionRule(
            pattern=(Softmax, TopK, MaskLogits),
            guard=softmax_topk_masklogits_guard,
            build=build_softmax_topk_masklogits,
            prio=100,
        ),
        FusionRule(
            pattern=(Softmax, Sampling),
            guard=softmax_sampling_guard,
            build=build_softmax_sampling,
            prio=10,
        ),
        FusionRule(
            pattern=(TopK, Sampling),
            guard=topk_sampling_guard,
            build=build_topk_sampling,
            prio=10,
        ),
        FusionRule(
            pattern=(TopP, Sampling),
            guard=topp_sampling_guard,
            build=build_topp_sampling,
            prio=10,
        ),
        FusionRule(
            pattern=(MinP, Sampling),
            guard=minp_sampling_guard,
            build=build_minp_sampling,
            prio=10,
        ),
    ]
