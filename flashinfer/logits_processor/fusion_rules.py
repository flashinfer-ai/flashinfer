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

from typing import Callable, List, NamedTuple, Tuple

from .op import Op
from .operators import (
    FusedProbsMinPSampleOp,
    FusedProbsTopKSampleOp,
    FusedProbsTopKTopPSampleOp,
    FusedProbsTopPSampleOp,
    FusedTemperatureSoftmaxOp,
    MinPOp,
    ProbsSampleOp,
    ProbsTopKOp,
    SoftmaxOp,
    TemperatureOp,
    TopPOp,
)


class FusionRule(NamedTuple):
    """
    Attributes:
        pattern: Tuple of operator types to match (e.g., (TopK, Sampling))
        guard: Function that takes the matched operators and returns True if fusion should apply. It accepts a window of operators parameter which is a subset of the operators in the pipeline and is an exact match of the pattern.
        build: Function that takes the matched operators and returns a fused operator
        prio: Priority for rule application (higher = tried earlier)
    """

    pattern: Tuple[type, ...]
    guard: Callable[[List[Op]], bool]
    build: Callable[[List[Op]], Op]
    prio: int = 0


def joint_topk_topp_sampleprobs_guard(window: List[Op]) -> bool:
    """
    Only fuse when joint_topk_topp is True
    """
    topk_op = window[0]
    joint_topk_topp = getattr(topk_op, "default_params", {}).get(
        "joint_topk_topp", False
    )
    return joint_topk_topp


def build_temperature_softmax(window: List[Op]) -> Op:
    softmax_op = window[1]
    enable_pdl = getattr(softmax_op, "default_params", {}).get("enable_pdl", None)
    return FusedTemperatureSoftmaxOp(enable_pdl=enable_pdl)


def build_topk_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedProbsTopKSampleOp(deterministic=deterministic)


def build_topp_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedProbsTopPSampleOp(deterministic=deterministic)


def build_minp_sampling(window: List[Op]) -> Op:
    sampling_op = window[1]
    deterministic = getattr(sampling_op, "default_params", {}).get(
        "deterministic", True
    )
    return FusedProbsMinPSampleOp(deterministic=deterministic)


def get_default_fusion_rules() -> List[FusionRule]:
    return [
        FusionRule(
            pattern=(TemperatureOp, SoftmaxOp),
            guard=lambda window: True,
            build=build_temperature_softmax,
            prio=100,
        ),
        FusionRule(
            pattern=(ProbsTopKOp, TopPOp, ProbsSampleOp),
            guard=joint_topk_topp_sampleprobs_guard,
            build=lambda window: FusedProbsTopKTopPSampleOp(),
            prio=100,
        ),
        FusionRule(
            pattern=(ProbsTopKOp, ProbsSampleOp),
            guard=lambda window: True,
            build=build_topk_sampling,
            prio=10,
        ),
        FusionRule(
            pattern=(TopPOp, ProbsSampleOp),
            guard=lambda window: True,
            build=build_topp_sampling,
            prio=10,
        ),
        FusionRule(
            pattern=(MinPOp, ProbsSampleOp),
            guard=lambda window: True,
            build=build_minp_sampling,
            prio=10,
        ),
    ]
