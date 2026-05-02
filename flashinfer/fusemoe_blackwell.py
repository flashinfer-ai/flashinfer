# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contributed FuseMoE Blackwell FP8 block-scale MoE kernel.

A hand-tuned monolithic kernel for FP8 block-scale MoE with DeepSeek-V3
routing on Blackwell SM100. The shape constants (hidden, intermediate,
num_experts, num_local_experts, top_k, n_group, topk_group) are JIT
parameters — each unique shape produces its own compiled module, cached
on disk via the standard flashinfer JIT URI.

The kernel was hand-tuned on ``DSV3_EP8_SHAPE`` (DeepSeek-V3 with 8-way
expert parallelism) and that is the only shape we have verified for both
correctness and performance. Other shapes that satisfy the divisibility
``static_assert``s in ``csrc/fusemoe_blackwell_config.jinja`` will compile
and run, but pass ``experimental_shape=True`` to acknowledge that you
are responsible for verifying correctness on your shape.
"""

import functools
from typing import FrozenSet, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.fusemoe_blackwell import gen_fusemoe_blackwell_module
from .trace.templates.fusemoe_blackwell import fusemoe_blackwell_fp8_dsv3_trace
from .utils import backend_requirement, supported_compute_capability

# Shape tuple ordering used everywhere in this file:
#   (hidden, intermediate, num_experts_global, num_local_experts,
#    top_k, n_group, topk_group).
ShapeKey = Tuple[int, int, int, int, int, int, int]

DSV3_EP8_SHAPE: ShapeKey = (7168, 2048, 256, 32, 8, 8, 4)

# Shapes the kernel author has tested. Anything else needs
# ``experimental_shape=True`` to acknowledge unverified output.
VERIFIED_SHAPES: FrozenSet[ShapeKey] = frozenset({DSV3_EP8_SHAPE})


@functools.cache
def _get_module(shape: ShapeKey):
    h, i, e, le, tk, ng, kg = shape
    return gen_fusemoe_blackwell_module(
        hidden_size=h,
        intermediate_size=i,
        num_experts_global=e,
        num_local_experts=le,
        top_k=tk,
        n_group=ng,
        topk_group=kg,
    ).build_and_load()


def _infer_shape(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
) -> ShapeKey:
    return (
        int(hidden_states.shape[1]),  # hidden
        int(gemm1_weights.shape[1] // 2),  # intermediate (gemm1 emits 2*I for SwiGLU)
        int(routing_logits.shape[1]),  # num_experts_global
        int(gemm1_weights.shape[0]),  # num_local_experts
        int(top_k),
        int(n_group),
        int(topk_group),
    )


@supported_compute_capability([100])
def _check_fusemoe_blackwell_problem_size(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    local_expert_offset: int = 0,
    routed_scaling_factor: float = 2.5,
    experimental_shape: bool = False,
) -> bool:
    if routing_logits.dim() != 2 or routing_logits.dtype != torch.float32:
        raise ValueError(
            "routing_logits must be 2-D float32, got "
            f"shape={tuple(routing_logits.shape)} dtype={routing_logits.dtype}."
        )
    if not hidden_states.is_cuda:
        raise ValueError("hidden_states must be a CUDA tensor.")
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"hidden_states must be fp8_e4m3fn, got {hidden_states.dtype}."
        )

    shape = _infer_shape(
        routing_logits, hidden_states, gemm1_weights, top_k, n_group, topk_group
    )
    h, i, e, le, tk, ng, kg = shape
    num_tokens = routing_logits.shape[0]

    # Layout / dtype / consistency checks across the eight tensors.
    if routing_bias.shape != (e,) or routing_bias.dtype != torch.bfloat16:
        raise ValueError(
            f"routing_bias must be bfloat16 [{e}], got "
            f"dtype={routing_bias.dtype} shape={tuple(routing_bias.shape)}."
        )
    if hidden_states.shape != (num_tokens, h):
        raise ValueError(
            f"hidden_states shape mismatch: expected ({num_tokens}, {h}), "
            f"got {tuple(hidden_states.shape)}."
        )
    if hidden_states_scale.shape != (h // 128, num_tokens):
        raise ValueError(
            f"hidden_states_scale must be [{h // 128}, num_tokens], got "
            f"{tuple(hidden_states_scale.shape)}."
        )
    if (
        gemm1_weights.shape != (le, 2 * i, h)
        or gemm1_weights.dtype != torch.float8_e4m3fn
    ):
        raise ValueError(
            f"gemm1_weights must be fp8_e4m3 [{le}, {2 * i}, {h}], got "
            f"dtype={gemm1_weights.dtype} shape={tuple(gemm1_weights.shape)}."
        )
    if gemm1_weights_scale.shape != (le, (2 * i) // 128, h // 128):
        raise ValueError(
            f"gemm1_weights_scale must be [{le}, {(2 * i) // 128}, {h // 128}], got "
            f"{tuple(gemm1_weights_scale.shape)}."
        )
    if gemm2_weights.shape != (le, h, i) or gemm2_weights.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"gemm2_weights must be fp8_e4m3 [{le}, {h}, {i}], got "
            f"dtype={gemm2_weights.dtype} shape={tuple(gemm2_weights.shape)}."
        )
    if gemm2_weights_scale.shape != (le, h // 128, i // 128):
        raise ValueError(
            f"gemm2_weights_scale must be [{le}, {h // 128}, {i // 128}], got "
            f"{tuple(gemm2_weights_scale.shape)}."
        )

    if not (0 <= int(local_expert_offset) <= e - le):
        raise ValueError(
            f"local_expert_offset must be in [0, {e - le}], got {local_expert_offset}."
        )

    # Divisibility constraints that the kernel relies on (kernel-level
    # static_asserts repeat these at compile time, but raising here gives
    # a Python-level error before nvcc is even invoked).
    if h % 128 != 0:
        raise ValueError(f"hidden_size ({h}) must be divisible by 128.")
    if i % 256 != 0 or i < 512:
        raise ValueError(
            f"intermediate_size ({i}) must be a multiple of 256 and >= 512."
        )
    if e % ng != 0:
        raise ValueError(
            f"num_experts_global ({e}) must be divisible by n_group ({ng})."
        )
    if e // ng > 32:
        raise ValueError(f"experts-per-group ({e // ng}) must fit in one warp (<=32).")
    if tk > 32:
        raise ValueError(f"top_k ({tk}) must be <=32.")
    if kg > ng:
        raise ValueError(f"topk_group ({kg}) cannot exceed n_group ({ng}).")

    # Verified-shape gate.
    if shape not in VERIFIED_SHAPES and not experimental_shape:
        raise ValueError(
            f"Shape {shape} (hidden, intermediate, num_experts, "
            "num_local_experts, top_k, n_group, topk_group) is not in the "
            "verified-shape allowlist. The kernel was hand-tuned for "
            f"{DSV3_EP8_SHAPE} and other shapes have not been validated for "
            "correctness or performance. Pass experimental_shape=True to opt "
            "in (and please verify your output against a reference)."
        )

    return True


@flashinfer_api(trace=fusemoe_blackwell_fp8_dsv3_trace)
@backend_requirement(
    backend_checks={},
    common_check=_check_fusemoe_blackwell_problem_size,
)
def fusemoe_blackwell_fp8_dsv3(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    local_expert_offset: int = 0,
    routed_scaling_factor: float = 2.5,
    experimental_shape: bool = False,
) -> torch.Tensor:
    """Run the contributed FP8 block-scale MoE kernel with DeepSeek-V3 routing.

    The shape (``hidden``, ``intermediate``, ``num_experts``,
    ``num_local_experts``) is inferred from the input tensors. The routing
    knobs (``top_k``, ``n_group``, ``topk_group``) are passed explicitly.

    The kernel was hand-tuned for the DeepSeek-V3 EP=8 shape
    ``(hidden=7168, intermediate=2048, num_experts=256, num_local_experts=32,
    top_k=8, n_group=8, topk_group=4)``. Other shapes need
    ``experimental_shape=True`` and your own correctness verification.

    Parameters
    ----------
    routing_logits : ``[num_tokens, num_experts]`` ``float32``
    routing_bias : ``[num_experts]`` ``bfloat16``
    hidden_states : ``[num_tokens, hidden]`` ``fp8_e4m3``
    hidden_states_scale : ``[hidden // 128, num_tokens]`` ``float32``
    gemm1_weights : ``[num_local_experts, 2 * intermediate, hidden]`` ``fp8_e4m3``
    gemm1_weights_scale : ``[num_local_experts, (2*intermediate)//128, hidden//128]`` ``float32``
    gemm2_weights : ``[num_local_experts, hidden, intermediate]`` ``fp8_e4m3``
    gemm2_weights_scale : ``[num_local_experts, hidden//128, intermediate//128]`` ``float32``
    top_k, n_group, topk_group : DSV3 routing knobs.
    local_expert_offset : index of the first local expert in the global expert
        space. ``[0, num_experts - num_local_experts]``.
    routed_scaling_factor : DSV3 routed scaling factor.
    experimental_shape : opt in for shapes not in ``VERIFIED_SHAPES``.

    Returns
    -------
    ``[num_tokens, hidden]`` ``bfloat16`` tensor.
    """
    shape = _infer_shape(
        routing_logits, hidden_states, gemm1_weights, top_k, n_group, topk_group
    )
    return _get_module(shape).run(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        int(local_expert_offset),
        float(routed_scaling_factor),
    )
