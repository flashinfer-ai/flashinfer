"""
Copyright (c) 2026 by FlashInfer team.

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

# Experimental Cake backend: the StepFun fused MoE (``SwiGLUStep`` with a
# per-expert clamp limit) on exact SM100 / SM103 as generated cake-build/1
# programs -- routing (Renormalize top-k), FC1 with the StepFun epilogue, the
# NVFP4 per-token requantization, FC2 and finalize -- orchestrated from Python.
#
# The one-shot helpers below are the public opt-in of the package (they warn
# once as an experimental API): ``stepfun_moe`` runs the whole forward, the
# five stage helpers run one exported stage on caller-owned buffers.  The
# prepare-once / launch-many surface (``prepare_stepfun_moe`` returning a
# ``StepFunMoeLaunch``, ``prepare_<stage>`` returning a ``StepFunStageLaunch``)
# and the host plan (``Plan``, ``plans_for``, ``default_plan``) live in
# ``cake_backend`` and are re-exported lazily; the JIT registry filled by the
# generated-program export lives in ``cake_jit``.  All stay unimported until
# first use so importing this package is cheap.  The stable entry point is
# ``flashinfer.fused_moe`` with ``CakeStepFunConfig(backend="cake_stepfun")``;
# nothing in this package imports ``flashinfer.fused_moe`` and nothing in it
# reads the environment.

from __future__ import annotations

from typing import Any, Optional

import torch

from ...api_logging import flashinfer_experimental_api

__all__ = [
    "Plan",
    "StepFunMoeLaunch",
    "StepFunStageLaunch",
    "default_plan",
    "fc1",
    "fc2",
    "finalize",
    "plans_for",
    "prepare_stepfun_moe",
    "requant",
    "routing",
    "stepfun_moe",
]

_BACKEND_EXPORTS = frozenset(
    {
        "Plan",
        "StepFunMoeLaunch",
        "StepFunStageLaunch",
        "default_plan",
        "plans_for",
        "prepare_stepfun_moe",
    }
)


@flashinfer_experimental_api(feature="cake_stepfun_moe.stepfun_moe")
def stepfun_moe(
    *,
    family: str,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor] = None,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: Optional[torch.Tensor] = None,
    output1_scale_scalar: Optional[torch.Tensor] = None,
    output1_scale_gate_scalar: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: torch.Tensor,
    per_token_scale: Optional[torch.Tensor] = None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: Optional[torch.Tensor] = None,
    output2_scale_scalar: Optional[torch.Tensor] = None,
    routing_logits: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    local_num_experts: int,
    plan: Any = None,
    output: Optional[torch.Tensor] = None,
    do_finalize: bool = True,
    enable_pdl: bool = True,
    reserved_sms: int = 8,
    global_scale_inv: float = 1.0 / (448.0 * 6.0),
    poison_routing_tail: bool = False,
) -> list[torch.Tensor]:
    """One StepFun fused-MoE forward on the exported Cake kernels.

    Prepares and launches in one call: routing -> FC1 -> (requantization) ->
    FC2 -> finalize on the current torch stream.  ``family`` is one of
    ``nvfp4``, ``nvfp4_bf16tok``, ``bf16``, ``fp8``, ``mxfp8``; the tensors
    are the trtllm-gen weight / activation views of that family (see the
    package README), ``gemm1_clamp_limit`` the per-expert fp32 StepFun limits
    in the units the kernels clamp, and exactly one of ``routing_logits`` (bf16
    or fp32 ``[num_tokens, num_experts]``) or ``topk_ids`` + ``topk_weights``
    (int32 / bf16-or-fp32 ``[num_tokens, top_k]``) selects the routing input.
    ``plan`` is ``None`` (the planner's centre tile), a tile or a ``Plan``.

    Returns ``[output]`` (bf16 ``[num_tokens, hidden_size]``) when
    ``do_finalize``, else ``[gemm2_output, expert_weights,
    expanded_idx_to_permuted_idx]`` in the trtllm-gen layout.  For repeated
    launches over the same tensors (or CUDA Graph capture) use
    ``prepare_stepfun_moe`` once and call ``launch()`` on the returned object.
    """
    from .cake_backend import prepare_stepfun_moe

    return prepare_stepfun_moe(
        family=family,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        gemm1_clamp_limit=gemm1_clamp_limit,
        per_token_scale=per_token_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        output2_scale_scalar=output2_scale_scalar,
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        plan=plan,
        output=output,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        reserved_sms=reserved_sms,
        global_scale_inv=global_scale_inv,
        poison_routing_tail=poison_routing_tail,
    ).launch()


@flashinfer_experimental_api(feature="cake_stepfun_moe.routing")
def routing(**kwargs: Any) -> None:
    """Run the exported Cake routing stage once on caller-owned tables
    (``cake_backend.prepare_routing(**kwargs).launch()``; see its keyword list)."""
    from .cake_backend import routing as run

    run(**kwargs)


@flashinfer_experimental_api(feature="cake_stepfun_moe.fc1")
def fc1(**kwargs: Any) -> None:
    """Run the exported Cake FC1 (StepFun epilogue) stage once on caller-owned
    buffers (``cake_backend.prepare_fc1(**kwargs).launch()``)."""
    from .cake_backend import fc1 as run

    run(**kwargs)


@flashinfer_experimental_api(feature="cake_stepfun_moe.requant")
def requant(**kwargs: Any) -> None:
    """Run the exported NVFP4 per-token requantization stage once on
    caller-owned buffers (``cake_backend.prepare_requant(**kwargs).launch()``)."""
    from .cake_backend import requant as run

    run(**kwargs)


@flashinfer_experimental_api(feature="cake_stepfun_moe.fc2")
def fc2(**kwargs: Any) -> None:
    """Run the exported Cake FC2 stage once on caller-owned buffers
    (``cake_backend.prepare_fc2(**kwargs).launch()``)."""
    from .cake_backend import fc2 as run

    run(**kwargs)


@flashinfer_experimental_api(feature="cake_stepfun_moe.finalize")
def finalize(**kwargs: Any) -> None:
    """Run the exported Cake finalize stage once on caller-owned buffers
    (``cake_backend.prepare_finalize(**kwargs).launch()``)."""
    from .cake_backend import finalize as run

    run(**kwargs)


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        from . import cake_backend

        return getattr(cake_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
