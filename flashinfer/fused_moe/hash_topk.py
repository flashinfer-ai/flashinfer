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

import functools
import math
from types import SimpleNamespace
from typing import Tuple

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit import gen_hash_topk_module
from flashinfer.trace.templates.moe import hash_topk_trace
from flashinfer.utils import (
    backend_requirement,
    device_support_pdl,
    register_custom_op,
    supported_compute_capability,
)

# hash_topk is plain CUDA (warp shuffle + table gather), portable across all
# tensor-core capable architectures.
_HASH_TOPK_SUPPORTED_CC = [80, 86, 89, 90, 100, 103, 110, 120, 121]


@supported_compute_capability(_HASH_TOPK_SUPPORTED_CC)
def _check_hash_topk_supported(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    launch_with_pdl: bool = True,
) -> bool:
    """Validate dtypes, shapes, device, and contiguity of hash_topk inputs.

    Mirrors the TVM-FFI-side checks so direct FFI callers and the Python API
    enforce the same contract. Returns ``True`` when all inputs are valid and
    raises ``ValueError`` otherwise.
    """
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must be 2D [num_tokens, num_experts], got {tuple(router_logits.shape)}"
        )
    if input_ids.dim() != 1:
        raise ValueError(f"input_ids must be 1D, got {tuple(input_ids.shape)}")
    if tid2eid.dim() != 2:
        raise ValueError(
            f"tid2eid must be 2D [vocab, topk], got {tuple(tid2eid.shape)}"
        )

    num_tokens = router_logits.shape[0]
    num_routed_experts = router_logits.shape[1]
    topk = tid2eid.shape[1]
    topk_fused = topk + num_fused_shared_experts

    if num_routed_experts < 1:
        raise ValueError(f"num_routed_experts must be >= 1, got {num_routed_experts}")
    if topk < 1:
        raise ValueError(f"topk (tid2eid.shape[1]) must be >= 1, got {topk}")
    if input_ids.shape[0] != num_tokens:
        raise ValueError(
            f"input_ids length ({input_ids.shape[0]}) must equal num_tokens ({num_tokens})"
        )
    if num_fused_shared_experts not in (0, 1):
        raise ValueError(
            f"num_fused_shared_experts must be 0 or 1, got {num_fused_shared_experts}"
        )
    if topk_fused > 32:
        raise ValueError(
            f"topk + num_fused_shared_experts ({topk_fused}) must be <= warp size 32"
        )
    if num_fused_shared_experts > 0 and (
        not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0.0
    ):
        raise ValueError(
            f"routed_scaling_factor must be positive and finite when a shared expert "
            f"is fused, got {routed_scaling_factor}"
        )
    if router_logits.dtype != torch.float32:
        raise ValueError(f"router_logits must be float32, got {router_logits.dtype}")
    if input_ids.dtype != torch.int64:
        raise ValueError(f"input_ids must be int64, got {input_ids.dtype}")
    if tid2eid.dtype != torch.int32:
        raise ValueError(f"tid2eid must be int32, got {tid2eid.dtype}")

    dev = router_logits.device
    for name, x in (
        ("input_ids", input_ids),
        ("tid2eid", tid2eid),
    ):
        if x.device != dev:
            raise ValueError(
                f"{name} must be on the same device as router_logits "
                f"({x.device} vs {dev})"
            )
    for name, x in (
        ("router_logits", router_logits),
        ("input_ids", input_ids),
        ("tid2eid", tid2eid),
    ):
        if not x.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return True


@functools.cache
def get_hash_topk_module():
    """Build, load, and cache the hash_topk JIT module as a custom op."""
    module = gen_hash_topk_module().build_and_load()

    @register_custom_op(
        "flashinfer::hash_topk",
        mutates_args=["topk_weights", "topk_ids"],
    )
    def hash_topk(
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        tid2eid: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routed_scaling_factor: float,
        launch_with_pdl: bool,
    ) -> None:
        """Custom-op wrapper that writes routing results into the output tensors."""
        module.hash_topk(
            router_logits,
            input_ids,
            tid2eid,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
            launch_with_pdl,
        )

    return SimpleNamespace(hash_topk=hash_topk)


@backend_requirement({}, common_check=_check_hash_topk_supported)
@flashinfer_api(trace=hash_topk_trace)
def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    launch_with_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Hash-based MoE expert routing for DeepSeek-V4.

    DSv4-Pro hash-MoE layers select experts from a precomputed token-to-expert
    table (``tid2eid``) instead of running a dynamic top-k. Routing is therefore
    an :math:`O(1)` table lookup followed by a ``sqrt(softplus(.))`` score
    normalization. One warp processes one token.

    The routed weight for a selected expert is
    ``sqrt(softplus(router_logits[token, expert])) / sum_over_routed``. When
    ``num_fused_shared_experts == 1``, an extra shared-expert slot is appended
    with id ``num_routed_experts`` and weight ``1 / routed_scaling_factor``.

    Parameters
    ----------
    router_logits : torch.Tensor
        Router logits of shape ``(num_tokens, num_routed_experts)``, ``float32``.
    input_ids : torch.Tensor
        Vocabulary token ids of shape ``(num_tokens,)``, ``int64``. Indexes the
        ``tid2eid`` table.
    tid2eid : torch.Tensor
        Precomputed token-to-expert table of shape ``(vocab, topk)``, ``int32``.
    num_fused_shared_experts : int
        Number of fused shared experts to append (0 or 1). Default 0.
    routed_scaling_factor : float
        Scaling factor for the shared-expert weight. Default 1.0.
    launch_with_pdl : bool
        Whether to launch with programmatic dependent launch (SM90+). Default
        ``True``.

    Returns
    -------
    topk_weights : torch.Tensor
        Routing weights of shape ``(num_tokens, topk + num_fused_shared_experts)``,
        ``float32``.
    topk_ids : torch.Tensor
        Selected expert ids of shape ``(num_tokens, topk + num_fused_shared_experts)``,
        ``int32``.

    Notes
    -----
    The signature matches SGLang's ``sglang.jit_kernel.deepseek_v4.hash_topk``
    so this can be used as a drop-in replacement. Implements ``MOE-01-HASH``
    from the DSv4 tracker.
    """
    num_tokens = router_logits.shape[0]
    topk = tid2eid.shape[1]
    topk_fused = topk + num_fused_shared_experts

    topk_ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )

    if num_tokens == 0:
        return topk_weights, topk_ids

    launch_with_pdl = launch_with_pdl and device_support_pdl(router_logits.device)

    get_hash_topk_module().hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        float(routed_scaling_factor),
        launch_with_pdl,
    )
    return topk_weights, topk_ids
