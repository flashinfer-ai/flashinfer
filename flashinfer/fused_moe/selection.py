"""Serving-like inputs for ``MoELayer``'s cross-backend selection.

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

``MoELayer`` times its eligible backends once per (token bucket, routing mode,
eligible set) and caches the winner; a CUDA graph captured afterwards keeps it.
The first call of a bucket is usually a framework's warm-up batch. Frameworks
capture CUDA graphs on static placeholder buffers, often zero-filled, and warm
up on them, so every token in that batch is identical: router logits are
identical across tokens (not across experts) and every token picks the same
``top_k`` experts. Backends rank differently on that routing than on served
traffic, so the candidates are timed on a pack built here instead of the
caller's tensors:

* the call's shapes, dtypes, strides and device are kept, and so are the real
  expert weights (``MoEWeightPack`` is untouched) and the routing bias, which
  is a model parameter;
* activations take the value distribution of the autotuner's profiling inputs:
  standard normal for floating-point payloads, uniform codes for packed integer
  payloads (NVFP4), unit block scales in the activation format's encoding and
  unit per-token scales;
* routing looks like a trained, load-balanced router's: iid standard-normal
  logits for ``FromLogits`` (the kernel routes them with the real bias), or,
  for precomputed routing, each token's ``top_k`` distinct experts drawn
  uniformly (within ``topk_group`` drawn groups for grouped DeepSeekV3
  routing). An integration can pass a recorded per-expert load instead, and
  experts are then drawn in proportion to it.

The values come from a private generator with a fixed seed, so a call geometry
always yields the same pack, in every call, process and rank, and the caller's
global RNG streams are not advanced.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional, Sequence, Union

import torch

from ..tllm_enums import RoutingInputMode, RoutingMethodType
from .api import MoEActivationPack, QuantFormat, RoutingConfig

# Temporary A/B switch: ``1`` makes the implicit selection of ``MoELayer.__call__``
# time the candidates on the caller's first-call tensors again (the behaviour
# before serving-like selection inputs). Read at selection time.
SELECT_ON_CALL_INPUTS_ENV = "FLASHINFER_MOE_SELECT_ON_CALL_INPUTS"

# Seed of the private generator; fixed so every process times the same values.
SELECTION_SEED = 0

# FromLogits routing that follows a recorded expert load: the drawn experts get
# logits near +8 and the others near -8, so softmax and sigmoid scores select
# exactly the drawn set even after a routing bias spanning less than ~0.99 is
# added (sigmoid(8 - 3 * 0.5) - sigmoid(-8 + 3 * 0.5) > 0.99).
_DRAWN_LOGIT = 8.0
_LOGIT_NOISE = 0.5

ExpertLoad = Union[torch.Tensor, Sequence[float]]


def selects_on_call_inputs() -> bool:
    """Whether ``FLASHINFER_MOE_SELECT_ON_CALL_INPUTS=1`` is set now."""
    return os.environ.get(SELECT_ON_CALL_INPUTS_ENV, "") == "1"


def is_capturing(act_pack) -> bool:
    """Whether the current CUDA stream is capturing a graph (CUDA packs only)."""
    hidden = getattr(act_pack, "hidden_states_q", None)
    if not isinstance(hidden, torch.Tensor) or hidden.device.type != "cuda":
        return False
    return bool(torch.cuda.is_current_stream_capturing())


def validate_expert_load(
    expert_load: Optional[ExpertLoad], routing: RoutingConfig
) -> Optional[torch.Tensor]:
    """Return ``expert_load`` as a float64 CPU vector, or raise ``ValueError``.

    The load is the relative number of tokens routed to each global expert, for
    example a histogram of ``topk_ids`` recorded from served traffic. Only its
    proportions matter.
    """
    if expert_load is None:
        return None
    if isinstance(expert_load, torch.Tensor):
        load = expert_load.detach().to(device="cpu", dtype=torch.float64)
    else:
        load = torch.as_tensor(expert_load, dtype=torch.float64)
    if load.shape != (routing.num_experts,):
        raise ValueError(
            f"expert_load must have shape ({routing.num_experts},), one entry per "
            f"global expert; got {tuple(load.shape)}."
        )
    if not bool(torch.isfinite(load).all()) or bool((load < 0).any()):
        raise ValueError("expert_load must be finite and non-negative.")
    if int((load > 0).sum()) < routing.top_k:
        raise ValueError(
            f"expert_load must give a positive load to at least top_k="
            f"{routing.top_k} experts; got {int((load > 0).sum())}."
        )
    return load


def representative_pack(
    act_pack: MoEActivationPack,
    routing: RoutingConfig,
    activation_format: QuantFormat,
    *,
    expert_load: Optional[ExpertLoad] = None,
    seed: int = SELECTION_SEED,
) -> MoEActivationPack:
    """A pack shaped like ``act_pack`` whose values look like served traffic.

    Parameters
    ----------
    act_pack : MoEActivationPack
        The call whose geometry (shapes, dtypes, strides, device, routing input
        mode) and routing bias the selection keeps. Its values are not read.
    routing : RoutingConfig
        The layer's routing configuration (experts, ``top_k``, method, groups).
    activation_format : QuantFormat
        The layer's activation format; it selects the byte encoding of a unit
        block scale (UE8M0 for MX formats, E4M3 otherwise).
    expert_load : torch.Tensor or sequence of float, optional
        Relative per-expert load (``[num_experts]``, global ids). When given,
        each token's ``top_k`` distinct experts are drawn with probability
        proportional to it instead of uniformly.
    seed : int
        Seed of the private generator.
    """
    load = validate_expert_load(expert_load, routing)
    hidden = act_pack.hidden_states_q
    device = hidden.device
    generator = _generator(device, seed)
    num_tokens = act_pack.num_tokens

    fields = {"hidden_states_q": _payload_like(hidden, generator)}
    if act_pack.hidden_states_scale is not None:
        fields["hidden_states_scale"] = _unit_scale_like(
            act_pack.hidden_states_scale, activation_format
        )
    if act_pack.per_token_scale is not None:
        fields["per_token_scale"] = _unit_scale_like(
            act_pack.per_token_scale, activation_format
        )

    mode = act_pack.routing_input_mode
    if mode == RoutingInputMode.FromLogits:
        logits = _layout_like(act_pack.routing_logits)
        if load is None:
            values = _randn(logits.shape, generator, device)
        else:
            ids = _draw_expert_ids(num_tokens, routing, load, generator, device)
            values = _randn(logits.shape, generator, device) * _LOGIT_NOISE
            values = (values - _DRAWN_LOGIT).scatter_add_(
                1, ids, torch.full_like(ids, 2 * _DRAWN_LOGIT, dtype=values.dtype)
            )
        fields["routing_logits"] = logits.copy_(values)
    else:
        ids = _draw_expert_ids(num_tokens, routing, load, generator, device)
        fields["topk_ids"] = _layout_like(act_pack.topk_ids).copy_(ids)
        weights = torch.softmax(_randn(ids.shape, generator, device), dim=-1)
        fields["topk_weights"] = _layout_like(act_pack.topk_weights).copy_(weights)
    # ``replace`` re-runs the pack's own validation on the synthesized fields;
    # routing_bias and every other field are the caller's.
    return dataclasses.replace(act_pack, **fields)


def _generator(device: torch.device, seed: int) -> Optional[torch.Generator]:
    if device.type == "meta":  # shape-only tensors: nothing to draw
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _randn(shape, generator, device) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device)


def _layout_like(tensor: torch.Tensor) -> torch.Tensor:
    """An uninitialized tensor with ``tensor``'s shape, dtype, device and strides."""
    overlapping = any(
        stride == 0 and size > 1
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    if overlapping:  # broadcast views cannot be written element-wise
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    return torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )


def _payload_like(tensor: torch.Tensor, generator) -> torch.Tensor:
    """The autotuner's activation distribution in ``tensor``'s layout."""
    out = _layout_like(tensor)
    if tensor.dtype.is_floating_point:
        # autotuner_initializer_randn, cast to the payload dtype (BF16, FP8, ...).
        return out.copy_(_randn(tensor.shape, generator, tensor.device))
    # Packed integer payloads (two NVFP4 codes per uint8): every code equally
    # likely. Casting normal draws, as the autotuner does, would leave almost
    # every byte at 0 or 1.
    info = torch.iinfo(tensor.dtype)
    low, high = max(info.min, -128), min(info.max, 255) + 1
    codes = torch.randint(
        low,
        high,
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=torch.int32,
    )
    return out.copy_(codes)


def _unit_scale_like(tensor: torch.Tensor, activation_format: QuantFormat):
    """Unit scales in ``tensor``'s layout and encoding.

    One-byte scales are E4M3 (NVFP4 block scales, 1.0 = 0x38) or UE8M0 (MX
    block scales, 1.0 = 127), whatever dtype carries the bytes; wider ones are
    plain floating-point (DeepSeek FP8 block scales, per-token scales).
    """
    out = _layout_like(tensor)
    if tensor.element_size() == 1:
        unit = (
            127 if activation_format in (QuantFormat.MXFP8, QuantFormat.MXFP4) else 0x38
        )
        out.view(torch.uint8).fill_(unit)
    else:
        out.fill_(1.0)
    return out


def _grouped_routing(routing: RoutingConfig) -> Optional[tuple[int, int]]:
    """``(n_group, topk_group)`` when DeepSeekV3 routing limits experts to groups."""
    n_group, topk_group = routing.n_group, routing.topk_group
    if (
        routing.method != RoutingMethodType.DeepSeekV3
        or not n_group
        or not topk_group
        or n_group <= 1
        or topk_group >= n_group
        or routing.num_experts % n_group
    ):
        return None
    return n_group, topk_group


def _gumbel(shape, generator, device) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    eps = torch.finfo(torch.float32).eps
    uniform = torch.rand(shape, generator=generator, device=device)
    return -torch.log(-torch.log(uniform.clamp_(min=tiny, max=1.0 - eps)))


def _draw_expert_ids(
    num_tokens: int,
    routing: RoutingConfig,
    load: Optional[torch.Tensor],
    generator,
    device: torch.device,
) -> torch.Tensor:
    """``[num_tokens, top_k]`` distinct global expert ids per token.

    Gumbel top-k: each token's experts are drawn without replacement with
    probability proportional to ``load`` (uniform when ``None``). Grouped
    DeepSeekV3 routing first draws ``topk_group`` groups in proportion to their
    total load, then the experts inside them.
    """
    num_experts = routing.num_experts
    if load is None:
        log_load = torch.zeros(num_experts, device=device)
    else:
        log_load = load.to(device=device, dtype=torch.float32).log()
    scores = log_load + _gumbel((num_tokens, num_experts), generator, device)
    groups = _grouped_routing(routing)
    if groups is not None:
        n_group, topk_group = groups
        per_group = num_experts // n_group
        group_scores = torch.logsumexp(log_load.view(n_group, per_group), dim=-1)
        group_scores = group_scores + _gumbel((num_tokens, n_group), generator, device)
        drawn = torch.topk(group_scores, topk_group, dim=-1).indices
        allowed = torch.zeros(num_tokens, n_group, dtype=torch.bool, device=device)
        allowed.scatter_(1, drawn, True)
        scores = scores.masked_fill(
            ~allowed.repeat_interleave(per_group, dim=1), float("-inf")
        )
    return torch.topk(scores, routing.top_k, dim=-1).indices
