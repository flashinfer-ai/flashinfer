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

"""Synthetic problems for the Cake SM120 MegaMoE tuner and benchmark.

Random but correctly typed MXFP4 routed weights (and an FP8 e4m3 shared expert),
Gaussian activations and softmax top-k routing, plus a helper that builds a
:class:`~flashinfer.moe_ep.MoEEpLayer` for one geometry and transport table.  These
measure latency and drive the offline transport-table tuner; correctness is covered by
``tests/moe_ep/test_moe_ep_sm120_w4a8_cake_*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# UE8M0 exponent ranges that keep the random-weight GEMM outputs O(1).
_ROUTED_EXPONENTS = (118, 122)
_SHARED_EXPONENTS = (116, 119)


@dataclass(frozen=True)
class Geometry:
    """Routed-expert geometry of one deployment (``experts`` = physical experts)."""

    hidden: int
    intermediate: int
    experts: int
    top_k: int


def _mxfp4(rows: int, k: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.randint(
        0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=g
    )
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    exponents = torch.randint(
        *_ROUTED_EXPONENTS,
        (rows, k // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    return packed, exponents


def _e4m3(rows: int, k: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.randn(rows, k, device="cuda", generator=g) * 4.0
    codes = values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    exponents = torch.randint(
        *_SHARED_EXPONENTS,
        (rows, k // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    return codes, exponents


def random_weights(
    rank: int, world_size: int, geometry: Geometry, *, shared: bool, seed: int = 0
) -> Any:
    """This rank's weight pack (its ``experts // world_size`` physical experts)."""

    from ...... import MoEWeightPack
    from .cake_weights import CakePrequantizedMoEWeightsWithSharedExpert

    local = geometry.experts // world_size
    H, I = geometry.hidden, geometry.intermediate
    parts = []
    for expert in range(rank * local, (rank + 1) * local):
        g = torch.Generator(device="cuda").manual_seed(seed * 4099 + expert)
        parts.append((*_mxfp4(2 * I, H, g), *_mxfp4(H, I, g)))
    w13, w13_scale, w2, w2_scale = (
        torch.stack([p[i] for p in parts]) for i in range(4)
    )
    routed = (
        w13,
        w2,
        w13_scale.view(torch.float8_e8m0fnu),
        w2_scale.view(torch.float8_e8m0fnu),
    )
    if not shared:
        return MoEWeightPack(*routed)
    g = torch.Generator(device="cuda").manual_seed(seed + 7919)
    s_w13, s_w13_scale = _e4m3(2 * I, H, g)
    s_w2, s_w2_scale = _e4m3(H, I, g)
    return CakePrequantizedMoEWeightsWithSharedExpert(
        *routed,
        shared_w13=s_w13.view(torch.float8_e4m3fn),
        shared_w2=s_w2.view(torch.float8_e4m3fn),
        shared_w13_scale=s_w13_scale.view(torch.float8_e8m0fnu),
        shared_w2_scale=s_w2_scale.view(torch.float8_e8m0fnu),
    )


def random_inputs(rows: int, geometry: Geometry, *, seed: int = 0) -> Any:
    """Gaussian BF16 activations and softmax top-k routing over all experts."""

    from ...... import MoEEpTensors

    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        rows, geometry.hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    logits = torch.randn(rows, geometry.experts, device="cuda", generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), geometry.top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)).float()
    return MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=ids.to(torch.int64).contiguous(),
        topk_weights=weights.contiguous(),
    )


def make_layer(
    rank: int,
    world_size: int,
    weights: Any,
    geometry: Geometry,
    *,
    max_tokens_per_rank: int,
    shared: bool,
    transport_table: Any = "all_gin",
    swiglu_limit: int = 10,
    codec_overflow_blocks: int = 16,
) -> Any:
    """A :class:`~flashinfer.moe_ep.MoEEpLayer` on this backend (EP group = WORLD)."""

    from ...... import BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer
    from .cake_config import (
        SHARED_EXPERT_FP8_IN_KERNEL,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, device=torch.cuda.current_device()
        ),
        fleet_params=FleetParams(
            num_experts=geometry.experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=geometry.hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                intermediate_size=geometry.intermediate,
                top_k=geometry.top_k,
                swiglu_limit=swiglu_limit,
                codec_overflow_blocks=codec_overflow_blocks,
                shared_expert=SHARED_EXPERT_FP8_IN_KERNEL if shared else None,
                transport_table=transport_table,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


__all__ = ["Geometry", "make_layer", "random_inputs", "random_weights"]
