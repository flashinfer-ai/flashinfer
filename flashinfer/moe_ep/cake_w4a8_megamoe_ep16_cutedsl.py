# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental fixed-geometry MXFP4/MXFP8 MegaMoE for EP16."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ..api_logging import flashinfer_experimental_api
from .weights import MoEWeightPack


@dataclass(frozen=True)
class CakeW4A8MegaMoeEp16CuteDslWeights:
    """Prepared MXFP4 weight bytes and packed K32 UE8M0 scale planes."""

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor


@flashinfer_experimental_api(feature="Cake W4A8 MegaMoE EP16 CuTeDSL preprocessing")
def preprocess_cake_w4a8_megamoe_ep16_cutedsl_weights(
    weights: MoEWeightPack,
) -> CakeW4A8MegaMoeEp16CuteDslWeights:
    """Prepare BF16 or prequantized MXFP4 weights using native DeepGEMM layout.

    Each rank owns 32 experts, with hidden size 3072 and intermediate size 5120.
    DeepGEMM is required for this setup operation, not for candidate forward.
    """
    from ..experimental.cake_w4a8_megamoe_ep16_cutedsl.backend import preprocess_weights

    return preprocess_weights(weights)


@flashinfer_experimental_api(feature="Cake W4A8 MegaMoE EP16 CuTeDSL")
def CakeW4A8MegaMoeEp16CuteDsl(
    weights: CakeW4A8MegaMoeEp16CuteDslWeights,
    topk_ids: torch.Tensor,
    *,
    process_group: dist.ProcessGroup | None = None,
) -> Any:
    """Collectively prepare a session for one fixed routing tensor.

    Requires 16 SM103a / 152-SM GPUs with NVLink peer mappings. All ranks must
    construct and invoke sessions in the same order. The returned session's
    ``forward(x, router_weights, out=...)`` consumes BF16 inputs and returns
    caller-owned BF16 output. Routing is cloned during setup; create a new
    session to change routing. See the experimental backend README for limits.
    ``session.prepare(x, router_weights, out=...)`` returns an eager forward
    with fixed storage bindings and live tensor contents on the setup stream.
    """
    from ..experimental.cake_w4a8_megamoe_ep16_cutedsl.backend import Session

    return Session(weights, topk_ids, process_group=process_group)


__all__ = [
    "CakeW4A8MegaMoeEp16CuteDsl",
    "CakeW4A8MegaMoeEp16CuteDslWeights",
    "preprocess_cake_w4a8_megamoe_ep16_cutedsl_weights",
]
