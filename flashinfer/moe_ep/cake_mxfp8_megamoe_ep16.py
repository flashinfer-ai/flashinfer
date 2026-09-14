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

# Public entry points for the experimental MXFP8 MegaMoE EP16 backend.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ..api_logging import flashinfer_experimental_api


@dataclass(frozen=True)
class CakeMxfp8MegaMoeEp16Weights:
    """Kernel-ready rank-local MXFP8 weights and packed E8M0 scales."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


@flashinfer_experimental_api(feature="Cake MXFP8 MegaMoE EP16 weight preprocessing")
def preprocess_cake_mxfp8_megamoe_ep16_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> CakeMxfp8MegaMoeEp16Weights:
    """Prepare rank-local BF16 expert weights for the specialized backend."""

    from ..experimental.cake_mxfp8_megamoe_ep16 import preprocess_weights

    return preprocess_weights(w13, w2)


@flashinfer_experimental_api(feature="Cake MXFP8 MegaMoE EP16")
def CakeMxfp8MegaMoeEp16(
    weights: CakeMxfp8MegaMoeEp16Weights,
    topk_ids: torch.Tensor,
    *,
    process_group: dist.ProcessGroup | None = None,
) -> Any:
    """Create a prepared session for the specialized EP16 execution path."""

    from ..experimental.cake_mxfp8_megamoe_ep16 import create_session

    return create_session(weights, topk_ids, process_group=process_group)


__all__ = [
    "CakeMxfp8MegaMoeEp16",
    "CakeMxfp8MegaMoeEp16Weights",
    "preprocess_cake_mxfp8_megamoe_ep16_weights",
]
